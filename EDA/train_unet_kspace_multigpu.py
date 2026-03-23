"""
DDP U-Net K-Space Training Script for knee-MRI Reconstruction
Launch: torchrun --nproc_per_node=2 kspace_real_ddp.py
"""

import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from fastmri.data.subsample import EquiSpacedMaskFunc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import updated_dataloader


# ==========================================
# MODEL ARCHITECTURE 
# ==========================================
class doubleConv2D(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
            doubleConv2D(input_ch=input_channels, output_ch=output_channels)
        )

    def forward(self, x):
        return self.downsample(x)


class UNET_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #        self.first_double_conv = ComplexDoubleConv2D(15, 32)
        # self.downsample1 = ComplexDownSample(32, 64)
        # self.downsample2 = ComplexDownSample(64, 128)
        # self.downsample3 = ComplexDownSample(128, 256)
        # self.downsample4 = ComplexDownSample(256, 512)
        self.first_double_conv = doubleConv2D(30, 48)
        self.downsample1 = DownSample(48, 96)
        self.downsample2 = DownSample(96, 184)
        self.downsample3 = DownSample(184, 360)
        self.downsample4 = DownSample(360, 720)

    def forward(self, x):
        f1 = self.first_double_conv(x)
        f2 = self.downsample1(f1)
        f3 = self.downsample2(f2)
        f4 = self.downsample3(f3)
        bottleneck = self.downsample4(f4)
        return bottleneck, [f4, f3, f2, f1]


class UNET_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcv1 = nn.ConvTranspose2d(720, 360, kernel_size=2, stride=2)
        self.double_conv1 = doubleConv2D(720, 360)

        self.tcv2 = nn.ConvTranspose2d(360, 184, kernel_size=2, stride=2)
        self.double_conv2 = doubleConv2D(368, 184)

        self.tcv3 = nn.ConvTranspose2d(184, 96, kernel_size=2, stride=2)
        self.double_conv3 = doubleConv2D(192, 96)

        self.tcv4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.double_conv4 = doubleConv2D(96, 48)

    def forward(self, bottleneck, skip_conns_list):
        upsample_1 = self.tcv1(bottleneck)
        concat_skip_1 = torch.cat([upsample_1, skip_conns_list[0]], dim=1)
        double_conv1_op = self.double_conv1(concat_skip_1)

        upsample_2 = self.tcv2(double_conv1_op)
        concat_skip_2 = torch.cat([upsample_2, skip_conns_list[1]], dim=1)
        double_conv2_op = self.double_conv2(concat_skip_2)

        upsample_3 = self.tcv3(double_conv2_op)
        concat_skip_3 = torch.cat([upsample_3, skip_conns_list[2]], dim=1)
        double_conv3_op = self.double_conv3(concat_skip_3)

        upsample_4 = self.tcv4(double_conv3_op)
        concat_skip_4 = torch.cat([upsample_4, skip_conns_list[3]], dim=1)
        double_conv4_op = self.double_conv4(concat_skip_4)

        return double_conv4_op


class UNET_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNET_encoder()
        self.decoder = UNET_decoder()
        self.one_Cross_one_conv = nn.Conv2d(48, 30, kernel_size=1, stride=1)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        final_kspace = self.one_Cross_one_conv(final_conv_op)
        return final_kspace


# ==========================================
#  FUNCTION — matches ComplexKSpaceLoss
# ==========================================
class RealKSpaceLoss(nn.Module):
    def __init__(self, num_coils=15):
        super().__init__()
        self.num_coils = num_coils
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred_kspace, target_kspace):
        # ── K-space L1 on complex magnitude of difference ──
        # (B, 30, H, W) → (B, 15, 2, H, W)
        B, _, H, W = pred_kspace.shape
        pred_ri = pred_kspace.reshape(B, self.num_coils, 2, H, W)
        tgt_ri = target_kspace.reshape(B, self.num_coils, 2, H, W)

        diff_real = pred_ri[:, :, 0] - tgt_ri[:, :, 0]
        diff_imag = pred_ri[:, :, 1] - tgt_ri[:, :, 1]
        complex_mag_diff = torch.sqrt(diff_real ** 2 + diff_imag ** 2 + 1e-12)
        l1_kspace = torch.mean(complex_mag_diff)

        # ── Image-space SSIM on RSS ──
        pred_rss = kspace_to_rss(pred_kspace)
        target_rss = kspace_to_rss(target_kspace)

        rss_max = target_rss.amax(dim=(1, 2, 3), keepdim=True)
        rss_max[rss_max == 0] = 1.0

        pred_img_norm = pred_rss / rss_max
        target_img_norm = target_rss / rss_max

        ssim_loss = 1.0 - self.ssim(pred_img_norm, target_img_norm)

        return l1_kspace + ssim_loss


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def prepare_input(batch, device):
    """(B, coils, H, W, 2) → (B, coils*2, H, W)"""
    x = batch['masked_k_space'].to(device, dtype=torch.float32)
    B, C, H, W, cr = x.shape
    x = x.permute(0, 1, 4, 2, 3)
    x = x.reshape(B, C * cr, H, W)
    return x


def prepare_target_kspace(batch, device):
    """(B, coils, H, W, 2) → (B, coils*2, H, W)  — full k-space target"""
    x = batch['full_k_space'].to(device, dtype=torch.float32)
    B, C, H, W, cr = x.shape
    x = x.permute(0, 1, 4, 2, 3)
    x = x.reshape(B, C * cr, H, W)
    return x


def compute_kspace_max(kspace_real, num_coils=15):
    """
    Compute per-sample complex magnitude max from real-valued k-space.
    (B, 30, H, W) → (B, 1, 1, 1)
    Equivalent to: torch.abs(complex_kspace).amax(dim=(1,2,3), keepdim=True)
    """
    B, _, H, W = kspace_real.shape
    reshaped = kspace_real.reshape(B, num_coils, 2, H, W)
    magnitude = torch.sqrt(reshaped[:, :, 0] ** 2 + reshaped[:, :, 1] ** 2)
    kmax = magnitude.amax(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
    kmax[kmax == 0] = 1.0
    return kmax


def apply_data_consistency(predicted_kspace, original_kspace):
    """k_final = M ⊙ k_masked + (1 - M) ⊙ k_predicted"""
    mask = (original_kspace != 0.0).float()
    dc_kspace = (original_kspace * mask) + (predicted_kspace * (1.0 - mask))
    return dc_kspace


def kspace_to_rss(kspace_tensor, num_coils=15):
    """(B, 30, H, W) → (B, 1, H, W) RSS image."""
    B, _, H, W = kspace_tensor.shape
    reshaped = kspace_tensor.reshape(B, num_coils, 2, H, W)

    complex_kspace = torch.complex(
        real=reshaped[:, :, 0, :, :],
        imag=reshaped[:, :, 1, :, :]
    )

    complex_shifted = torch.fft.ifftshift(complex_kspace, dim=(-2, -1))
    image_complex = torch.fft.ifft2(complex_shifted, dim=(-2, -1), norm="ortho")
    image_complex = torch.fft.fftshift(image_complex, dim=(-2, -1))

    image_mag = torch.abs(image_complex)
    rss = torch.sqrt(torch.sum(image_mag ** 2, dim=1))
    return rss.unsqueeze(1)


def reduce_scalar(value, device):
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ==========================================
# MAIN DDP TRAINING LOOP
# ==========================================
def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    num_epochs = 50
    batch_size = 6                  # matched to complex U-Net
    learning_rate = 2e-4
    patience = 5

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/real_unet_kspace_multichannel_model_ddp.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "training_metrics_real_kspace_multichannel_ddp.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))

    if rank == 0:
        print(f"Found {len(train_files)} train / {len(val_files)} val files.")
        print(f"World size: {world_size}, per-GPU batch: {batch_size}")
        print("Training: Real K-Space U-Net (Complex-matched normalization & loss)")

    mask_func = EquiSpacedMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])

    train_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=train_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )
    val_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=val_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )

    train_sampler = DistributedSampler(train_data, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_data, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False,
        sampler=train_sampler, num_workers=8, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=8, pin_memory=True,
    )

    model = UNET_final().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_func = RealKSpaceLoss().to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    trigger_times = 0
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    try:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            train_loss_sum = 0.0
            train_psnr_sum = 0.0
            train_batches = 0

            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                disable=(rank != 0),
            )

            for batch in train_iter:
                input_tensor = prepare_input(batch, device)           # (B, 30, H, W)
                target_kspace = prepare_target_kspace(batch, device)  # (B, 30, H, W)
                target_rss = batch['full_rss_combined'].unsqueeze(1).to(
                    device, dtype=torch.float32
                )

                # ── Per-sample k-space magnitude max ──
                batch_maxes = compute_kspace_max(target_kspace)  # (B, 1, 1, 1)

                input_tensor = input_tensor / batch_maxes
                target_kspace = target_kspace / batch_maxes
                target_rss = target_rss / batch_maxes

                # ── Forward → Data Consistency ──
                predictions = model(input_tensor)
                predictions = apply_data_consistency(predictions, input_tensor)

                # ── Loss in k-space  ──
                loss = loss_func(predictions, target_kspace)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # ── PSNR on RSS images ──
                with torch.no_grad():
                    final_images_pred = kspace_to_rss(predictions)
                    psnr_val = psnr_module(final_images_pred, target_rss)

                train_loss_sum += loss.item()
                train_psnr_sum += psnr_val.item()
                train_batches += 1

                if rank == 0 and train_batches > 0:
                    train_iter.set_postfix(
                        avg_kspace_loss=f"{train_loss_sum / train_batches:.4f}",
                        avg_ispace_psnr=f"{train_psnr_sum / train_batches:.2f}",
                    )

            train_loss_global = reduce_scalar(train_loss_sum, device)
            train_psnr_global = reduce_scalar(train_psnr_sum, device)
            train_n = reduce_scalar(float(train_batches), device)
            avg_train_loss = train_loss_global / max(train_n, 1.0)
            avg_train_psnr = train_psnr_global / max(train_n, 1.0)

            #validation
            model.eval()
            val_loss_sum = 0.0
            val_psnr_sum = 0.0
            val_batches = 0

            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                disable=(rank != 0),
            )

            with torch.no_grad():
                for batch in val_iter:
                    input_tensor = prepare_input(batch, device)
                    target_kspace = prepare_target_kspace(batch, device)
                    target_rss = batch['full_rss_combined'].unsqueeze(1).to(
                        device, dtype=torch.float32
                    )

                    batch_maxes = compute_kspace_max(target_kspace)

                    input_tensor = input_tensor / batch_maxes
                    target_kspace = target_kspace / batch_maxes
                    target_rss = target_rss / batch_maxes

                    predictions = model(input_tensor)
                    predictions = apply_data_consistency(predictions, input_tensor)

                    val_loss = loss_func(predictions, target_kspace)

                    final_images_pred = kspace_to_rss(predictions)
                    psnr_val = psnr_module(final_images_pred, target_rss)

                    val_loss_sum += val_loss.item()
                    val_psnr_sum += psnr_val.item()
                    val_batches += 1

                    if rank == 0 and val_batches > 0:
                        val_iter.set_postfix(
                            avg_val_loss=f"{val_loss_sum / val_batches:.4f}",
                            avg_psnr=f"{val_psnr_sum / val_batches:.2f}",
                        )

            val_loss_global = reduce_scalar(val_loss_sum, device)
            val_psnr_global = reduce_scalar(val_psnr_sum, device)
            val_n = reduce_scalar(float(val_batches), device)
            avg_val_loss = val_loss_global / max(val_n, 1.0)
            avg_val_psnr = val_psnr_global / max(val_n, 1.0)

            # ── Logging, checkpointing, early stopping ──
            stop_flag = 0
            if rank == 0:
                print(
                    f"\nEpoch {epoch+1} Summary | "
                    f"Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.2f} dB | "
                    f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB"
                )

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["train_psnr"].append(avg_train_psnr)
                history["val_psnr"].append(avg_val_psnr)
                with open(metrics_path, "w") as f:
                    json.dump(history, f, indent=4)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    trigger_times = 0
                    torch.save(model.module.state_dict(), saved_model_path)
                    print(f"  --> Best model saved! (Val Loss: {best_val_loss:.4f})\n")
                else:
                    trigger_times += 1
                    print(f"  --> No improvement ({trigger_times}/{patience})\n")
                    if trigger_times >= patience:
                        print(f"Early stopping at epoch {epoch+1}.")
                        stop_flag = 1

            sync = torch.tensor(
                [best_val_loss if rank == 0 else 0.0,
                 float(trigger_times if rank == 0 else 0),
                 float(stop_flag)],
                dtype=torch.float64, device=device,
            )
            dist.broadcast(sync, src=0)
            best_val_loss = sync[0].item()
            trigger_times = int(sync[1].item())
            if int(sync[2].item()) == 1:
                break

    finally:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Model saved to: {saved_model_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()