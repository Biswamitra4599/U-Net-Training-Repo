#!/usr/bin/env python3
"""
DDP Image-Space U-Net Training for MRI Reconstruction
Input:  masked_rss_combined (B, 1, H, W)
Output: full_rss_combined   (B, 1, H, W)
Launch: torchrun --nproc_per_node=2 train_unet_image_ddp.py
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
# MODEL ARCHITECTURE (channels: 48→96→184→360→720)
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
        self.first_double_conv = doubleConv2D(1, 48)      # 1-channel RSS input
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
        self.one_Cross_one_conv = nn.Conv2d(48, 1, kernel_size=1, stride=1)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        final_image = self.one_Cross_one_conv(final_conv_op)
        return final_image


# ==========================================
# LOSS FUNCTION
# ==========================================
    
class SpecialLossFunc(nn.Module):
    """
    L1 + SSIM loss with internal re-normalization.
    
    Inputs arrive normalized by INPUT max (so target can be > 1.0).
    Inside the loss, we re-normalize by TARGET max to ensure
    SSIM sees data in [0, 1] range with data_range=1.0.
    
    This is training-only — no issue using target here since
    loss is never called at inference time.
    """
    def __init__(self, alpha=0.86):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, preds, targets):
        # ── L1: directly on input-normalized outputs (no extra scaling) ──
        l1_loss = self.l1_loss(preds, targets)

        # ── SSIM: renormalize to [0, ~1] because SSIM needs data_range=1.0 ──
        t_max = targets.amax(dim=(1, 2, 3), keepdim=True)
        t_max = torch.clamp(t_max, min=1e-9)

        pred_norm = preds / t_max
        target_norm = targets / t_max

        ssim_val = self.ssim(pred_norm, target_norm)
        ssim_loss = 1.0 - ssim_val

        return (1.0 - self.alpha) * l1_loss + self.alpha * ssim_loss


# ==========================================
# HELPERS
# ==========================================
def normalize_by_input(input_tensor, target_tensor):
    """
    Per-sample normalization by INPUT max.
    Returns: (input_norm, target_norm, scale)
    
    At inference: just do input / input.amax(...) → model → output * scale
    """
    scale = input_tensor.amax(dim=(1, 2, 3), keepdim=True)   # (B, 1, 1, 1)
    scale = torch.clamp(scale, min=1e-9)

    input_norm = input_tensor / scale
    target_norm = target_tensor / scale     # same scale, target can be > 1.0

    return input_norm, target_norm, scale


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

    # ── Hyperparameters ──
    num_epochs = 50
    batch_size = 6
    learning_rate = 2e-4
    patience = 5

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/unet_ispace_model_rss_ddp.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "training_metrics_ispace_rss_ddp.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))

    if rank == 0:
        print(f"Found {len(train_files)} train / {len(val_files)} val files.")
        print(f"World size: {world_size}, per-GPU batch: {batch_size}")
        print("Training: Image-Space U-Net (DDP, input-normalized)")

    mask_func = EquiSpacedMaskFunc(
        center_fractions=[0.08, 0.04], accelerations=[4, 8]
    )

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
        sampler=train_sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True,
    )

    model = UNET_final().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_func = SpecialLossFunc(alpha=0.86).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    trigger_times = 0
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    try:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)

            # ==================== TRAINING ====================
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
                input_tensor = batch['masked_rss_combined'].unsqueeze(1).to(
                    device, dtype=torch.float32
                )
                target_tensor = batch['full_rss_combined'].unsqueeze(1).to(
                    device, dtype=torch.float32
                )

                # ── Per-sample normalization by INPUT max ──
                input_norm, target_norm, scale = normalize_by_input(
                    input_tensor, target_tensor
                )

                # ── Forward ──
                predictions = model(input_norm)

                # ── Loss (internally re-normalizes for SSIM) ──
                loss = loss_func(predictions, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # ── PSNR: denormalize → normalize by target max ──
                with torch.no_grad():
                    pred_denorm = predictions * scale                 # back to original scale
                    rss_max = target_tensor.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(
                        pred_denorm / rss_max,
                        target_tensor / rss_max
                    )

                train_loss_sum += loss.item()
                train_psnr_sum += psnr_val.item()
                train_batches += 1

                if rank == 0 and train_batches > 0:
                    train_iter.set_postfix(
                        loss=f"{train_loss_sum / train_batches:.4f}",
                        psnr=f"{train_psnr_sum / train_batches:.2f}",
                    )

            train_loss_global = reduce_scalar(train_loss_sum, device)
            train_psnr_global = reduce_scalar(train_psnr_sum, device)
            train_n = reduce_scalar(float(train_batches), device)
            avg_train_loss = train_loss_global / max(train_n, 1.0)
            avg_train_psnr = train_psnr_global / max(train_n, 1.0)

            # ==================== VALIDATION ====================
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
                    input_tensor = batch['masked_rss_combined'].unsqueeze(1).to(
                        device, dtype=torch.float32
                    )
                    target_tensor = batch['full_rss_combined'].unsqueeze(1).to(
                        device, dtype=torch.float32
                    )

                    input_norm, target_norm, scale = normalize_by_input(
                        input_tensor, target_tensor
                    )

                    predictions = model(input_norm)
                    val_loss = loss_func(predictions, target_norm)

                    pred_denorm = predictions * scale
                    rss_max = target_tensor.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(
                        pred_denorm / rss_max,
                        target_tensor / rss_max
                    )

                    val_loss_sum += val_loss.item()
                    val_psnr_sum += psnr_val.item()
                    val_batches += 1

                    if rank == 0 and val_batches > 0:
                        val_iter.set_postfix(
                            loss=f"{val_loss_sum / val_batches:.4f}",
                            psnr=f"{val_psnr_sum / val_batches:.2f}",
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