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

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fastmri.data.subsample import EquiSpacedMaskFunc

from complex_layer_nipun.complex_layers import (
    ComplexConv2d,
    ComplexTransposeConv2d,
    EfficientComplexBatchNorm2d,
    ComplexReLU,
    ComplexAvgPool2d,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import updated_dataloader


# ==========================================
# 1. MODEL ARCHITECTURE (unchanged)
# ==========================================
class ComplexDoubleConv2D(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()
        self.conv1 = ComplexConv2d(input_ch, output_ch, k=3, s=1, p=1)
        self.bn1 = EfficientComplexBatchNorm2d(output_ch)
        self.relu1 = ComplexReLU()

        self.conv2 = ComplexConv2d(output_ch, output_ch, k=3, s=1, p=1)
        self.bn2 = EfficientComplexBatchNorm2d(output_ch)
        self.relu2 = ComplexReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class ComplexDownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super().__init__()
        self.pool = ComplexAvgPool2d(k=kernel_size, s=stride)
        self.double_conv = ComplexDoubleConv2D(input_channels, output_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class ComplexUNET_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_double_conv = ComplexDoubleConv2D(15, 32)
        self.downsample1 = ComplexDownSample(32, 64)
        self.downsample2 = ComplexDownSample(64, 128)
        self.downsample3 = ComplexDownSample(128, 256)
        self.downsample4 = ComplexDownSample(256, 512)

    def forward(self, x):
        f1 = self.first_double_conv(x)
        f2 = self.downsample1(f1)
        f3 = self.downsample2(f2)
        f4 = self.downsample3(f3)
        bottleneck = self.downsample4(f4)
        return bottleneck, [f4, f3, f2, f1]


class ComplexUNET_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcv1 = ComplexTransposeConv2d(512, 256, k=2, s=2, p=0)
        self.double_conv1 = ComplexDoubleConv2D(512, 256)

        self.tcv2 = ComplexTransposeConv2d(256, 128, k=2, s=2, p=0)
        self.double_conv2 = ComplexDoubleConv2D(256, 128)

        self.tcv3 = ComplexTransposeConv2d(128, 64, k=2, s=2, p=0)
        self.double_conv3 = ComplexDoubleConv2D(128, 64)

        self.tcv4 = ComplexTransposeConv2d(64, 32, k=2, s=2, p=0)
        self.double_conv4 = ComplexDoubleConv2D(64, 32)

    def forward(self, bottleneck, skip_conns_list):
        upsample_1 = self.tcv1(bottleneck)
        concat_skip_1 = torch.concat([upsample_1, skip_conns_list[0]], dim=1)
        double_conv1_op = self.double_conv1(concat_skip_1)

        upsample_2 = self.tcv2(double_conv1_op)
        concat_skip_2 = torch.concat([upsample_2, skip_conns_list[1]], dim=1)
        double_conv2_op = self.double_conv2(concat_skip_2)

        upsample_3 = self.tcv3(double_conv2_op)
        concat_skip_3 = torch.concat([upsample_3, skip_conns_list[2]], dim=1)
        double_conv3_op = self.double_conv3(concat_skip_3)

        upsample_4 = self.tcv4(double_conv3_op)
        concat_skip_4 = torch.concat([upsample_4, skip_conns_list[3]], dim=1)
        double_conv4_op = self.double_conv4(concat_skip_4)

        return double_conv4_op


class UNET_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ComplexUNET_encoder()
        self.decoder = ComplexUNET_decoder()
        self.one_cross_one_conv = ComplexConv2d(32, 15, k=1, s=1, p=0)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        final_kspace = self.one_cross_one_conv(final_conv_op)
        return final_kspace + x


# ==========================================
# 2. LOSS FUNCTION & HELPERS
# ==========================================
class ComplexKSpaceLoss(nn.Module):
    def __init__(self, alpha=0.70):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.alpha = alpha

    def forward(self, pred_kspace_sim, target_kspace_sim):
        pred_c = torch.view_as_complex(pred_kspace_sim.contiguous())
        target_c = torch.view_as_complex(target_kspace_sim.contiguous())

        complex_difference = pred_c - target_c

        pred_rss = kspace_to_rss(pred_kspace_sim)
        real_rss = kspace_to_rss(target_kspace_sim)

        rss_max = real_rss.amax(dim=(1, 2, 3), keepdim=True)
        rss_max = torch.clamp(rss_max, min=1e-9)

        pred_img_norm = pred_rss / rss_max
        target_img_norm = real_rss / rss_max
        ssim_val = 1 - self.ssim(pred_img_norm, target_img_norm)

        return (1 - self.alpha) * torch.mean(torch.abs(complex_difference)) + self.alpha * ssim_val


def prepare_input(batch, device):
    return batch["masked_k_space"].to(device, dtype=torch.float32)


def prepare_target_kspace(batch, device):
    return batch["full_k_space"].to(device, dtype=torch.float32)


def normalize_sim_complex_batch(tensor):
    """
    Per-sample normalization by complex magnitude max.
    tensor: (B, C, H, W, 2)
    Returns: normalized tensor, scale (B, 1, 1, 1, 1)
    """
    mag = torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)  # (B,C,H,W)
    scale = mag.amax(dim=(1, 2, 3), keepdim=True)                 # (B,1,1,1)
    scale = torch.clamp(scale, min=1e-9)
    scale = scale.unsqueeze(-1)                                    # (B,1,1,1,1)
    return tensor / scale, scale


def apply_data_consistency(predicted_kspace_sim, original_kspace_sim):
    pred_c = torch.view_as_complex(predicted_kspace_sim.contiguous())
    orig_c = torch.view_as_complex(original_kspace_sim.contiguous())

    mask = (torch.abs(orig_c) != 0.0).float()
    dc_kspace = (orig_c * mask) + (pred_c * (1.0 - mask))

    return torch.view_as_real(dc_kspace)


def kspace_to_rss(kspace_sim):
    kspace_complex = torch.view_as_complex(kspace_sim.contiguous())
    complex_shifted = torch.fft.ifftshift(kspace_complex, dim=(-2, -1))
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
# 3. MAIN DDP TRAINING LOOP
# ==========================================
def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    num_epochs = 50
    batch_size = 6
    learning_rate = 2e-4
    patience = 5
    max_grad_norm = 1.0                                                    

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"

    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/complex_kspace_multicoil_ret_v2.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "complex_kspace_multicoil_ret_v2_metrics.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))

    if rank == 0:
        print(f"Found {len(train_files)} training files and {len(val_files)} validation files.")
        print(f"World size: {world_size}, per-GPU batch size: {batch_size}")

    mask_func = EquiSpacedMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])

    train_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=train_files,
        mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1],
        output_req=[1, 1, 1, 1],
        methods_flags=[0, 0],
    )

    val_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=val_files,
        mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1],
        output_req=[1, 1, 1, 1],
        methods_flags=[0, 0],
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

    loss_func = ComplexKSpaceLoss().to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ← NEW: Learning rate scheduler (was missing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    trigger_times = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_psnr": [],
        "val_psnr": []
    }

    try:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)

            model.train()
            train_loss_sum_local = 0.0
            train_psnr_sum_local = 0.0
            train_batches_local = 0

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", disable=(rank != 0))

            for batch in train_iter:
                input_tensor = prepare_input(batch, device)       # (B,15,H,W,2)
                target_kspace = prepare_target_kspace(batch, device)  # (B,15,H,W,2)

                # ← CHANGED: raw RSS kept un-normalized for PSNR
                target_rss_raw = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)

                # ← CHANGED: normalize by INPUT's magnitude max
                input_norm, scale = normalize_sim_complex_batch(input_tensor)
                target_norm = target_kspace / scale

                predictions = model(input_norm)
                predictions = apply_data_consistency(predictions, input_norm)

                loss = loss_func(predictions, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # ← NEW
                optimizer.step()

                # ← CHANGED: denormalize prediction, then normalize both by RSS max for PSNR
                with torch.no_grad():
                    pred_denorm = predictions * scale
                    final_images_pred = kspace_to_rss(pred_denorm)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(final_images_pred / rss_max, target_rss_raw / rss_max)

                train_loss_sum_local += loss.item()
                train_psnr_sum_local += psnr_val.item()
                train_batches_local += 1

                if rank == 0 and train_batches_local > 0:
                    train_iter.set_postfix(
                        avg_kspace_loss=train_loss_sum_local / train_batches_local,
                        avg_ispace_psnr=train_psnr_sum_local / train_batches_local,
                    )

            train_loss_sum_global = reduce_scalar(train_loss_sum_local, device)
            train_psnr_sum_global = reduce_scalar(train_psnr_sum_local, device)
            train_batches_global = reduce_scalar(float(train_batches_local), device)

            avg_train_loss = train_loss_sum_global / max(train_batches_global, 1.0)
            avg_train_psnr = train_psnr_sum_global / max(train_batches_global, 1.0)

            # ---------- VALIDATION ----------
            model.eval()
            val_loss_sum_local = 0.0
            val_psnr_sum_local = 0.0
            val_batches_local = 0

            val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", disable=(rank != 0))

            with torch.no_grad():
                for batch in val_iter:
                    input_tensor = prepare_input(batch, device)
                    target_kspace = prepare_target_kspace(batch, device)

                    # ← CHANGED: raw RSS kept un-normalized
                    target_rss_raw = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)

                    # ← CHANGED: normalize by INPUT's magnitude max
                    input_norm, scale = normalize_sim_complex_batch(input_tensor)
                    target_norm = target_kspace / scale

                    predictions = model(input_norm)
                    predictions = apply_data_consistency(predictions, input_norm)

                    val_loss = loss_func(predictions, target_norm)

                    # ← CHANGED: denormalize prediction, proper PSNR
                    pred_denorm = predictions * scale
                    final_images_pred = kspace_to_rss(pred_denorm)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(final_images_pred / rss_max, target_rss_raw / rss_max)

                    val_loss_sum_local += val_loss.item()
                    val_psnr_sum_local += psnr_val.item()
                    val_batches_local += 1

                    if rank == 0 and val_batches_local > 0:
                        val_iter.set_postfix(
                            avg_val_loss=val_loss_sum_local / val_batches_local,
                            avg_psnr=val_psnr_sum_local / val_batches_local,
                        )

            val_loss_sum_global = reduce_scalar(val_loss_sum_local, device)
            val_psnr_sum_global = reduce_scalar(val_psnr_sum_local, device)
            val_batches_global = reduce_scalar(float(val_batches_local), device)

            avg_val_loss = val_loss_sum_global / max(val_batches_global, 1.0)
            avg_val_psnr = val_psnr_sum_global / max(val_batches_global, 1.0)

            scheduler.step(avg_val_loss)                                   # ← NEW

            stop_flag = 0
            if rank == 0:
                print(
                    f"\nEpoch {epoch + 1} Summary | "
                    f"Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.2f} dB | "
                    f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"            # ← NEW
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
                    print(f"--> Best model saved! (Val Loss: {best_val_loss:.4f})\n")
                else:
                    trigger_times += 1
                    print(f"--> No improvement. Early stopping trigger: {trigger_times} / {patience}\n")

                if trigger_times >= patience:
                    print(f"Early stopping triggered! Training halted at epoch {epoch + 1}.")
                    stop_flag = 1

            sync_state = torch.tensor(
                [best_val_loss if rank == 0 else 0.0, float(trigger_times if rank == 0 else 0), float(stop_flag)],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(sync_state, src=0)

            best_val_loss = float(sync_state[0].item())
            trigger_times = int(sync_state[1].item())
            stop_flag = int(sync_state[2].item())

            if stop_flag == 1:
                break

    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()