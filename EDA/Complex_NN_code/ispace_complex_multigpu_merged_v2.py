#!/usr/bin/env python3
"""
Complex U-Net training on RSS images using simulated-complex (B,C,H,W,2) format.
Input:  RSS(masked_multicoil) as 1-channel simulated complex (real=RSS, imag=0)
Output: Full RSS as 1-channel simulated complex
Loss:   L1 + SSIM on magnitude images

Launch: torchrun --nproc_per_node=2 complex_rss_ddp_v2.py
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

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fastmri.data.subsample import EquiSpacedMaskFunc

# ← CHANGED: import from updated package
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
# 1. MODEL ARCHITECTURE
#    All tensors flow as (B, C, H, W, 2) float32
# ==========================================
class ComplexDoubleConv2D(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()
        self.conv1 = ComplexConv2d(input_ch, output_ch, k=3, s=1, p=1)
        self.bn1 = EfficientComplexBatchNorm2d(output_ch)              # ← CHANGED
        self.relu1 = ComplexReLU()
        self.conv2 = ComplexConv2d(output_ch, output_ch, k=3, s=1, p=1)
        self.bn2 = EfficientComplexBatchNorm2d(output_ch)              # ← CHANGED
        self.relu2 = ComplexReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class ComplexDownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super().__init__()
        self.pool = ComplexAvgPool2d(k=kernel_size, s=stride)
        self.double_conv = ComplexDoubleConv2D(input_channels, output_channels)

    def forward(self, x):
        return self.double_conv(self.pool(x))


class ComplexUNET_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_double_conv = ComplexDoubleConv2D(1, 32)
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
        # torch.cat on (B,C,H,W,2) along dim=1 → (B, C1+C2, H,W,2) ✓
        x = self.double_conv1(torch.cat([self.tcv1(bottleneck), skip_conns_list[0]], dim=1))
        x = self.double_conv2(torch.cat([self.tcv2(x), skip_conns_list[1]], dim=1))
        x = self.double_conv3(torch.cat([self.tcv3(x), skip_conns_list[2]], dim=1))
        x = self.double_conv4(torch.cat([self.tcv4(x), skip_conns_list[3]], dim=1))
        return x


class UNET_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ComplexUNET_encoder()
        self.decoder = ComplexUNET_decoder()
        self.one_cross_one_conv = ComplexConv2d(32, 1, k=1, s=1, p=0)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        residual = self.one_cross_one_conv(final_conv_op)
        return x + residual   # (B,1,H,W,2) + (B,1,H,W,2) ✓


# ==========================================
# 2. LOSS FUNCTION & HELPERS
# ==========================================

# ← CHANGED: returns (B,1,H,W,2) float32 instead of native complex
def prepare_rss_input(batch, device):
    """
    Compute RSS from masked multi-coil i-space, return as
    simulated complex (real=RSS, imag=0) with shape (B, 1, H, W, 2).
    """
    x = batch["masked_i_space"].to(device, dtype=torch.float32)   # (B,coils,H,W,2)
    mag_sq = x[..., 0] ** 2 + x[..., 1] ** 2                      # (B,coils,H,W)
    rss = torch.sqrt(torch.sum(mag_sq, dim=1, keepdim=True))       # (B,1,H,W)
    zeros = torch.zeros_like(rss)
    return torch.stack([rss, zeros], dim=-1)                        # (B,1,H,W,2)


# ← CHANGED: returns (B,1,H,W,2) float32 instead of native complex
def prepare_rss_target(batch, device):
    """
    Load ground-truth RSS image, return as simulated complex
    (real=RSS, imag=0) with shape (B, 1, H, W, 2).
    """
    rss = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)  # (B,1,H,W)
    zeros = torch.zeros_like(rss)
    return torch.stack([rss, zeros], dim=-1)                        # (B,1,H,W,2)


# ← CHANGED: normalize simulated complex (B,C,H,W,2) by magnitude max
def normalize_sim_complex_batch(tensor):
    """
    Per-sample normalization by complex magnitude max.
    tensor: (B, C, H, W, 2)
    Returns: normalized tensor, scale (B, 1, 1, 1, 1)
    """
    mag = torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)   # (B,C,H,W)
    scale = mag.amax(dim=(1, 2, 3), keepdim=True)                  # (B,1,1,1)
    scale = torch.clamp(scale, min=1e-9)
    scale = scale.unsqueeze(-1)                                     # (B,1,1,1,1)
    return tensor / scale, scale


# ← NEW: extract magnitude from simulated complex
def sim_complex_magnitude(tensor):
    """
    (B, C, H, W, 2) → (B, C, H, W) magnitude
    """
    return torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)


def reduce_scalar(value, device):
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()

class SpecialLossFunc(nn.Module):
    """
    Hybrid L1 + SSIM on magnitude of simulated complex output.
    """

    def __init__(self, alpha=0.70):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, preds, targets):
        # preds, targets: (B, 1, H, W, 2)

        # 1. Extract magnitude: (B, 1, H, W)
        pred_mag = sim_complex_magnitude(preds)
        target_mag = sim_complex_magnitude(targets)

        # 2. L1 Loss directly on the safely scaled tensors
        l1_loss = self.l1(pred_mag, target_mag)

        # 3. Normalize to [0, ~1] ONLY for SSIM
        mag_max = target_mag.amax(dim=(1, 2, 3), keepdim=True)
        mag_max = torch.clamp(mag_max, min=1e-9)

        pred_ssim_norm = pred_mag / mag_max
        target_ssim_norm = target_mag / mag_max

        # 4. SSIM Loss
        ssim_val = self.ssim(pred_ssim_norm, target_ssim_norm)
        ssim_loss = 1.0 - ssim_val

        return (1.0 - self.alpha) * l1_loss + self.alpha * ssim_loss


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
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/complex_rss_model_ispace_merged_v2.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "training_metrics_ispace_rss_merged_v2.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))

    if rank == 0:
        print(f"Found {len(train_files)} train / {len(val_files)} val files.")
        print(f"World size: {world_size}, per-GPU batch: {batch_size}")
        print("Training: Complex U-Net (simulated-complex) on RSS with DDP")

    mask_func = EquiSpacedMaskFunc(
        center_fractions=[0.08, 0.04], accelerations=[4, 8]
    )

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
        sampler=train_sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True,
    )

    model = UNET_final().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_func = SpecialLossFunc(alpha=0.70).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

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
                # ← CHANGED: returns (B,1,H,W,2) float32
                input_sim = prepare_rss_input(batch, device)
                target_sim = prepare_rss_target(batch, device)
                target_rss_raw = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)

                # ← CHANGED: normalize by INPUT magnitude, scale (B,1,1,1,1)
                input_norm, scale = normalize_sim_complex_batch(input_sim)
                target_norm = target_sim / scale

                # Forward: (B,1,H,W,2) → (B,1,H,W,2)
                pred_norm = model(input_norm)

                loss = loss_func(pred_norm, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # ← CHANGED: denormalize, extract magnitude for PSNR
                with torch.no_grad():
                    pred_denorm = pred_norm * scale                             # (B,1,H,W,2)
                    pred_rss = sim_complex_magnitude(pred_denorm)               # (B,1,H,W)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(pred_rss / rss_max, target_rss_raw / rss_max)

                train_loss_sum += loss.item()
                train_psnr_sum += psnr_val.item()
                train_batches += 1

                if rank == 0 and train_batches > 0:
                    train_iter.set_postfix(
                        loss=f"{train_loss_sum/train_batches:.4f}",
                        psnr=f"{train_psnr_sum/train_batches:.2f}",
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
                    input_sim = prepare_rss_input(batch, device)
                    target_sim = prepare_rss_target(batch, device)
                    target_rss_raw = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)

                    # ← CHANGED: same input-based normalization
                    input_norm, scale = normalize_sim_complex_batch(input_sim)
                    target_norm = target_sim / scale

                    pred_norm = model(input_norm)
                    val_loss = loss_func(pred_norm, target_norm)

                    # ← CHANGED: same PSNR computation
                    pred_denorm = pred_norm * scale
                    pred_rss = sim_complex_magnitude(pred_denorm)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(pred_rss / rss_max, target_rss_raw / rss_max)

                    val_loss_sum += val_loss.item()
                    val_psnr_sum += psnr_val.item()
                    val_batches += 1

                    if rank == 0 and val_batches > 0:
                        val_iter.set_postfix(
                            loss=f"{val_loss_sum/val_batches:.4f}",
                            psnr=f"{val_psnr_sum/val_batches:.2f}",
                        )

            val_loss_global = reduce_scalar(val_loss_sum, device)
            val_psnr_global = reduce_scalar(val_psnr_sum, device)
            val_n = reduce_scalar(float(val_batches), device)
            avg_val_loss = val_loss_global / max(val_n, 1.0)
            avg_val_psnr = val_psnr_global / max(val_n, 1.0)

            scheduler.step(avg_val_loss)

            stop_flag = 0
            if rank == 0:
                print(
                    f"\nEpoch {epoch+1} | "
                    f"Train Loss: {avg_train_loss:.4f}  PSNR: {avg_train_psnr:.2f} dB | "
                    f"Val Loss: {avg_val_loss:.4f}  PSNR: {avg_val_psnr:.2f} dB | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
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
                    print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})\n")
                else:
                    trigger_times += 1
                    print(f"  ✗ No improvement ({trigger_times}/{patience})\n")
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


if __name__ == "__main__":
    main()