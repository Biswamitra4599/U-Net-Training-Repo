#!/usr/bin/env python3
"""
DDP U-Net I-Space Training Script for MRI Reconstruction
Trains on 30-channel real representation of multi-coil complex i-space

Launch: torchrun --nproc_per_node=2 ispace_real_ddp.py
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
# MODEL ARCHITECTURE (identical)
# ==========================================
class doubleConv2D(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
            doubleConv2D(input_channels, output_channels),
        )

    def forward(self, x):
        return self.downsample(x)


class UNET_encoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.one_cross_one_conv = nn.Conv2d(48, 30, kernel_size=1, stride=1)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        residual = self.one_cross_one_conv(final_conv_op)
        return x + residual


# ==========================================
# HELPER FUNCTIONS (identical)
# ==========================================
def prepare_input(batch_tensor, device):
    """(B, coils, H, W, 2) → (B, coils*2, H, W)"""
    x = batch_tensor.to(device, dtype=torch.float32)
    B, C, H, W, cr = x.shape
    x = x.permute(0, 1, 4, 2, 3)
    x = x.reshape(B, C * cr, H, W)
    return x


def normalize_batch(tensor):
    """Per-sample normalization by absolute max."""
    scale = tensor.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(scale, min=1e-9)
    return tensor / scale, scale


def real30ch_to_rss(tensor_30ch, num_coils=15):
    """(B, 30, H, W) → (B, 1, H, W) RSS magnitude."""
    B, _, H, W = tensor_30ch.shape
    reshaped = tensor_30ch.reshape(B, num_coils, 2, H, W)
    mag_sq = reshaped[:, :, 0] ** 2 + reshaped[:, :, 1] ** 2
    rss = torch.sqrt(torch.sum(mag_sq, dim=1, keepdim=True))
    return rss


def reduce_scalar(value, device):
    """All-reduce a scalar across all DDP ranks."""
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ==========================================
# LOSS FUNCTION (identical)
# ==========================================
class SpecialLossFunc(nn.Module):
    def __init__(self, alpha=0.86):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, preds, targets):
        l1_loss = self.l1_loss(preds, targets)

        pred_rss = real30ch_to_rss(preds)
        target_rss = real30ch_to_rss(targets)

        rss_max = target_rss.amax(dim=(1, 2, 3), keepdim=True)
        rss_max = torch.clamp(rss_max, min=1e-9)

        ssim_val = self.ssim(pred_rss / rss_max, target_rss / rss_max)
        ssim_loss = 1.0 - ssim_val

        return (1.0 - self.alpha) * l1_loss + self.alpha * ssim_loss


# ==========================================
# MAIN DDP TRAINING LOOP
# ==========================================
def main():
    # ── DDP initialization ──
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
    max_grad_norm = 1.0

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/multi_coil_ispace_real_ddp_retrain.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "training_metrics_ispace_real_multicoil_ddp.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))

    if rank == 0:
        print(f"Found {len(train_files)} train / {len(val_files)} val files.")
        print(f"World size: {world_size}, per-GPU batch: {batch_size}")
        print("Training: Real 30-ch I-Space U-Net with DDP")

    mask_func = EquiSpacedMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8])

    train_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=train_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )
    val_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=val_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )

    # ── DDP samplers (replace shuffle=True) ──
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

    # ── Model wrapped with DDP ──
    model = UNET_final().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_func = SpecialLossFunc(alpha=0.86).to(device)
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
                input_tensor = prepare_input(batch["masked_i_space"], device)
                target_tensor = prepare_input(batch["full_i_space"], device)
                target_rss_raw = (
                    batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
                )

                input_norm, scale = normalize_batch(input_tensor)
                target_norm = target_tensor / scale

                pred_norm = model(input_norm)

                loss = loss_func(pred_norm, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    pred_rss = real30ch_to_rss(pred_norm * scale)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(pred_rss / rss_max, target_rss_raw / rss_max)

                train_loss_sum += loss.item()
                train_psnr_sum += psnr_val.item()
                train_batches += 1

                if rank == 0 and train_batches > 0:
                    train_iter.set_postfix(
                        loss=f"{train_loss_sum / train_batches:.4f}",
                        psnr=f"{train_psnr_sum / train_batches:.2f}",
                    )

            # ── Aggregate across GPUs ──
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
                    input_tensor = prepare_input(batch["masked_i_space"], device)
                    target_tensor = prepare_input(batch["full_i_space"], device)
                    target_rss_raw = (
                        batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
                    )

                    input_norm, scale = normalize_batch(input_tensor)
                    target_norm = target_tensor / scale

                    pred_norm = model(input_norm)
                    val_loss = loss_func(pred_norm, target_norm)

                    pred_rss = real30ch_to_rss(pred_norm * scale)
                    rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                    psnr_val = psnr_module(pred_rss / rss_max, target_rss_raw / rss_max)

                    val_loss_sum += val_loss.item()
                    val_psnr_sum += psnr_val.item()
                    val_batches += 1

                    if rank == 0 and val_batches > 0:
                        val_iter.set_postfix(
                            loss=f"{val_loss_sum / val_batches:.4f}",
                            psnr=f"{val_psnr_sum / val_batches:.2f}",
                        )

            # ── Aggregate across GPUs ──
            val_loss_global = reduce_scalar(val_loss_sum, device)
            val_psnr_global = reduce_scalar(val_psnr_sum, device)
            val_n = reduce_scalar(float(val_batches), device)
            avg_val_loss = val_loss_global / max(val_n, 1.0)
            avg_val_psnr = val_psnr_global / max(val_n, 1.0)

            scheduler.step(avg_val_loss)

            # ── Logging, checkpointing, early stopping (rank 0 only) ──
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

            # ── Broadcast early stopping state to all ranks ──
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