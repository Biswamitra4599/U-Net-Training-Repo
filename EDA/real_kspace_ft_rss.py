#!/usr/bin/env python3
"""
Real-valued U-Net training on FFT of RSS images.
Input:  FFT(downsampled_RSS) as 2-channel real tensor (real, imag)
Output: FFT(full_RSS) as 2-channel real tensor (real, imag)
Loss:   L1 on k-space + SSIM on IFFT(prediction) vs ground truth RSS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fastmri.data.subsample import EquiSpacedMaskFunc
import glob
import os
import json
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import updated_dataloader


# ==========================================
# 1. MODEL ARCHITECTURE (2-channel in/out)
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
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            doubleConv2D(input_channels, output_channels),
        )

    def forward(self, x):
        return self.downsample(x)


class UNET_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 input channels: real + imaginary parts of FFT(RSS)
        self.first_double_conv = doubleConv2D(2, 64)
        self.downsample1 = DownSample(64, 128)
        self.downsample2 = DownSample(128, 256)
        self.downsample3 = DownSample(256, 512)
        self.downsample4 = DownSample(512, 1024)

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
        self.tcv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.double_conv1 = doubleConv2D(1024, 512)
        self.tcv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_conv2 = doubleConv2D(512, 256)
        self.tcv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_conv3 = doubleConv2D(256, 128)
        self.tcv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv4 = doubleConv2D(128, 64)

    def forward(self, bottleneck, skip_conns_list):
        x = self.double_conv1(torch.cat([self.tcv1(bottleneck), skip_conns_list[0]], dim=1))
        x = self.double_conv2(torch.cat([self.tcv2(x), skip_conns_list[1]], dim=1))
        x = self.double_conv3(torch.cat([self.tcv3(x), skip_conns_list[2]], dim=1))
        x = self.double_conv4(torch.cat([self.tcv4(x), skip_conns_list[3]], dim=1))
        return x


class UNET_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNET_encoder()
        self.decoder = UNET_decoder()
        # 2 output channels: real + imaginary of predicted k-space
        self.one_cross_one_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1)

    def forward(self, x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck, skip_conns_list)
        residual = self.one_cross_one_conv(final_conv_op)
        return x + residual  # ── RESIDUAL LEARNING ──


# ==========================================
# 2. FFT/IFFT HELPERS
# ==========================================
def fft2c(x):
    """
    Centered FFT2: real/complex image → centered complex k-space.
    Pipeline: ifftshift → fft2 → fftshift
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1)
    )


def ifft2c(kspace):
    """
    Centered IFFT2: centered complex k-space → complex image.
    Pipeline: ifftshift → ifft2 → fftshift
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1)
    )


def complex_to_2ch(complex_tensor):
    """
    (B, 1, H, W) complex → (B, 2, H, W) real
    Channel 0 = real, Channel 1 = imaginary.
    """
    return torch.cat([complex_tensor.real, complex_tensor.imag], dim=1)


def twoch_to_complex(tensor_2ch):
    """
    (B, 2, H, W) real → (B, 1, H, W) complex
    Channel 0 = real, Channel 1 = imaginary.
    """
    return torch.complex(tensor_2ch[:, 0:1], tensor_2ch[:, 1:2])


def twoch_kspace_to_image(tensor_2ch):
    """
    (B, 2, H, W) real k-space → (B, 1, H, W) real magnitude image.
    Flow: 2ch real → complex → ifft2c → |magnitude|
    """
    complex_kspace = twoch_to_complex(tensor_2ch)      # (B, 1, H, W) complex
    complex_image = ifft2c(complex_kspace)              # (B, 1, H, W) complex
    return torch.abs(complex_image)                     # (B, 1, H, W) real


# ==========================================
# 3. DATA PREPARATION
# ==========================================
def prepare_rss_kspace_input(batch, device):
    """
    Masked multi-coil i-space → RSS → FFT → 2-channel real tensor.

    Flow:
        masked_i_space (B, 15, H, W, 2)
        → complex (B, 15, H, W)
        → RSS magnitude (B, 1, H, W) real
        → fft2c → (B, 1, H, W) complex
        → 2-channel real → (B, 2, H, W)
    """
    x = batch["masked_i_space"].to(device, dtype=torch.float32)
    multicoil = torch.complex(x[..., 0], x[..., 1])                # (B, 15, H, W)
    mag = torch.abs(multicoil)                                      # per-coil magnitude
    rss = torch.sqrt(torch.sum(mag ** 2, dim=1, keepdim=True))     # (B, 1, H, W)
    kspace_complex = fft2c(rss)                                     # (B, 1, H, W) complex
    return complex_to_2ch(kspace_complex)                           # (B, 2, H, W) real


def prepare_rss_kspace_target(batch, device):
    """
    Full RSS image → FFT → 2-channel real tensor.

    Flow:
        full_rss_combined (B, H, W)
        → unsqueeze → (B, 1, H, W)
        → fft2c → (B, 1, H, W) complex
        → 2-channel real → (B, 2, H, W)
    """
    rss = batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
    kspace_complex = fft2c(rss)                                     # (B, 1, H, W) complex
    return complex_to_2ch(kspace_complex)                           # (B, 2, H, W) real


def normalize_batch(tensor):
    """
    Per-sample normalization by absolute max across all channels.
    Handles both positive and negative values (real & imag parts).
    Returns: (normalized, scale) where scale is (B, 1, 1, 1)
    """
    scale = tensor.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(scale, min=1e-9)
    return tensor / scale, scale


# ==========================================
# 4. LOSS FUNCTION
# ==========================================
class KSpaceImageHybridLoss(nn.Module):
    """
    Hybrid loss:
        (1 - α) × L1_kspace  +  α × (1 - SSIM_image)

    - L1 computed directly on the 2-channel k-space (already normalized)
    - SSIM computed on IFFT(predicted) vs IFFT(target), both normalized
      to [0, 1] by target image max
    """

    def __init__(self, alpha=0.70):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred_kspace_2ch, target_kspace_2ch):
        # 1. K-space L1 (on normalized 2-channel real tensors)
        kspace_l1 = self.l1(pred_kspace_2ch, target_kspace_2ch)

        # 2. Convert to image domain for SSIM
        pred_img = twoch_kspace_to_image(pred_kspace_2ch)      # (B, 1, H, W)
        target_img = twoch_kspace_to_image(target_kspace_2ch)  # (B, 1, H, W)

        # 3. Normalize by TARGET image max for SSIM
        img_max = target_img.amax(dim=(1, 2, 3), keepdim=True)
        img_max = torch.clamp(img_max, min=1e-9)

        pred_img_norm = pred_img / img_max
        target_img_norm = target_img / img_max

        # 4. SSIM on normalized images
        ssim_val = self.ssim(pred_img_norm, target_img_norm)
        ssim_loss = 1.0 - ssim_val

        return (1.0 - self.alpha) * kspace_l1 + self.alpha * ssim_loss


# ==========================================
# 5. MAIN TRAINING FUNCTION
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Hyperparameters ──
    num_epochs = 50
    batch_size = 16
    learning_rate = 2e-4
    patience = 5
    max_grad_norm = 1.0

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/unet_rss_fft_model.pth"
    saved_model_dir = os.path.dirname(saved_model_path)
    metrics_path = os.path.join(saved_model_dir, "training_metrics_rss_fft.json")
    os.makedirs(saved_model_dir, exist_ok=True)

    print(f"Device: {device}")
    print("Training: FFT(downsampled_RSS) → U-Net → FFT(full_RSS)")

    # ── Data ──
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npy")))
    print(f"Found {len(train_files)} train / {len(val_files)} val files.")

    mask_func = EquiSpacedMaskFunc(center_fractions=[0.08], accelerations=[20])

    train_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=train_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )
    val_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=val_files, mask_func=mask_func,
        input_req=[1, 1, 1, 1, 1], output_req=[1, 1, 1, 1], methods_flags=[0, 0],
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
    )
    print(f"Train batches/epoch: {len(train_loader)}, Val batches/epoch: {len(val_loader)}")

    # ── Model ──
    model = UNET_final()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_func = KSpaceImageHybridLoss(alpha=0.70).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    trigger_times = 0
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    # ── Sanity check: verify FFT → IFFT roundtrip ──
    print("\nRunning FFT roundtrip sanity check...")
    test_batch = next(iter(train_loader))
    raw_rss = test_batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
    target_kspace_check = prepare_rss_kspace_target(test_batch, device)
    reconstructed_rss = twoch_kspace_to_image(target_kspace_check)
    roundtrip_error = torch.abs(reconstructed_rss - raw_rss).max().item()
    print(f"  Max roundtrip error: {roundtrip_error:.2e}")
    if roundtrip_error > 1e-4:
        print("  ⚠ WARNING: FFT roundtrip error is large!")
    else:
        print("  ✓ FFT roundtrip is consistent.")
    del test_batch, raw_rss, target_kspace_check, reconstructed_rss

    print("\n" + "=" * 70)
    print("Starting Training: FFT(RSS) k-space with image-domain SSIM")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):

        # ==================== TRAINING ====================
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        train_batches = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_loop:
            # ── Prepare: RSS → FFT → 2-channel real (B, 2, H, W) ──
            input_kspace = prepare_rss_kspace_input(batch, device)
            target_kspace = prepare_rss_kspace_target(batch, device)
            target_rss_raw = (
                batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
            )

            # ── Per-sample normalization by INPUT abs-max ──
            input_norm, scale = normalize_batch(input_kspace)
            target_norm = target_kspace / scale

            # ── Forward ──
            pred_norm = model(input_norm)

            # ── Loss (on normalized k-space + image-domain SSIM) ──
            loss = loss_func(pred_norm, target_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # ── PSNR: denormalize → IFFT → magnitude → compare with raw RSS ──
            with torch.no_grad():
                pred_denorm = pred_norm * scale
                pred_img = twoch_kspace_to_image(pred_denorm)  # (B, 1, H, W)
                rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                psnr_val = psnr_module(pred_img / rss_max, target_rss_raw / rss_max)

            running_loss += loss.item()
            running_psnr += psnr_val.item()
            train_batches += 1

            train_loop.set_postfix(
                loss=f"{running_loss / train_batches:.4f}",
                psnr=f"{running_psnr / train_batches:.2f}",
            )

        avg_train_loss = running_loss / max(train_batches, 1)
        avg_train_psnr = running_psnr / max(train_batches, 1)

        # ==================== VALIDATION ====================
        model.eval()
        running_loss = 0.0
        running_psnr = 0.0
        val_batches = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for batch in val_loop:
                input_kspace = prepare_rss_kspace_input(batch, device)
                target_kspace = prepare_rss_kspace_target(batch, device)
                target_rss_raw = (
                    batch["full_rss_combined"].unsqueeze(1).to(device, dtype=torch.float32)
                )

                # ── Same normalization as training ──
                input_norm, scale = normalize_batch(input_kspace)
                target_norm = target_kspace / scale

                pred_norm = model(input_norm)
                val_loss = loss_func(pred_norm, target_norm)

                # ── PSNR (identical to training) ──
                pred_denorm = pred_norm * scale
                pred_img = twoch_kspace_to_image(pred_denorm)
                rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
                psnr_val = psnr_module(pred_img / rss_max, target_rss_raw / rss_max)

                running_loss += val_loss.item()
                running_psnr += psnr_val.item()
                val_batches += 1

                val_loop.set_postfix(
                    loss=f"{running_loss / val_batches:.4f}",
                    psnr=f"{running_psnr / val_batches:.2f}",
                )

        avg_val_loss = running_loss / max(val_batches, 1)
        avg_val_psnr = running_psnr / max(val_batches, 1)

        scheduler.step(avg_val_loss)

        print(
            f"\nEpoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f}  PSNR: {avg_train_psnr:.2f} dB | "
            f"Val Loss: {avg_val_loss:.4f}  PSNR: {avg_val_psnr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── Metrics tracking ──
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_psnr"].append(avg_train_psnr)
        history["val_psnr"].append(avg_val_psnr)
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)

        # ── Early stopping & checkpointing ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(sd, saved_model_path)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})\n")
        else:
            trigger_times += 1
            print(f"  ✗ No improvement ({trigger_times}/{patience})\n")
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Model: {saved_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()