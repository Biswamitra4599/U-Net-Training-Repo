import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fastmri.data.subsample import EquiSpacedMaskFunc


from ispace_complex_multigpu_merged import (
    UNET_final,
    prepare_rss_input,
    prepare_rss_target,
    normalize_complex_batch,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import updated_dataloader


def center_crop(data, shape=(320, 320)):
    """Applies a center crop to the last two dimensions (H, W)."""
    h, w = data.shape[-2:]
    crop_h, crop_w = shape
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return data[..., start_h:start_h + crop_h, start_w:start_w + crop_w]


def eval_rss_model():
    # ── Paths ──
    model_dir = "/home/biswamitra/health/knee_data/EDA/saved_model/complex_rss_model_merged.pth"
    test_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    file_list = sorted(glob.glob(os.path.join(test_dir, "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = UNET_final().to(device)
    model.load_state_dict(
        torch.load(model_dir, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Metrics ──
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ── Mask & Dataset ──
    mask = EquiSpacedMaskFunc(
        center_fractions=[0.08, 0.04], accelerations=[4, 8]
    )

    data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=file_list,
        mask_func=mask,
        input_req=[1, 1, 1, 1, 1],
        output_req=[1, 1, 1, 1],
        methods_flags=[0, 0],
    )

    dataloader = DataLoader(
        dataset=data, batch_size=10, shuffle=False,
        num_workers=8, pin_memory=True,
    )

    running_psnr = 0.0
    running_ssim = 0.0
    total_batch = tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for batch in total_batch:
            # ── Prepare RSS complex input & target (B, 1, H, W) ──
            input_complex = prepare_rss_input(batch, device)
            target_complex = prepare_rss_target(batch, device)
            target_rss_raw = batch["full_rss_combined"].unsqueeze(1).to(
                device, dtype=torch.float32
            )

            # ── Normalize by input magnitude max (same as training) ──
            input_norm, scale = normalize_complex_batch(input_complex)

            # ── Forward ──
            pred_norm = model(input_norm)

            # ── Denormalize → magnitude ──
            pred_rss = torch.abs(pred_norm * scale)  # (B, 1, H, W)

            # ── Center crop ──
            pred_rss = center_crop(pred_rss)
            target_rss_raw = center_crop(target_rss_raw)

            # ── Normalize to [0,~1] by target max for metrics ──
            rss_max = target_rss_raw.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
            pred_eval = pred_rss / rss_max
            target_eval = target_rss_raw / rss_max

            # ── Metrics ──
            psnr_val = psnr_module(pred_eval, target_eval)
            ssim_val = ssim_module(pred_eval, target_eval)

            running_psnr += psnr_val.item()
            running_ssim += ssim_val.item()

            total_batch.set_postfix(
                psnr=f"{running_psnr / (total_batch.n):.2f}",
                ssim=f"{running_ssim / (total_batch.n):.4f}",
            )

    num_batches = len(dataloader)
    print(f"\nAverage PSNR over whole data: {running_psnr / num_batches:.2f} dB")
    print(f"Average SSIM over whole data: {running_ssim / num_batches:.4f}")


if __name__ == "__main__":
    eval_rss_model()