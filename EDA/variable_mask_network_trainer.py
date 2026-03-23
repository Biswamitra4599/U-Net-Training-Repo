#!/usr/bin/env python3
"""
=============================================================================
Acceleration-Conditioned MRI Mask Learning Pipeline
=============================================================================
Trains a mask network that learns optimal k-space undersampling patterns
conditioned on acceleration factor. Integrates with Custom_FMRI_DataLoader_nil.

NO reconstruction network included — uses zero-filled RSS as baseline.
Plug in your own reconstruction network later for better results.

Usage:
    python mask_training.py \
        --train_data_dir /path/to/train_npy/ \
        --val_data_dir /path/to/val_npy/ \
        --epochs 100 \
        --accel_factors 4.0 8.0

=============================================================================
"""

import os
import sys
import math
import random
import argparse
import logging
import glob
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import EquiSpacedMaskFunc

# ─── Import your dataloader ───
# Make sure this file is in your Python path or same directory
from load_mri_data import show_coils, show_multicoil_K_I, convert_K_to_I, convert_I_to_K, rss_combine
from updated_dataloader import Custom_FMRI_DataLoader_nil


try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """All hyperparameters in one place."""

    # ── Data paths ──
    train_data_dir: str = ""                # directory with .npy k-space files
    val_data_dir: str = ""
    file_pattern: str = "*.npy"             # glob pattern for data files

    # ── Data dimensions (from your data: coils, 640, 368) ──
    num_pe_lines: int = 368                 # W dimension = phase encode lines
    num_readout: int = 640                  # H dimension = readout (fully sampled)

    # ── Mask network ──
    embed_dim: int = 64
    mask_hidden_dim: int = 256
    mask_num_layers: int = 4

    # ── Training ──
    epochs: int = 100
    batch_size: int = 4
    lr_mask: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # ── Acceleration factors ──
    accel_factors: List[float] = field(default_factory=lambda: [4.0, 8.0])

    # ── Loss weights ──
    alpha_msssim: float = 0.84              # MS-SSIM weight in combined loss
    beta_freq: float = 0.1                  # frequency loss weight
    high_freq_weight: float = 2.0           # amplification for high-freq errors
    msssim_n_scales: int = 4                # number of scales for MS-SSIM

    # ── Temperature annealing ──
    temp_start: float = 5.0
    temp_end: float = 0.5

    # ── Center fraction ──
    # Set to 0.0 to let network learn whether to include center lines
    center_fraction: float = 0.04           # 4% center lines always sampled

    # ── Dataloader flags for Custom_FMRI_DataLoader_nil ──
    # We need: full_k_space and full_rss_combined only
    input_req: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    output_req: List[int] = field(default_factory=lambda: [1, 0, 1, 0])
    methods_flags: List[int] = field(default_factory=lambda: [0, 0])

    # ── Misc ──
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    save_dir: str = "./mask_checkpoints"
    log_interval: int = 10
    vis_interval: int = 5
    seed: int = 42


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# SECTION 2: UTILITIES
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_temperature(epoch: int, total_epochs: int,
                    t_start: float = 5.0, t_end: float = 0.5) -> float:
    """Cosine annealing for sigmoid temperature."""
    progress = epoch / max(total_epochs - 1, 1)
    return t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * progress))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_batch(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-image normalization to [0, 1].
    
    Args:
        images: (B, H, W) or (B, 1, H, W)
    Returns:
        normalized images with same shape
    """
    if images.dim() == 4:
        B = images.shape[0]
        flat = images.view(B, -1)
        mins = flat.min(dim=1)[0].view(B, 1, 1, 1)
        maxs = flat.max(dim=1)[0].view(B, 1, 1, 1)
    else:  # dim == 3
        B = images.shape[0]
        flat = images.view(B, -1)
        mins = flat.min(dim=1)[0].view(B, 1, 1)
        maxs = flat.max(dim=1)[0].view(B, 1, 1)

    return (images - mins) / (maxs - mins + eps)


# =============================================================================
# SECTION 3: MASK NETWORK
# =============================================================================
class AccelerationEmbedding(nn.Module):
    """
    Converts scalar acceleration factor R into a rich feature vector
    using sinusoidal positional encoding + MLP.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def sinusoidal_encode(self, R: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=R.device)
            * -(math.log(10000.0) / half_dim)
        )
        args = R.float().view(-1) * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)])
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros(1, device=R.device)])
        return embedding

    def forward(self, acceleration: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_encode(acceleration))


class SparsityNormalization(nn.Module):
    """
    Rescales probabilities so their mean equals alpha = 1/R.
    Guarantees the correct number of PE lines are selected on average.
    From LOUPE (Bahadir et al., 2020).
    """
    def forward(self, probs: torch.Tensor, alpha: float) -> torch.Tensor:
        p_mean = probs.mean()
        eps = 1e-8
        if p_mean >= alpha:
            normalized = (alpha / (p_mean + eps)) * probs
        else:
            normalized = 1.0 - ((1.0 - alpha) / (1.0 - p_mean + eps)) * (1.0 - probs)
        return normalized.clamp(0.0, 1.0)


def binarize_ste(probs: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for binarization.
    Forward: hard threshold at 0.5
    Backward: gradient passes through as identity
    """
    hard = (probs > 0.5).float()
    return hard - probs.detach() + probs


class AccelerationConditionedMaskNet(nn.Module):
    """
    Given acceleration factor R, outputs a binary 1D mask of length
    num_pe_lines indicating which phase-encode lines to acquire.

    Input:  acceleration (scalar tensor, e.g., tensor(4.0))
    Output: mask (num_pe_lines,) binary
            probs (num_pe_lines,) continuous probabilities
    """
    def __init__(self, num_pe_lines: int = 368, embed_dim: int = 64,
                 hidden_dim: int = 256, num_layers: int = 4,
                 center_fraction: float = 0.04):
        super().__init__()
        self.num_pe_lines = num_pe_lines
        self.center_fraction = center_fraction
        self.num_center = int(num_pe_lines * center_fraction)

        # Embedding
        self.embedding = AccelerationEmbedding(embed_dim)

        # Score network
        layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_pe_lines))
        self.score_network = nn.Sequential(*layers)

        # Normalization
        self.normalizer = SparsityNormalization()

        # Temperature buffer
        self.register_buffer('temperature', torch.tensor(5.0))

        # Center mask buffer
        center_mask = torch.zeros(num_pe_lines)
        if self.num_center > 0:
            center_start = num_pe_lines // 2 - self.num_center // 2
            center_mask[center_start:center_start + self.num_center] = 1.0
        self.register_buffer('center_mask', center_mask)

        self._initialize_center_bias()

    def _initialize_center_bias(self):
        """Warm-start: bias final layer to favor center PE lines."""
        with torch.no_grad():
            final_layer = self.score_network[-1]
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                center = self.num_pe_lines // 2
                positions = torch.arange(self.num_pe_lines, dtype=torch.float32)
                distances = (positions - center).float()
                sigma = self.num_pe_lines / 6.0
                initial_bias = 2.0 * torch.exp(-distances ** 2 / (2 * sigma ** 2))
                final_layer.bias.copy_(initial_bias)

    def forward(self, acceleration: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        accel_embed = self.embedding(acceleration)
        logits = self.score_network(accel_embed)
        probs = torch.sigmoid(logits / self.temperature)

        if self.num_center > 0:
            probs = torch.max(probs, self.center_mask)

        alpha = 1.0 / acceleration.item()
        probs_normalized = self.normalizer(probs, alpha)
        mask = binarize_ste(probs_normalized)

        return mask, probs_normalized

    def set_temperature(self, temp: float):
        self.temperature.fill_(temp)

    @torch.no_grad()
    def get_selected_lines(self, acceleration: torch.Tensor) -> torch.Tensor:
        mask, _ = self.forward(acceleration)
        return torch.where(mask > 0.5)[0]

    @torch.no_grad()
    def get_mask_info(self, acceleration: torch.Tensor) -> Dict:
        """Get detailed info about the generated mask."""
        mask, probs = self.forward(acceleration)
        selected = torch.where(mask > 0.5)[0]
        info = {
            'num_lines': len(selected),
            'target_lines': int(self.num_pe_lines / acceleration.item()),
            'selected_indices': selected.cpu().tolist(),
            'probs_min': probs.min().item(),
            'probs_max': probs.max().item(),
            'probs_mean': probs.mean().item(),
        }
        if len(selected) > 1:
            diffs = torch.diff(selected.float())
            info['min_spacing'] = diffs.min().item()
            info['max_spacing'] = diffs.max().item()
            info['mean_spacing'] = diffs.mean().item()
        return info


# =============================================================================
# SECTION 4: LOSS FUNCTIONS
# =============================================================================
def _gaussian_window_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window_2d(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    w1d = _gaussian_window_1d(window_size, 1.5, device)
    w2d = w1d.unsqueeze(1) @ w1d.unsqueeze(0)
    window = w2d.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_per_channel(img1, img2, window, data_range, k1=0.01, k2=0.03):
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    channel = img1.size(1)
    pad = window.size(-1) // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return ssim_map.mean(dim=(-2, -1)), cs_map.mean(dim=(-2, -1))


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, img1, img2):
        channel = img1.size(1)
        window = _create_window_2d(self.window_size, channel, img1.device)
        ssim_val, _ = ssim_per_channel(img1, img2, window, self.data_range)
        return ssim_val.mean()


class MSSSIM(nn.Module):
    def __init__(self, data_range: float = 1.0, window_size: int = 11,
                 n_scales: int = 4, weights: Optional[List[float]] = None):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.n_scales = n_scales

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363][:n_scales]
        self.weights = weights[:n_scales]
        w_sum = sum(self.weights)
        self.weights = [w / w_sum for w in self.weights]

    def forward(self, img1, img2):
        device = img1.device
        channel = img1.size(1)
        window = _create_window_2d(self.window_size, channel, device)

        cs_values = []
        ssim_val = None

        for i in range(self.n_scales):
            if img1.size(-1) < self.window_size or img1.size(-2) < self.window_size:
                break
            ssim_val, cs_val = ssim_per_channel(img1, img2, window, self.data_range)
            cs_values.append(cs_val.mean())
            if i < self.n_scales - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        if ssim_val is None:
            return torch.tensor(1.0, device=device, requires_grad=True)

        ms_ssim = ssim_val.mean()

        for i in range(len(cs_values) - 1):
            # ═══════════════════════════════════════════════════
            # FIX: clamp cs to POSITIVE values before power op.
            # This is THE main NaN source.
            # ═══════════════════════════════════════════════════
            cs_clamped = cs_values[i].clamp(min=1e-8)  # MUST be positive
            ms_ssim = ms_ssim * (cs_clamped ** self.weights[i])

        # ═══════════════════════════════════════════════════
        # FIX: final safety clamp
        # ═══════════════════════════════════════════════════
        ms_ssim = ms_ssim.clamp(0.0, 1.0)

        return ms_ssim


class FrequencyLoss(nn.Module):
    """
    Image-domain frequency loss.

    Computes FFT of the magnitude images (reconstruction vs ground truth)
    and penalizes errors weighted toward high frequencies.

    This is NOT comparing raw k-space. It is comparing the frequency
    content of the final images, which is a valid image-quality metric
    that emphasizes edges and fine detail.

    Mathematical justification:
        Parseval's theorem: ||f - g||² = ||F(f) - F(g)||²
        So unweighted frequency loss = L2 image loss (equivalent).
        Weighted frequency loss = emphasizes specific spatial frequencies
        in the image, which is NOT equivalent to L2 and provides
        complementary information to L1 and SSIM.
    """
    def __init__(self, high_freq_weight: float = 2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self._freq_weight_cache = {}

    def _get_freq_weight(self, H: int, W: int, device: torch.device):
        key = (H, W)
        if key not in self._freq_weight_cache:
            ky = torch.arange(H, dtype=torch.float32) - H // 2
            kx = torch.arange(W, dtype=torch.float32) - W // 2
            ky, kx = torch.meshgrid(ky, kx, indexing='ij')
            freq_radius = torch.sqrt(kx ** 2 + ky ** 2)
            freq_radius = freq_radius / (freq_radius.max() + 1e-8)
            freq_weight = 1.0 + self.high_freq_weight * freq_radius
            freq_weight = torch.fft.fftshift(freq_weight)
            self._freq_weight_cache[key] = freq_weight
        return self._freq_weight_cache[key].to(device)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = torch.fft.fft2(prediction, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')

        diff = pred_freq - target_freq
        
        # SAFE MAGNITUDE: Instead of torch.abs(), calculate safely
        # using real and imag parts with epsilon
        freq_error = torch.sqrt(diff.real**2 + diff.imag**2 + 1e-11)

        H, W = prediction.shape[-2], prediction.shape[-1]
        freq_weight = self._get_freq_weight(H, W, prediction.device)

        weighted_error = freq_error * freq_weight.unsqueeze(0).unsqueeze(0)
        return weighted_error.mean()


class CombinedReconstructionLoss(nn.Module):
    """
    α * (1 - MS-SSIM) + (1-α) * L1 + β * FrequencyLoss

    - MS-SSIM: preserves edges, structure, local contrast
    - L1: maintains pixel accuracy, prevents intensity drift
    - Frequency: preserves important k-space information
    """
    def __init__(self, alpha_msssim: float = 0.84, beta_freq: float = 0.1,
                 data_range: float = 1.0, n_scales: int = 4,
                 high_freq_weight: float = 2.0):
        super().__init__()
        self.alpha = alpha_msssim
        self.beta = beta_freq
        self.ms_ssim = MSSSIM(data_range=data_range, n_scales=n_scales)
        self.l1_loss = nn.L1Loss()
        self.freq_loss = FrequencyLoss(high_freq_weight=high_freq_weight)

    def forward(self, prediction, target):
        ms_ssim_val = self.ms_ssim(prediction, target)
        ms_ssim_loss = 1.0 - ms_ssim_val
        l1_val = self.l1_loss(prediction, target)
        freq_val = self.freq_loss(prediction, target)

        loss = self.alpha * ms_ssim_loss + (1.0 - self.alpha) * l1_val + self.beta * freq_val

        metrics = {
            'total_loss': loss.item(),
            'ms_ssim': ms_ssim_val.item(),
            'ms_ssim_loss': ms_ssim_loss.item(),
            'l1': l1_val.item(),
            'freq': freq_val.item(),
        }
        return loss, metrics


# =============================================================================
# SECTION 5: TRAINING PIPELINE
# =============================================================================
class MaskTrainingPipeline(nn.Module):
    """
    End-to-end mask training pipeline.

    Flow:
        1. Mask network generates binary mask from acceleration R
        2. Mask applied to full k-space (zeros unsampled PE lines)
        3. IFFT → per-coil images → RSS combine → magnitude image
        4. (Optional) reconstruction network cleans up the image
        5. Compare to ground truth RSS → loss → backprop through mask

    Data format (from your Custom_FMRI_DataLoader_nil):
        full_k_space:       (B, coils, H=640, W=368, 2)  [fastmri tensor format]
        full_rss_combined:  (B, H=640, W=368)             [ground truth RSS]

    Mask is applied along W=368 dimension (phase encode lines).

    ┌─────────────────────────────────────────────────────────────┐
    │  IMPORTANT: Without a reconstruction network, the learned  │
    │  masks will be biased toward center-heavy sampling.        │
    │  The MS-SSIM + frequency loss helps, but for best results  │
    │  plug in your reconstruction network via recon_net param.  │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, config: Config, recon_net: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Mask network
        self.mask_net = AccelerationConditionedMaskNet(
            num_pe_lines=config.num_pe_lines,
            embed_dim=config.embed_dim,
            hidden_dim=config.mask_hidden_dim,
            num_layers=config.mask_num_layers,
            center_fraction=config.center_fraction,
        )

        # Optional reconstruction network
        # If None: evaluation uses zero-filled RSS directly
        # If provided: zero-filled RSS → recon_net → cleaned image
        self.recon_net = recon_net

        # Loss
        self.loss_fn = CombinedReconstructionLoss(
            alpha_msssim=config.alpha_msssim,
            beta_freq=config.beta_freq,
            high_freq_weight=config.high_freq_weight,
            n_scales=config.msssim_n_scales,
        )

    def apply_mask_to_kspace(self, kspace: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D PE line mask to multi-coil k-space.

        Args:
            kspace: (B, coils, H, W, 2) — fastmri tensor format
            mask:   (W,) — binary mask, one value per PE line

        Returns:
            masked_kspace: (B, coils, H, W, 2) with unsampled lines zeroed

        The mask broadcasts across:
            B (batch), coils, H (readout), and the 2 (real/imag) dims.
        """
        # Reshape mask for broadcasting: (1, 1, 1, W, 1)
        mask_5d = mask.view(1, 1, 1, -1, 1)
        return kspace * mask_5d

    def compute_rss_from_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        # IFFT: (B, coils, H, W, 2)
        ispace = fastmri.ifft2c(kspace)

        # Magnitude squared: real^2 + imag^2
        mag_sq = ispace[..., 0]**2 + ispace[..., 1]**2
        
        # RSS squared: sum over coils
        rss_sq = mag_sq.sum(dim=1)
        
        # SAFE SQRT: add epsilon to prevent NaN gradients on background pixels
        rss = torch.sqrt(rss_sq + 1e-11)
        
        return rss

    def forward(self, full_kspace, gt_rss, acceleration):
        mask, probs = self.mask_net(acceleration)
        masked_kspace = self.apply_mask_to_kspace(full_kspace, mask)
        zf_rss = self.compute_rss_from_kspace(masked_kspace)

        B = gt_rss.shape[0]

        # ═══════════════════════════════════════════════════════════════
        # FIX: robust normalization that handles near-zero slices.
        # Some MRI slices (edges of volume) are nearly empty.
        # gt_max ≈ 0 → division produces huge values → NaN in loss.
        # ═══════════════════════════════════════════════════════════════
        gt_max = gt_rss.reshape(B, -1).max(dim=1)[0].view(B, 1, 1)

        # Skip samples with near-zero ground truth
        valid_mask = gt_max.view(-1) > 1e-6  # (B,)

        if not valid_mask.any():
            # Entire batch is empty — return zero loss
            dummy_loss = torch.tensor(0.0, device=full_kspace.device, requires_grad=True)
            metrics = {
                'total_loss': 0.0, 'ms_ssim': 1.0, 'ms_ssim_loss': 0.0,
                'l1': 0.0, 'freq': 0.0, 'num_lines': mask.sum().item(),
                'target_lines': self.config.num_pe_lines / acceleration.item(),
                'temperature': self.mask_net.temperature.item(), 'psnr': 0.0,
                'skipped_samples': B,
            }
            return zf_rss.unsqueeze(1)[:1], dummy_loss, metrics, mask, probs

        # Only use valid samples
        gt_rss_valid = gt_rss[valid_mask]
        zf_rss_valid = zf_rss[valid_mask]
        B_valid = gt_rss_valid.shape[0]

        gt_max_valid = gt_rss_valid.reshape(B_valid, -1).max(dim=1)[0].view(B_valid, 1, 1) + 1e-8
        gt_normalized = gt_rss_valid / gt_max_valid
        zf_normalized = (zf_rss_valid / gt_max_valid).clamp(0, 1)

        gt_4d = gt_normalized.unsqueeze(1)
        zf_4d = zf_normalized.unsqueeze(1)

        output = self.recon_net(zf_4d) if self.recon_net else zf_4d

        # ═══════════════════════════════════════════════════════════════
        # FIX: NaN guard on loss input — catch any remaining NaN
        # ═══════════════════════════════════════════════════════════════
        if torch.any(torch.isnan(output)) or torch.any(torch.isnan(gt_4d)):
            dummy_loss = torch.tensor(0.0, device=full_kspace.device, requires_grad=True)
            metrics = {
                'total_loss': 0.0, 'ms_ssim': 1.0, 'ms_ssim_loss': 0.0,
                'l1': 0.0, 'freq': 0.0, 'num_lines': mask.sum().item(),
                'target_lines': self.config.num_pe_lines / acceleration.item(),
                'temperature': self.mask_net.temperature.item(), 'psnr': 0.0,
                'skipped_samples': B,
            }
            return output, dummy_loss, metrics, mask, probs

        loss, metrics = self.loss_fn(output, gt_4d)

        metrics['num_lines'] = mask.sum().item()
        metrics['target_lines'] = self.config.num_pe_lines / acceleration.item()
        metrics['temperature'] = self.mask_net.temperature.item()
        metrics['skipped_samples'] = int(B - B_valid)

        mse = F.mse_loss(output, gt_4d)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        metrics['psnr'] = psnr.item()

        return output, loss, metrics, mask, probs


# =============================================================================
# SECTION 6: TRAINER
# =============================================================================
class MaskTrainer:
    """Handles training, validation, checkpointing, and visualization."""

    def __init__(self, pipeline: MaskTrainingPipeline, config: Config):
        self.pipeline = pipeline.to(config.device)
        self.config = config

        # Optimizer for mask network only
        # (if you add a recon_net, add its params here too)
        params = list(pipeline.mask_net.parameters())
        if pipeline.recon_net is not None:
            params += list(pipeline.recon_net.parameters())

        self.optimizer = torch.optim.Adam(
            params,
            lr=config.lr_mask,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr_mask * 0.01
        )

        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        os.makedirs(config.save_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        self.pipeline.train()
        device = self.config.device

        temp = get_temperature(epoch, self.config.epochs,
                               self.config.temp_start, self.config.temp_end)
        self.pipeline.mask_net.set_temperature(temp)

        epoch_metrics = {k: 0.0 for k in [
            'total_loss', 'ms_ssim', 'l1', 'freq', 'psnr', 'num_lines'
        ]}
        num_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]",
                    leave=False)

        for batch_idx, batch in enumerate(pbar):
            # ── Get data from your dataloader ──
            full_kspace = batch['full_k_space'].to(device)       # (B, coils, H, W, 2)
            gt_rss = batch['full_rss_combined'].to(device)       # (B, H, W)

            # ── Randomly pick acceleration factor ──
            R = random.choice(self.config.accel_factors)
            accel = torch.tensor(R, device=device)

            # ── Forward ──
            output, loss, metrics, mask, probs = self.pipeline(
                full_kspace, gt_rss, accel
            )

            # ── Backward ──
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.pipeline.parameters(), max_norm=self.config.grad_clip
            )
            self.optimizer.step()

            # ── Track metrics ──
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'ssim': f"{metrics['ms_ssim']:.3f}",
                    'psnr': f"{metrics['psnr']:.1f}",
                    'R': f"{R}x",
                    'lines': f"{metrics['num_lines']:.0f}/{metrics['target_lines']:.0f}",
                    'τ': f"{temp:.2f}",
                })

        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_batches, 1)
        self.scheduler.step()

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        self.pipeline.eval()
        device = self.config.device

        val_metrics = {k: 0.0 for k in [
            'total_loss', 'ms_ssim', 'l1', 'freq', 'psnr'
        ]}
        num_batches = 0

        for batch in val_loader:
            full_kspace = batch['full_k_space'].to(device)
            gt_rss = batch['full_rss_combined'].to(device)

            for R in self.config.accel_factors:
                accel = torch.tensor(R, device=device)
                output, loss, metrics, mask, probs = self.pipeline(
                    full_kspace, gt_rss, accel
                )
                for k in val_metrics:
                    if k in metrics:
                        val_metrics[k] += metrics[k]
                num_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(num_batches, 1)

        return val_metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'mask_net_state_dict': self.pipeline.mask_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }

        if self.pipeline.recon_net is not None:
            checkpoint['recon_net_state_dict'] = self.pipeline.recon_net.state_dict()

        path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_mask_net.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"  ★ New best model saved (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device)
        self.pipeline.mask_net.load_state_dict(checkpoint['mask_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {path} (epoch {epoch+1})")
        return epoch

    @torch.no_grad()
    def visualize(self, val_loader: DataLoader, epoch: int):
        """Visualize masks, zero-filled images, and error maps."""
        if not HAS_MATPLOTLIB:
            return

        self.pipeline.eval()
        device = self.config.device

        batch = next(iter(val_loader))
        full_kspace = batch['full_k_space'][:1].to(device)   # single sample
        gt_rss = batch['full_rss_combined'][:1].to(device)

        n_accels = len(self.config.accel_factors)
        fig, axes = plt.subplots(4, n_accels + 1, figsize=(5 * (n_accels + 1), 20))

        # Ground truth column
        gt_np = gt_rss[0].cpu().numpy()
        gt_norm = gt_np / (gt_np.max() + 1e-8)

        axes[0, 0].imshow(gt_norm, cmap='gray')
        axes[0, 0].set_title('Ground Truth RSS', fontsize=10)
        axes[0, 0].axis('off')

        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Ground\nTruth', ha='center', va='center',
                        fontsize=14, transform=axes[1, 0].transAxes)

        axes[2, 0].axis('off')
        axes[3, 0].axis('off')

        for col, R in enumerate(self.config.accel_factors, 1):
            accel = torch.tensor(R, device=device)
            output, loss, metrics, mask, probs = self.pipeline(
                full_kspace, gt_rss, accel
            )

            out_np = output[0, 0].cpu().numpy()

            # Row 0: Reconstruction
            axes[0, col].imshow(out_np, cmap='gray')
            axes[0, col].set_title(
                f'R={R}x | SSIM={metrics["ms_ssim"]:.3f}\n'
                f'PSNR={metrics["psnr"]:.1f}dB | Lines={metrics["num_lines"]:.0f}',
                fontsize=9
            )
            axes[0, col].axis('off')

            # Row 1: Error map
            gt_4d_np = output[0, 0].cpu().numpy()  # normalized gt
            error = np.abs(gt_norm[:out_np.shape[0], :out_np.shape[1]] - out_np)
            im = axes[1, col].imshow(error, cmap='hot', vmin=0,
                                      vmax=max(0.2, error.max()))
            axes[1, col].set_title(f'Error Map (L1={metrics["l1"]:.4f})', fontsize=9)
            axes[1, col].axis('off')

            # Row 2: Mask as vertical bars
            mask_np = mask.cpu().numpy()
            mask_img = np.zeros((30, len(mask_np)))
            mask_img[:, :] = mask_np[None, :]
            axes[2, col].imshow(mask_img, cmap='gray', aspect='auto')
            axes[2, col].set_title(f'Mask ({int(mask_np.sum())} / {len(mask_np)} lines)',
                                   fontsize=9)
            axes[2, col].set_xlabel('PE Line Index', fontsize=8)
            axes[2, col].set_yticks([])

            # Row 3: Probability distribution
            probs_np = probs.cpu().numpy()
            axes[3, col].bar(range(len(probs_np)), probs_np, width=1.0,
                             color='steelblue', alpha=0.7)
            axes[3, col].set_xlabel('PE Line Index', fontsize=8)
            axes[3, col].set_ylabel('Probability', fontsize=8)
            axes[3, col].set_title('Sampling Probability', fontsize=9)
            axes[3, col].set_ylim(0, 1.05)
            axes[3, col].axhline(y=1.0/R, color='red', linestyle='--',
                                 alpha=0.5, label=f'α={1/R:.2f}')
            axes[3, col].legend(fontsize=7)

        plt.suptitle(
            f'Epoch {epoch + 1} | τ = {self.pipeline.mask_net.temperature.item():.2f}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        save_path = os.path.join(self.config.save_dir, f"vis_epoch_{epoch+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Visualization saved: {save_path}")

    @torch.no_grad()
    def compare_with_equispaced(self, val_loader: DataLoader):
        """
        Compare learned masks against EquiSpacedMaskFunc baselines.
        Shows why the learned mask is (or isn't) better.
        """
        self.pipeline.eval()
        device = self.config.device
        N = self.config.num_pe_lines

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON: Learned Mask vs EquiSpaced Baselines")
        logger.info("=" * 70)

        for R in self.config.accel_factors:
            R_int = int(R)
            center_frac = self.config.center_fraction if self.config.center_fraction > 0 else 0.08
            num_keep = int(N / R)

            # ── Baseline: EquiSpacedMaskFunc ──
            equi_mask_func = EquiSpacedMaskFunc(
                center_fractions=[center_frac],
                accelerations=[R_int]
            )

            # ── Learned mask ──
            accel = torch.tensor(R, device=device)
            learned_mask, _ = self.pipeline.mask_net(accel)

            # Evaluate both on validation data
            results = {'Learned': [], 'EquiSpaced': []}

            for batch in val_loader:
                full_kspace = batch['full_k_space'].to(device)
                gt_rss = batch['full_rss_combined'].to(device)
                B = full_kspace.shape[0]
                gt_max = gt_rss.reshape(B, -1).max(dim=1)[0].view(B, 1, 1) + 1e-8

                # ── Learned mask eval ──
                masked_ks = self.pipeline.apply_mask_to_kspace(full_kspace, learned_mask)
                zf_rss = self.pipeline.compute_rss_from_kspace(masked_ks)
                zf_norm = (zf_rss / gt_max).clamp(0, 1)
                gt_norm = gt_rss / gt_max
                mse = F.mse_loss(zf_norm, gt_norm)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                results['Learned'].append(psnr.item())

                # ── EquiSpaced mask eval ──
                # Generate equispaced mask for this batch
                shape = (1,) * (full_kspace.dim() - 3) + tuple(full_kspace.shape[-3:])
                equi_mask_tensor, _, _ = equi_mask_func(shape)
                equi_mask_tensor = equi_mask_tensor.to(device)
                # equi_mask shape: typically (1, 1, W, 1) — need to adapt
                masked_ks_equi = full_kspace * equi_mask_tensor
                zf_rss_equi = self.pipeline.compute_rss_from_kspace(masked_ks_equi)
                zf_norm_equi = (zf_rss_equi / gt_max).clamp(0, 1)
                mse_equi = F.mse_loss(zf_norm_equi, gt_norm)
                psnr_equi = 10 * torch.log10(1.0 / (mse_equi + 1e-10))
                results['EquiSpaced'].append(psnr_equi.item())

            logger.info(f"\n  R = {R}x:")
            logger.info(f"    {'Method':<15} {'PSNR (dB)':<15} {'Lines':<10}")
            logger.info(f"    {'-'*40}")
            logger.info(f"    {'Learned':<15} {np.mean(results['Learned']):<15.2f} "
                         f"{int(learned_mask.sum().item()):<10}")
            logger.info(f"    {'EquiSpaced':<15} {np.mean(results['EquiSpaced']):<15.2f} "
                         f"{num_keep:<10}")
            
            diff = np.mean(results['Learned']) - np.mean(results['EquiSpaced'])
            logger.info(f"    → Learned is {'+' if diff > 0 else ''}{diff:.2f} dB "
                         f"{'better' if diff > 0 else 'worse'}")

    def print_final_masks(self):
        """Print the learned masks for each acceleration factor."""
        self.pipeline.eval()
        device = self.config.device

        logger.info("\n" + "=" * 60)
        logger.info("LEARNED MASKS SUMMARY")
        logger.info("=" * 60)

        for R in self.config.accel_factors:
            accel = torch.tensor(R, device=device)
            info = self.pipeline.mask_net.get_mask_info(accel)
            logger.info(f"\n  R = {R}x:")
            logger.info(f"    Selected lines: {info['num_lines']} "
                         f"(target: {info['target_lines']})")
            logger.info(f"    Indices: {info['selected_indices'][:20]}..."
                         if len(info['selected_indices']) > 20
                         else f"    Indices: {info['selected_indices']}")
            if 'min_spacing' in info:
                logger.info(f"    Spacing — min: {info['min_spacing']:.0f}, "
                             f"max: {info['max_spacing']:.0f}, "
                             f"mean: {info['mean_spacing']:.1f}")
            logger.info(f"    Prob range: [{info['probs_min']:.3f}, {info['probs_max']:.3f}]")

    def plot_training_curves(self):
        """Plot and save training curves."""
        if not HAS_MATPLOTLIB or len(self.train_history) == 0:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_range = range(1, len(self.train_history) + 1)

        # Loss
        axes[0].plot(epochs_range,
                     [m['total_loss'] for m in self.train_history],
                     label='Train', linewidth=2)
        axes[0].plot(epochs_range,
                     [m['total_loss'] for m in self.val_history],
                     label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MS-SSIM
        axes[1].plot(epochs_range,
                     [m['ms_ssim'] for m in self.train_history],
                     label='Train', linewidth=2)
        axes[1].plot(epochs_range,
                     [m['ms_ssim'] for m in self.val_history],
                     label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MS-SSIM')
        axes[1].set_title('MS-SSIM (↑)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # PSNR
        train_psnr = [m.get('psnr', 0) for m in self.train_history]
        val_psnr = [m.get('psnr', 0) for m in self.val_history]
        axes[2].plot(epochs_range, train_psnr, label='Train', linewidth=2)
        axes[2].plot(epochs_range, val_psnr, label='Val', linewidth=2, color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('PSNR (dB)')
        axes[2].set_title('PSNR (↑)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.save_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved: {save_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("MASK NETWORK TRAINING")
        logger.info("=" * 60)
        logger.info(f"  Mask Net params:   {count_parameters(self.pipeline.mask_net):,}")
        logger.info(f"  Recon Net:         "
                     f"{'YES (' + str(count_parameters(self.pipeline.recon_net)) + ' params)' if self.pipeline.recon_net else 'None (zero-filled RSS)'}")
        logger.info(f"  Accel factors:     {self.config.accel_factors}")
        logger.info(f"  PE lines:          {self.config.num_pe_lines}")
        logger.info(f"  Center fraction:   {self.config.center_fraction}")
        logger.info(f"  Loss:              {self.config.alpha_msssim}×MS-SSIM + "
                     f"{1-self.config.alpha_msssim:.2f}×L1 + "
                     f"{self.config.beta_freq}×Freq")
        logger.info(f"  Device:            {self.config.device}")
        logger.info(f"  Temperature:       {self.config.temp_start} → {self.config.temp_end}")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)

            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)

            logger.info(
                f"Epoch {epoch+1:3d}/{self.config.epochs} │ "
                f"T.Loss: {train_metrics['total_loss']:.4f} │ "
                f"V.Loss: {val_metrics['total_loss']:.4f} │ "
                f"V.SSIM: {val_metrics['ms_ssim']:.4f} │ "
                f"V.PSNR: {val_metrics['psnr']:.1f}dB │ "
                f"Lines: {train_metrics['num_lines']:.0f}"
            )

            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            self.save_checkpoint(epoch, val_metrics['total_loss'], is_best)

            if (epoch + 1) % self.config.vis_interval == 0 or epoch == 0:
                self.visualize(val_loader, epoch)

        # ── Post-training ──
        self.visualize(val_loader, self.config.epochs - 1)
        self.plot_training_curves()
        self.print_final_masks()
        self.compare_with_equispaced(val_loader)

        logger.info(f"\n✓ Training complete! Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"  Checkpoints saved to: {self.config.save_dir}")


# =============================================================================
# SECTION 7: HELPER — CREATE DATALOADERS FROM YOUR CLASS
# =============================================================================
def create_dataloaders(config: Config):
    """
    Create train and val DataLoaders using Custom_FMRI_DataLoader_nil.

    Your dataloader flags:
        input_req=[0,0,0,0,0]   — we don't need any masked inputs
        output_req=[1,0,1,0]    — we need full_k_space and full_rss_combined
        methods_flags=[0,0]     — no grappa/espirit needed for mask training

    The pipeline applies its OWN learned mask, so we only need
    the fully sampled k-space and ground truth from the dataloader.
    """
    # ── Collect file paths ──
    train_paths = sorted(glob.glob(os.path.join(config.train_data_dir, config.file_pattern)))
    val_paths = sorted(glob.glob(os.path.join(config.val_data_dir, config.file_pattern)))

    if len(train_paths) == 0:
        raise FileNotFoundError(
            f"No files matching '{config.file_pattern}' in {config.train_data_dir}"
        )
    if len(val_paths) == 0:
        raise FileNotFoundError(
            f"No files matching '{config.file_pattern}' in {config.val_data_dir}"
        )

    logger.info(f"  Train files: {len(train_paths)}")
    logger.info(f"  Val files:   {len(val_paths)}")

    # ── Dummy mask func (won't be used since input_req masks are all 0) ──
    # Your dataloader requires a mask_func parameter even though we won't
    # use the masked outputs. Using a simple one as placeholder.
    dummy_mask_func = EquiSpacedMaskFunc(
        center_fractions=[0.08], accelerations=[4]
    )

    # ── Create datasets ──
    train_dataset = Custom_FMRI_DataLoader_nil(
        data_paths=train_paths,
        mask_func=dummy_mask_func,
        input_req=config.input_req,       # [0,0,0,0,0] — no masked inputs
        output_req=config.output_req,     # [1,0,1,0]   — full_k_space + full_rss
        methods_flags=config.methods_flags,  # [0,0]     — no grappa/espirit
    )

    val_dataset = Custom_FMRI_DataLoader_nil(
        data_paths=val_paths,
        mask_func=dummy_mask_func,
        input_req=config.input_req,
        output_req=config.output_req,
        methods_flags=config.methods_flags,
    )

    # ── Create loaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# SECTION 8: EXPORT LEARNED MASK FOR USE WITH YOUR DATALOADER
# =============================================================================
class LearnedMaskFunc:
    """
    Drop-in replacement for EquiSpacedMaskFunc that uses the learned mask.

    Usage:
        # Load trained mask network
        learned_mask_func = LearnedMaskFunc.from_checkpoint(
            'mask_checkpoints/best_mask_net.pt',
            acceleration=4.0,
            num_pe_lines=368
        )

        # Use with your dataloader
        dataset = Custom_FMRI_DataLoader_nil(
            data_paths=paths,
            mask_func=learned_mask_func,
            ...
        )
    """
    def __init__(self, mask_tensor: torch.Tensor):
        """
        Args:
            mask_tensor: (W,) binary tensor — the learned mask
        """
        self.mask = mask_tensor.cpu()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, acceleration: float,
                        num_pe_lines: int = 368, config: Optional[Config] = None):
        """Load from a training checkpoint."""
        if config is None:
            config = Config(num_pe_lines=num_pe_lines)

        mask_net = AccelerationConditionedMaskNet(
            num_pe_lines=config.num_pe_lines,
            embed_dim=config.embed_dim,
            hidden_dim=config.mask_hidden_dim,
            num_layers=config.mask_num_layers,
            center_fraction=config.center_fraction,
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        mask_net.load_state_dict(checkpoint['mask_net_state_dict'])
        mask_net.eval()

        with torch.no_grad():
            mask, _ = mask_net(torch.tensor(acceleration))

        return cls(mask)

    def __call__(self, shape, offset=None, seed=None):
        """
        Compatible with fastmri's apply_mask interface.

        Args:
            shape: shape of the k-space data (used to determine mask shape)

        Returns:
            mask: tensor broadcastable to k-space shape
            num_low_frequencies: 0 (not applicable for learned masks)
        """
        # shape is typically (1, coils, H, W, 2) or similar
        # Mask should be (1, 1, 1, W, 1) for broadcasting
        W = shape[-2]  # PE dimension
        assert W == len(self.mask), \
            f"Mask size mismatch: expected {W}, got {len(self.mask)}"

        # Reshape to match fastmri convention
        ndim = len(shape)
        mask_shape = [1] * ndim
        mask_shape[-2] = W  # PE dimension
        mask_reshaped = self.mask.reshape(mask_shape)

        return mask_reshaped, 0


# =============================================================================
# SECTION 9: MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Acceleration-Conditioned Mask Network"
    )

    # Data
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help='Directory with training .npy k-space files')
    parser.add_argument('--val_data_dir', type=str, required=True,
                        help='Directory with validation .npy k-space files')
    parser.add_argument('--file_pattern', type=str, default='*.npy',
                        help='Glob pattern for data files')

    # Data dimensions
    parser.add_argument('--num_pe_lines', type=int, default=368,
                        help='Number of phase-encode lines (W dimension)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_mask', type=float, default=1e-4)

    # Acceleration
    parser.add_argument('--accel_factors', type=float, nargs='+',
                        default=[4.0, 8.0])

    # Loss
    parser.add_argument('--alpha_msssim', type=float, default=0.84)
    parser.add_argument('--beta_freq', type=float, default=0.1)

    # Center fraction
    parser.add_argument('--center_fraction', type=float, default=0.04,
                        help='0.0 = let network learn, >0 = enforce center')

    # Temperature
    parser.add_argument('--temp_start', type=float, default=5.0)
    parser.add_argument('--temp_end', type=float, default=0.5)

    # Misc
    parser.add_argument('--save_dir', type=str, default='./mask_checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vis_interval', type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        file_pattern=args.file_pattern,
        num_pe_lines=args.num_pe_lines,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_mask=args.lr_mask,
        accel_factors=args.accel_factors,
        alpha_msssim=args.alpha_msssim,
        beta_freq=args.beta_freq,
        center_fraction=args.center_fraction,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        save_dir=args.save_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        vis_interval=args.vis_interval,
    )

    set_seed(config.seed)

    # ── Data ──
    train_loader, val_loader = create_dataloaders(config)

    # ── Pipeline ──
    # Pass recon_net=None for mask-only training (zero-filled RSS evaluation)
    # To add your reconstruction network later:
    #   pipeline = MaskTrainingPipeline(config, recon_net=YourReconNet())
    pipeline = MaskTrainingPipeline(config, recon_net=None)

    logger.info(f"\nMask Network: {count_parameters(pipeline.mask_net):,} parameters")

    # ── Train ──
    trainer = MaskTrainer(pipeline, config)
    trainer.train(train_loader, val_loader)

    # ── Export learned masks ──
    logger.info("\n" + "=" * 60)
    logger.info("EXPORTING LEARNED MASKS")
    logger.info("=" * 60)

    for R in config.accel_factors:
        # Save mask as numpy array
        accel = torch.tensor(R, device=config.device)
        with torch.no_grad():
            mask, probs = pipeline.mask_net(accel)
        mask_np = mask.cpu().numpy()
        probs_np = probs.cpu().numpy()

        mask_path = os.path.join(config.save_dir, f"learned_mask_R{R}x.npy")
        probs_path = os.path.join(config.save_dir, f"learned_probs_R{R}x.npy")
        np.save(mask_path, mask_np)
        np.save(probs_path, probs_np)
        logger.info(f"  R={R}x: mask saved to {mask_path} "
                     f"({int(mask_np.sum())} lines)")

    logger.info("\nDone! To use learned masks with your dataloader:")
    logger.info("  learned_mask_func = LearnedMaskFunc.from_checkpoint(")
    logger.info("      'mask_checkpoints/best_mask_net.pt',")
    logger.info("      acceleration=4.0, num_pe_lines=368")
    logger.info("  )")
    logger.info("  dataset = Custom_FMRI_DataLoader_nil(")
    logger.info("      data_paths=paths, mask_func=learned_mask_func, ...")
    logger.info("  )")


if __name__ == "__main__":
    main()