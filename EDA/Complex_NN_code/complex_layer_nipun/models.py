import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_utils import (
    complex_modulate,
    complex_demodulate,
    complex_abs,
    complex_matmul
)

from complex_layers import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    ComplexReLU,
    ComplexResBlock,
    ComplexAdaptiveAvgPool2d,
    ComplexLinear
)

# ============================================================
# ENCODERS
# ============================================================

class DeepComplexEncoder(nn.Module):
    def __init__(self, in_c=2):
        super().__init__()

        self.stem = nn.Sequential(
            ComplexConv2d(in_c, 32, 3, 2, 1),
            ComplexBatchNorm2d(32),
            ComplexReLU()
        )

        self.stage1 = nn.Sequential(
            ComplexConv2d(32, 64, 3, 2, 1),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexResBlock(64),
            ComplexResBlock(64)
        )

        self.stage2 = nn.Sequential(
            ComplexConv2d(64, 128, 3, 1, 1),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexResBlock(128),
            ComplexResBlock(128)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class ComplexWaveletEncoder(nn.Module):
    def __init__(self, in_c=8):
        super().__init__()

        self.stem = nn.Sequential(
            ComplexConv2d(in_c, 32, 3, 1, 1),
            ComplexBatchNorm2d(32),
            ComplexReLU()
        )

        self.stage1 = nn.Sequential(
            ComplexConv2d(32, 64, 3, 2, 1),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexResBlock(64),
            ComplexResBlock(64)
        )

        self.stage2 = nn.Sequential(
            ComplexConv2d(64, 128, 3, 1, 1),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexResBlock(128),
            ComplexResBlock(128)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


# ============================================================
# COMPLEX SELF ATTENTION (STABLE)
# ============================================================

class ComplexSelfAttention(nn.Module):

    def __init__(self, channels=128, attn_dropout=0.1):
        super().__init__()

        self.bn = ComplexBatchNorm2d(channels)

        self.q = ComplexConv2d(channels, channels, 1)
        self.k = ComplexConv2d(channels, channels, 1)
        self.v = ComplexConv2d(channels, channels, 1)

        self.scale = channels ** -0.5

        # Smaller gamma for stability
        self.gamma = nn.Parameter(torch.tensor(0.05))

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, C, H, W, _ = x.shape
        N = H * W

        x = self.bn(x)

        q = self.q(x).view(B, C, N, 2).permute(0, 2, 1, 3)
        k = self.k(x).view(B, C, N, 2).permute(0, 2, 1, 3)
        v = self.v(x).view(B, C, N, 2).permute(0, 2, 1, 3)

        # Conjugate transpose
        kr, ki = complex_demodulate(k)
        k_conj = complex_modulate(kr, -ki)
        k_t = k_conj.permute(0, 2, 1, 3)

        attn = complex_matmul(q, k_t)

        # KEEP Re(QKᴴ)
        attn_real, _ = complex_demodulate(attn)
        attn_real = attn_real * self.scale

        attn_weights = torch.softmax(attn_real, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_complex = complex_modulate(attn_weights, torch.zeros_like(attn_weights))

        out = complex_matmul(attn_complex, v)
        out = out.permute(0, 2, 1, 3).view(B, C, H, W, 2)

        return x + self.gamma * out


# ============================================================
# COMPLEX DOMAIN FUSION (STABLE + RESIDUAL CHANNEL ATTENTION)
# ============================================================

class ComplexDomainFusion(nn.Module):

    def __init__(self, channels=128):
        super().__init__()

        # Deeper gating
        self.gate = nn.Sequential(
            ComplexConv2d(channels * 3, 128, 1),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexConv2d(128, 3, 1)
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            ComplexAdaptiveAvgPool2d(1),
            ComplexConv2d(channels, channels // 4, 1),
            ComplexReLU(),
            ComplexConv2d(channels // 4, channels, 1)
        )

    def forward(self, zr, zf, zw):

        z = torch.cat([zr, zf, zw], dim=1)

        g_complex = self.gate(z)
        g = complex_abs(g_complex)
        g = torch.softmax(g, dim=1)

        gr = g[:, 0:1].unsqueeze(-1)
        gf = g[:, 1:2].unsqueeze(-1)
        gw = g[:, 2:3].unsqueeze(-1)

        fused = gr * zr + gf * zf + gw * zw

        # Stable channel attention (residual)
        ca = self.channel_attn(fused)
        ca = complex_abs(ca)
        ca = torch.sigmoid(ca).unsqueeze(-1)

        fused = fused * ca + fused   # residual stabilization

        return fused


# ============================================================
# CLASSIFICATION HEAD (SAFE)
# ============================================================

class ComplexClassificationHead(nn.Module):

    def __init__(self, in_dim=128, num_classes=7):
        super().__init__()

        self.pool = ComplexAdaptiveAvgPool2d(1)
        self.fc = ComplexLinear(in_dim, num_classes)

        # Reduced dropout for stability
        self.dropout = nn.Dropout(0.1)

    def forward(self, z):

        z = self.pool(z)
        z = z.view(z.shape[0], -1, 2)

        logits_complex = self.fc(z)

        # Correct complex-safe dropout
        real = self.dropout(logits_complex[..., 0])
        imag = self.dropout(logits_complex[..., 1])

        logits_complex = torch.stack([real, imag], dim=-1)

        logits = complex_abs(logits_complex)

        return logits


# ============================================================
# COMPLETE MODEL
# ============================================================

class ComplexEndToEndModel(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.er = DeepComplexEncoder(2)
        self.ef = DeepComplexEncoder(2)
        self.ew = ComplexWaveletEncoder(8)

        self.ar = ComplexSelfAttention(128)
        self.af = ComplexSelfAttention(128)
        self.aw = ComplexSelfAttention(128)

        self.fusion = ComplexDomainFusion(128)

        self.head = ComplexClassificationHead(128, num_classes)

    def forward(self, r, f, w):

        zr = self.ar(self.er(r))
        zf = self.af(self.ef(f))
        zw = self.aw(self.ew(w))

        z = self.fusion(zr, zf, zw)

        logits = self.head(z)

        return logits
