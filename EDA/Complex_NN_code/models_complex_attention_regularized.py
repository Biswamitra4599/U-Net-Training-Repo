import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_utils import complex_abs
from complex_layers import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    ComplexReLU,
    ComplexResBlock,
    ComplexLinear,
    ComplexAdaptiveAvgPool2d
)

# ============================================================
# Parameter Counter
# ============================================================

def print_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n================ PARAMETER COUNT ================")
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
    print("=================================================\n")


# ============================================================
# Encoder
# ============================================================

class DeepComplexEncoder(nn.Module):
    def __init__(self, in_c):
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
            ComplexConv2d(64, 128, 3, 2, 1),
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
# Self-Attention (With Dropout + Residual Scaling)
# ============================================================

class ComplexSelfAttention(nn.Module):
    def __init__(self, dim, attn_dropout=0.2):
        super().__init__()

        self.q = ComplexLinear(dim, dim)
        self.k = ComplexLinear(dim, dim)
        self.v = ComplexLinear(dim, dim)

        self.norm = ComplexBatchNorm2d(dim)
        self.scale = dim ** 0.5

        self.dropout = nn.Dropout(attn_dropout)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # residual scaling

        self.last_attn = None

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W

        x_flat = x.view(B, C, N).permute(0, 2, 1)

        Q = self.q(x_flat)
        K = self.k(x_flat)
        V = self.v(x_flat)

        K_conj = torch.conj(K)

        attn = torch.matmul(Q, K_conj.transpose(1, 2))
        attn_real = attn.real / self.scale

        attn_weights = torch.softmax(attn_real, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.to(V.dtype)

        self.last_attn = attn_weights.detach()

        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        x = x + self.gamma * out
        x = self.norm(x)

        return x


# ============================================================
# Cross-Attention (With Dropout + Residual Scaling)
# ============================================================

class ComplexCrossAttention(nn.Module):
    def __init__(self, dim, attn_dropout=0.2):
        super().__init__()

        self.q = ComplexLinear(dim, dim)
        self.k = ComplexLinear(dim, dim)
        self.v = ComplexLinear(dim, dim)

        self.scale = dim ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, target, source):

        B, C, Ht, Wt = target.shape
        Nt = Ht * Wt

        _, _, Hs, Ws = source.shape
        Ns = Hs * Ws

        t_flat = target.view(B, C, Nt).permute(0, 2, 1)
        s_flat = source.view(B, C, Ns).permute(0, 2, 1)

        Q = self.q(t_flat)
        K = self.k(s_flat)
        V = self.v(s_flat)

        K_conj = torch.conj(K)

        attn = torch.matmul(Q, K_conj.transpose(1, 2))
        attn_real = attn.real / self.scale

        attn_weights = torch.softmax(attn_real, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.to(V.dtype)

        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1).view(B, C, Ht, Wt)

        return self.gamma * out


# ============================================================
# Domain Fusion
# ============================================================

class DomainFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.pool = ComplexAdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim * 3, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, zr, zf, zw):

        pr = complex_abs(self.pool(zr)).flatten(1)
        pf = complex_abs(self.pool(zf)).flatten(1)
        pw = complex_abs(self.pool(zw)).flatten(1)

        x = torch.cat([pr, pf, pw], dim=1)

        gates = torch.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

        wr = gates[:, 0].view(-1, 1, 1, 1)
        wf = gates[:, 1].view(-1, 1, 1, 1)
        ww = gates[:, 2].view(-1, 1, 1, 1)

        zr_p = self.pool(zr)
        zf_p = self.pool(zf)
        zw_p = self.pool(zw)

        z = wr * zr_p + wf * zf_p + ww * zw_p
        return z


# ============================================================
# Full Model
# ============================================================

class TriDomainComplexAttentionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.er = DeepComplexEncoder(2)
        self.ef = DeepComplexEncoder(2)
        self.ew = DeepComplexEncoder(8)

        self.attn_r = ComplexSelfAttention(128)
        self.attn_f = ComplexSelfAttention(128)
        self.attn_w = ComplexSelfAttention(128)

        self.cross_rf = ComplexCrossAttention(128)
        self.cross_rw = ComplexCrossAttention(128)
        self.cross_fr = ComplexCrossAttention(128)
        self.cross_fw = ComplexCrossAttention(128)
        self.cross_wr = ComplexCrossAttention(128)
        self.cross_wf = ComplexCrossAttention(128)

        self.cross_norm_r = ComplexBatchNorm2d(128)
        self.cross_norm_f = ComplexBatchNorm2d(128)
        self.cross_norm_w = ComplexBatchNorm2d(128)

        self.fusion = DomainFusion(128)

        self.pool = ComplexAdaptiveAvgPool2d(1)

        self.fc1 = ComplexLinear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.act = ComplexReLU()
        self.fc2 = ComplexLinear(256, num_classes)

        print_parameter_count(self)

    def forward(self, r, f, w):

        zr = self.er(r)
        zf = self.ef(f)
        zw = self.ew(w)

        zr = self.attn_r(zr)
        zf = self.attn_f(zf)
        zw = self.attn_w(zw)

        zr_new = zr + self.cross_rf(zr, zf) + self.cross_rw(zr, zw)
        zf_new = zf + self.cross_fr(zf, zr) + self.cross_fw(zf, zw)
        zw_new = zw + self.cross_wr(zw, zr) + self.cross_wf(zw, zf)

        zr = self.cross_norm_r(zr_new)
        zf = self.cross_norm_f(zf_new)
        zw = self.cross_norm_w(zw_new)

        z = self.fusion(zr, zf, zw)

        z = self.pool(z).flatten(1)

        z = self.act(self.fc1(z))
        # nn.Dropout doesn't support ComplexFloat; apply to real & imag separately
        z = torch.complex(self.dropout(z.real), self.dropout(z.imag))
        z = self.fc2(z)

        logits = complex_abs(z)

        return logits, z
