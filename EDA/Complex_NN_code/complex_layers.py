

import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_utils import (
    complex_modulate,
    complex_demodulate,
)


# #imp for convs
# class ComplexConv2d(nn.Module):

#     def __init__(self, in_c, out_c, k, s=1, p=0, bias=True):
#         super().__init__()

#         self.r = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
#         self.i = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)

#         nn.init.xavier_normal_(self.r.weight)
#         nn.init.xavier_normal_(self.r.weight)

#     def forward(self, x):

#         r, i = complex_demodulate(x)

#         yr = self.r(r) - self.i(i)
#         yi = self.r(i) + self.i(r)
        
#         return complex_modulate(yr, yi)
    
# class ComplexTransposeConv2d(nn.Module):

#     def __init__(self, in_c, out_c, k, s=1, p=0, bias=True):
#         super().__init__()

#         self.r = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=bias)
#         self.i = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=bias)

#         nn.init.xavier_normal_(self.r.weight)
#         nn.init.xavier_normal_(self.r.weight)

#     def forward(self, x):

#         r, i = complex_demodulate(x)

#         yr = self.r(r) - self.i(i)
#         yi = self.r(i) + self.i(r)

#         return complex_modulate(yr, yi)


# class ComplexLinear(nn.Module):

#     def __init__(self, in_f, out_f, bias=True):
#         super().__init__()

#         self.r = nn.Linear(in_f, out_f, bias=bias)
#         self.i = nn.Linear(in_f, out_f, bias=bias)

#     def forward(self, x):

#         r, i = complex_demodulate(x)

#         yr = self.r(r) - self.i(i)
#         yi = self.r(i) + self.i(r)

#         return complex_modulate(yr, yi)



class _ComplexBatchNorm(nn.Module):
    """
    Base class for Complex Batch Normalization.
    Keeps running statistics for 2x2 covariance matrix elements.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # We need 3 components for 2x2 weight matrix (symmetric) and 2 for bias
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2)) # Real, Imag mean
            self.register_buffer('running_covar', torch.zeros(num_features, 3)) # Vrr, Vii, Vri
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

            self.running_covar[:, 0] = 0.5
            self.running_covar[:, 1] = 0.5

        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 0.5
            self.running_covar[:, 1] = 0.5
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 0.70710678)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)




class ComplexBatchNorm2d(_ComplexBatchNorm):
    """
    Complex Batch Normalization for simulated complex tensors.
    Input: (B, C, H, W, 2)
    """

    def forward(self, input):
        # Input shape: (B, C, H, W, 2)
        # We need to treat the last dimension as real/imag parts
        
        # Extract Real and Imaginary parts
        real = input[..., 0]
        imag = input[..., 1]
        
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                     cma_factor = 1.0 / float(self.num_batches_tracked)
                     if cma_factor > self.momentum:
                         exponential_average_factor = cma_factor
                     else:
                         exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # Mean calculation
            mean_r = real.mean([0, 2, 3])
            mean_i = imag.mean([0, 2, 3])
            
            # Stack mean for broadcasting: (C, 2)
            mean = torch.stack([mean_r, mean_i], dim=1)
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # Update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                    + (1 - exponential_average_factor) * \
                    self.running_mean

        input_centered = input - mean[None, :, None, None, :]
        real_centered = input_centered[..., 0]
        imag_centered = input_centered[..., 1]

        if self.training or (not self.training and not self.track_running_stats):
            # Covariance matrix elements
            n = input.numel() / input.size(1) / 2 # Divide by 2 because complex count
            # Actually n should be B*H*W
            n = real.numel() / real.size(1)
            
            Crr = 1. / n * real_centered.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * imag_centered.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (real_centered.mul(imag_centered)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 2]

        # Inverse square root of covariance matrix
        det = Crr * Cii - Cri.pow(2)
        #add a small value to det to avoid zero division
        det = torch.clamp(det, min=1e-8)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        Rrr = Rrr[None, :, None, None]
        Rii = Rii[None, :, None, None]
        Rri = Rri[None, :, None, None]
        
        out_real = Rrr * real_centered + Rri * imag_centered
        out_imag = Rii * imag_centered + Rri * real_centered

        if self.affine:
            
            Wrr = self.weight[None, :, 0, None, None]
            Wii = self.weight[None, :, 1, None, None]
            Wri = self.weight[None, :, 2, None, None]
            
            Br = self.bias[None, :, 0, None, None]
            Bi = self.bias[None, :, 1, None, None]
            
            final_real = Wrr * out_real + Wri * out_imag + Br
            final_imag = Wii * out_imag + Wri * out_real + Bi
            
            return complex_modulate(final_real, final_imag)
        else:
            return complex_modulate(out_real, out_imag)


class ComplexBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:

            self.weight = nn.Parameter(
                torch.eye(2).unsqueeze(0).repeat(num_features, 1, 1)
            )

            self.bias = nn.Parameter(
                torch.zeros(num_features, 2)
            )

        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer(
            "running_mean",
            torch.zeros(num_features, 2)
        )

        self.register_buffer(
            "running_covar",
            torch.eye(2).unsqueeze(0).repeat(num_features, 1, 1)
        )


    def forward(self, x):

        if torch.is_complex(x):
            B, C, H, W = x.shape
            xr = x.real
            xi = x.imag
            # raise Exception("Complex Number Came and Stopping") 
        else:
            B, C, H, W, _ = x.shape
            xr = x[..., 0]
            xi = x[..., 1]

        xr = xr.reshape(B, C, -1)
        xi = xi.reshape(B, C, -1)


        # Mean
        if self.training:

            mean_r = xr.mean(dim=(0, 2))
            mean_i = xi.mean(dim=(0, 2))

            mean = torch.stack([mean_r, mean_i], dim=-1)

        else:

            mean = self.running_mean


        # Center
        xr_c = xr - mean[None, :, 0, None]
        xi_c = xi - mean[None, :, 1, None]


        # Covariance
        if self.training:

            cov_rr = (xr_c * xr_c).mean(dim=(0, 2)) + self.eps
            cov_ii = (xi_c * xi_c).mean(dim=(0, 2)) + self.eps
            cov_ri = (xr_c * xi_c).mean(dim=(0, 2))

            cov = torch.stack(
                [
                    torch.stack([cov_rr, cov_ri], dim=-1),
                    torch.stack([cov_ri, cov_ii], dim=-1),
                ],
                dim=-2,
            )

        else:

            cov = self.running_covar


        # Update running stats
        if self.training:

            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(self.momentum * mean.detach())

            self.running_covar.mul_(1 - self.momentum)
            self.running_covar.add_(self.momentum * cov.detach())


        # Whitening
        # L = torch.linalg.cholesky(cov) 
        # Manual 2x2 Cholesky for stability and to avoid "lazy wrapper" errors in DP[-]
        a = cov[:, 0, 0]
        b = cov[:, 0, 1]
        c = cov[:, 1, 1]

        l00 = torch.sqrt(a)
        l10 = b / (l00 + 1e-12)
        l11 = torch.sqrt(torch.clamp(c - l10**2, min=1e-12))
        
        # Construct L (C, 2, 2)
        zeros = torch.zeros_like(l00)
        L = torch.stack([
            torch.stack([l00, zeros], dim=-1),
            torch.stack([l10, l11], dim=-1)
        ], dim=-2)

        vec = torch.stack([xr_c, xi_c], dim=2)

        # normed = torch.linalg.solve_triangular(
        #     L[None, :, :, :],
        #     vec,
        #     upper=False
        # )
        
        # Manual Forward Substitution for L * x = vec => x = L^-1 * vec
        # vec is (B, C, 2, N) where N = H*W
        # L is (C, 2, 2) implicitly via l00, l10, l11
        
        # reshapes for broadcasting
        # vec shape: (B, C, 2, N)
        v0 = vec[:, :, 0, :]
        v1 = vec[:, :, 1, :]
        
        # l parameters shape: (C,) -> (1, C, 1)
        _l00 = l00.view(1, -1, 1)
        _l10 = l10.view(1, -1, 1)
        _l11 = l11.view(1, -1, 1)
        
        x0 = v0 / (_l00 + 1e-12)
        x1 = (v1 - _l10 * x0) / (_l11 + 1e-12)
        
        normed = torch.stack([x0, x1], dim=2)


        # Affine (correct)
        if self.affine:

            normed = torch.einsum(
                "bcin,cij->bcjn",
                normed,
                self.weight
            ) + self.bias[None, :, :, None]


        # Reshape
        normed = normed.reshape(B, C, 2, H, W)
        
        # Return complex tensor
        # return complex_modulate(
        #     normed[:, :, 0],
        #     normed[:, :, 1]
        # )
        return torch.complex(normed[:, :, 0], normed[:, :, 1])




class ComplexReLU(nn.Module):

    def forward(self, x):

        r, i = complex_demodulate(x)

        return complex_modulate(
            F.relu(r),
            F.relu(i)
        )


class ComplexAvgPool2d(nn.Module):

    def __init__(self, k=2, s=2):
        super().__init__()
        self.pool = nn.AvgPool2d(k, s)

    def forward(self, x):

        r, i = complex_demodulate(x)

        return complex_modulate(
            self.pool(r),
            self.pool(i)
        )


class ComplexAdaptiveAvgPool2d(nn.Module):

    def __init__(self, size=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):

        r, i = complex_demodulate(x)

        return complex_modulate(
            self.pool(r),
            self.pool(i)
        )



class ComplexResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.c1 = ComplexConv2d(c, c, 3, 1, 1)
        self.bn1 = ComplexBatchNorm2d(c)

        self.c2 = ComplexConv2d(c, c, 3, 1, 1)
        self.bn2 = ComplexBatchNorm2d(c)

        self.a = ComplexReLU()

    def forward(self, x):

        z = self.a(self.bn1(self.c1(x)))
        z = self.bn2(self.c2(z))

        return self.a(z + x)
