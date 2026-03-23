
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_utils import (
    complex_modulate,
    complex_demodulate,
    complex_abs,
    complex_angle
)


class ComplexConv2d(nn.Module):
    """
    Simulated Complex Convolution using two real-valued Conv2d layers.
    Input: (B, C, H, W, 2)
    Output: (B, C_out, H_out, W_out, 2)
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, bias=False):
        super().__init__()

        self.r = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
        self.i = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)

    def forward(self, x):
        # x: (B, C, H, W, 2)
        r, i = complex_demodulate(x)

        # Complex multiplication rule: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # Weights (w_r + i*w_i)
        # Input (x_r + i*x_i)
        
        # Real part of output: x_r*w_r - x_i*w_i
        yr = self.r(r) - self.i(i)
        
        # Imaginary part of output: x_r*w_i + x_i*w_r
        yi = self.r(i) + self.i(r)

        return complex_modulate(yr, yi)


class ComplexLinear(nn.Module):
    """
    Simulated Complex Linear layer.
    Input: (..., in_f, 2)
    Output: (..., out_f, 2)
    """

    def __init__(self, in_f, out_f, bias=True):
        super().__init__()

        self.r = nn.Linear(in_f, out_f, bias=bias)
        self.i = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        r, i = complex_demodulate(x)

        yr = self.r(r) - self.i(i)
        yi = self.r(i) + self.i(r)

        return complex_modulate(yr, yi)


class ComplexReLU(nn.Module):
    """
    EqReLU(x) = {
                    x - if angle(x) belongs to [-pi/2, pi/2]
                    0 - otherwise
                }

    In the Argand plane, if the complex number lies in the 1st or 4th quadrant 
    (i.e., real part >= 0), its value is retained, else set to zero.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Unpack the (B, C, H, W, 2) tensor
        r, i = complex_demodulate(x)
        
        zeros = torch.zeros_like(r)
        
        # Apply the condition: keep values if real >= 0, otherwise set to 0
        out_r = torch.where(r >= 0, r, zeros)
        out_i = torch.where(r >= 0, i, zeros)
        
        # Repack into the consistent (B, C, H, W, 2) format
        return complex_modulate(out_r, out_i)


class ComplexAvgPool2d(nn.Module):
    """
    Complex Average Pooling.
    """

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
    """
    Complex Adaptive Average Pooling.
    """

    def __init__(self, size=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        r, i = complex_demodulate(x)
        return complex_modulate(
            self.pool(r),
            self.pool(i)
        )


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
        is_native_complex = input.is_complex()
        
        if is_native_complex:
            real = input.real
            imag = input.imag
        else:
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

class ComplexTransposeConv2d(nn.Module):
    """
    Simulated Complex Transposed Convolution.
    Input:  (B, C_in, H, W, 2)
    Output: (B, C_out, H_out, W_out, 2)
    """
    def __init__(self, in_c, out_c, k, s=1, p=0, bias=False):
        super().__init__()
        self.r = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=bias)
        self.i = nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=bias)

    def forward(self, x):
        r, i = complex_demodulate(x)           
        yr = self.r(r) - self.i(i)
        yi = self.r(i) + self.i(r)
        return complex_modulate(yr, yi)
    

class EfficientComplexBatchNorm2d(nn.Module):
    """
    Memory-efficient and numerically stable Complex Batch Normalization.
    Fuses whitening and affine transforms to reduce VRAM usage.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))   
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))     
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))  
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.running_covar[:, 0] = 0.5
            self.running_covar[:, 1] = 0.5
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_covar', None)
            self.register_buffer('num_batches_tracked', None)

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

    def forward(self, input):
        # Stop assuming the input shape. Check it dynamically.
        is_native_complex = input.is_complex()
        
        if is_native_complex:
            real = input.real
            imag = input.imag
        else:
            real = input[..., 0]
            imag = input[..., 1]

        use_batch_stats = self.training or not self.track_running_stats

        if use_batch_stats:
            mean_r = real.mean([0, 2, 3]) # per channel mean
            mean_i = imag.mean([0, 2, 3])
            #centering
            real_c = real - mean_r[None, :, None, None]
            imag_c = imag - mean_i[None, :, None, None]

            n = float(real.numel()) / real.size(1) 
            n_safe = max(n - 1.0, 1.0) 
            bc = n / n_safe
            #applying tikhonov regularization
            Crr = (1.0 / n) * real_c.pow(2).sum(dim=[0, 2, 3]) + self.eps # this is ((real-mean)^2)/n
            Cii = (1.0 / n) * imag_c.pow(2).sum(dim=[0, 2, 3]) + self.eps #same here
            Cri = real_c.mul(imag_c).mean(dim=[0, 2, 3]) #cov -> sum((real-mean)*(comp-mean))/n

            if self.track_running_stats and self.training:
                self.num_batches_tracked += 1
                
                if self.momentum is None:
                    exp_avg_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    cma_factor = 1.0 / float(self.num_batches_tracked)
                    exp_avg_factor = cma_factor if cma_factor > self.momentum else self.momentum
                
                with torch.no_grad():
                    self.running_mean[:, 0].mul_(1 - exp_avg_factor).add_(mean_r * exp_avg_factor)
                    self.running_mean[:, 1].mul_(1 - exp_avg_factor).add_(mean_i * exp_avg_factor)
                    self.running_covar[:, 0].mul_(1 - exp_avg_factor).add_(Crr * bc * exp_avg_factor)
                    self.running_covar[:, 1].mul_(1 - exp_avg_factor).add_(Cii * bc * exp_avg_factor)
                    self.running_covar[:, 2].mul_(1 - exp_avg_factor).add_(Cri * bc * exp_avg_factor)
        else:
            mean_r = self.running_mean[:, 0]
            mean_i = self.running_mean[:, 1]
            real_c = real - mean_r[None, :, None, None]
            imag_c = imag - mean_i[None, :, None, None]
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        det = (Crr * Cii - Cri.pow(2)).clamp(min=self.eps) 
        
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inv_st = 1.0 / (s * t)

        Rrr = (Cii + s) * inv_st
        Rii = (Crr + s) * inv_st
        Rri = -Cri * inv_st

        if self.affine:
            Wrr = self.weight[:, 0]
            Wii = self.weight[:, 1]
            Wri = self.weight[:, 2]

            M00 = (Wrr * Rrr + Wri * Rri)[None, :, None, None]
            M01 = (Wrr * Rri + Wri * Rii)[None, :, None, None]
            M10 = (Wri * Rrr + Wii * Rri)[None, :, None, None]
            M11 = (Wri * Rri + Wii * Rii)[None, :, None, None]

            Br = self.bias[:, 0][None, :, None, None]
            Bi = self.bias[:, 1][None, :, None, None]

            out_real = M00 * real_c + M01 * imag_c + Br
            out_imag = M10 * real_c + M11 * imag_c + Bi
        else:
            Rrr = Rrr[None, :, None, None]
            Rii = Rii[None, :, None, None]
            Rri = Rri[None, :, None, None]

            out_real = Rrr * real_c + Rri * imag_c
            out_imag = Rri * real_c + Rii * imag_c

        # Return the tensor in the exact same format it arrived in
        if is_native_complex:
            return torch.complex(out_real, out_imag)
        else:
            return torch.stack([out_real, out_imag], dim=-1)

class ComplexResBlock(nn.Module):
    """
    Complex Residual Block using simulated complex ops.
    """

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
