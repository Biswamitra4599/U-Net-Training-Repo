
import torch

def complex_modulate(real, imag):
    """Stack real and imaginary parts along the last dimension."""
    return torch.stack([real, imag], dim=-1)

def complex_demodulate(x):
    """Unstack the last dimension to get real and imaginary parts."""
    return x[..., 0], x[..., 1]

def complex_abs(x):
    """Compute the complex magnitude."""
    r, i = complex_demodulate(x)
    return torch.sqrt(r*r + i*i + 1e-8)

def complex_angle(x):
    """Compute the complex phase."""
    r, i = complex_demodulate(x)
    return torch.atan2(i, r)

def complex_mul(x, y):
    """
    Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    """
    xr, xi = complex_demodulate(x)
    yr, yi = complex_demodulate(y)
    
    real = xr * yr - xi * yi
    imag = xr * yi + xi * yr
    
    return complex_modulate(real, imag)

def complex_matmul(A, B):
    """
    Complex matrix multiplication.
    A: (..., M, K, 2)
    B: (..., K, N, 2)
    """
    Ar, Ai = complex_demodulate(A)
    Br, Bi = complex_demodulate(B)
    
    real = torch.matmul(Ar, Br) - torch.matmul(Ai, Bi)
    imag = torch.matmul(Ar, Bi) + torch.matmul(Ai, Br)
    
    return complex_modulate(real, imag)
