# # complex_utils.py

# import torch


# def complex_modulate(real, imag):
#     return torch.complex(real, imag)

# # def complex_modulate(real, imag):
# #     return torch.complex(real, imag)


# def complex_demodulate(x):
#     if torch.is_complex(x):
#         return x.real, x.imag
#     else:
#         return x[..., 0], x[..., 1]


# def complex_abs(x):
#     r, i = complex_demodulate(x)
#     return torch.sqrt(r*r + i*i + 1e-8)


# def complex_angle(x):
#     r, i = complex_demodulate(x)
#     return torch.atan2(i, r)

# def complex_matmul(A, B):
#     return torch.matmul(A, B)
