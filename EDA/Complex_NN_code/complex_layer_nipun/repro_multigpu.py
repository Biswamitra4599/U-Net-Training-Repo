# repro_multigpu.py
import torch
import torch.nn as nn
from complex_layers import ComplexBatchNorm2d
import os

def repro_multigpu():
    print("Checking Multi-GPU behavior...", flush=True)
    if torch.cuda.device_count() < 2:
        print("Not enough GPUs to test DataParallel issue!", flush=True)
        return

    device = torch.device("cuda")
    
    bn = ComplexBatchNorm2d(10).to(device)
    bn.train()
    
    print(f"Init Mean (Main): {bn.running_mean[0]}", flush=True)
    
    model = nn.DataParallel(bn, device_ids=[0, 1])
    
    x = torch.zeros(2, 10, 4, 4, 2).to(device)
    x[..., 0] = 10.0 # Real part 10
    

    _ = model(x)
    
    print(f"Post-Forward Mean (Main): {bn.running_mean[0]}", flush=True)
    
    if bn.running_mean[0, 0] > 1.0:
        print("SURPRISE: DataParallel synced stats!", flush=True)
    else:
        print("CONFIRMED: DataParallel did NOT sync stats. Main model mean is still ~0.", flush=True)

if __name__ == "__main__":
    repro_multigpu()
