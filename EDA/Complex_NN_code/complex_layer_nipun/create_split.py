import torch
from dataset import S1SLCDataset

DATA_PATH = "/home/neha/Aashutosh-Joshi/S1SLC_CVDL"
SPLIT_FILE = "data_split.pth"
SEED = 42

torch.manual_seed(SEED)

ds = S1SLCDataset.from_root(DATA_PATH)
n = len(ds)

n_train = int(0.7 * n)
n_val   = int(0.15 * n)

indices = torch.randperm(n).tolist()

split = {
    "train": indices[:n_train],
    "val":   indices[n_train:n_train + n_val],
    "test":  indices[n_train + n_val:]
}

torch.save(split, SPLIT_FILE)

print("Fixed dataset split saved to", SPLIT_FILE)
print(f"Train: {len(split['train'])}")
print(f"Val:   {len(split['val'])}")
print(f"Test:  {len(split['test'])}")
