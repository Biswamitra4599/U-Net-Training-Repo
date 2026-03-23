
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

def complex_recon_loss(x_hat, x):
    return torch.mean(torch.abs(x_hat - x) ** 2) / (
        torch.mean(torch.abs(x) ** 2) + 1e-8
    )

def classification_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = 100 * np.mean(preds == labels)
    f1 = 100 * f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)
    return acc, f1, cm


