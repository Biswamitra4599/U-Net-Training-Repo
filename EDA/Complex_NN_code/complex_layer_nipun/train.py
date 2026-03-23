import os
import random
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from dataset import S1SLCDataset
from models import ComplexEndToEndModel
from label_mappings import NUM_CLASSES, CLASS_NAMES
from logger import Logger


# REPRODUCIBILITY

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# WARMUP + COSINE SCHEDULER
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    size_mb = total_params * 4 / (1024 ** 2)  # assuming float32 (4 bytes)

    print("\n" + "=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {frozen_params:,}")
    print(f"Approx Model Size    : {size_mb:.2f} MB (float32)")
    print("=" * 60 + "\n")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / \
                       (self.max_epochs - self.warmup_epochs)
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


# TRAIN FUNCTION

def train(config):

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG = Logger(config.log_file)
    LOG.separator()
    LOG.log("FULL COMPLEX END-TO-END CLASSIFICATION (ECCV READY)")
    LOG.log(str(vars(config)))
    LOG.separator()
    LOG.log(f"Device: {device}")
    LOG.separator()

    # Dataset

    dataset = S1SLCDataset.from_root(config.data_path)
    split = torch.load(config.split_file)

    train_ds = Subset(dataset, split["train"])
    val_ds   = Subset(dataset, split["val"])
    test_ds  = Subset(dataset, split["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model

    model = ComplexEndToEndModel(num_classes=NUM_CLASSES)
    print_model_parameters(model)

    model = model.to(device)

    # Loss

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer

    # If the model is wrapped in DataParallel, we should access the parameters via model.module
    # However, nn.DataParallel forwards attribute access to the module, but let's be safe
    # and use model.parameters() which works correctly for both cases as DataParallel is an nn.Module

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.epochs
    )

    # Tracking

    best_val_acc = 0.0
    early_stop_counter = 0

    train_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    val_f1_hist = []

    # TRAIN LOOP

    for epoch in range(config.epochs):

        LOG.separator()
        LOG.log(f"Epoch {epoch+1}/{config.epochs}")
        LOG.separator()

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="Train", dynamic_ncols=True)

        for batch in train_bar:

            r = batch['raw'].to(device, non_blocking=True)
            f = batch['fourier'].to(device, non_blocking=True)
            w = batch['wavelet'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)

            logits = model(r, f, w)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # VALIDATION

        model.eval()
        val_correct = 0
        val_total = 0

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:

                r = batch['raw'].to(device)
                f = batch['fourier'].to(device)
                w = batch['wavelet'].to(device)
                y = batch['label'].to(device).view(-1)

                logits = model(r, f, w)

                preds = logits.argmax(dim=1)

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")

        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        LOG.log(
            f"Epoch {epoch+1:03d} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val F1: {val_f1:.4f}"
        )

        # BEST MODEL

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save unwrapped model for compatibility
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), config.ckpt_out)
            else:
                torch.save(model.state_dict(), config.ckpt_out)
                
            early_stop_counter = 0
            LOG.log(f"  ↳ New best model saved ({val_acc:.2f}%)")
        else:
            early_stop_counter += 1

        # EARLY STOP

        if early_stop_counter >= config.early_stop_patience:
            LOG.log("Early stopping triggered.")
            break

    # SAVE TRAIN CURVES

    plt.figure()
    plt.plot(train_loss_hist, label="Train Loss")
    plt.legend()
    plt.savefig("train_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(val_f1_hist, label="Val F1")
    plt.legend()
    plt.savefig("val_f1_curve.png")
    plt.close()

    # TEST

    LOG.separator()
    LOG.log("Testing Best Model")
    LOG.separator()

    # Load best checkpoint
    checkpoint = torch.load(config.ckpt_out, map_location=device)
    
    # If model is wrapped (multi-GPU) but checkpoint is unwrapped (single-GPU style),
    # load into model.module.
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:

            r = batch['raw'].to(device)
            f = batch['fourier'].to(device)
            w = batch['wavelet'].to(device)
            y = batch['label'].to(device).view(-1)

            logits = model(r, f, w)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = 100 * np.mean(all_preds == all_labels)
    test_f1 = f1_score(all_labels, all_preds, average="macro")

    LOG.log(f"TEST ACCURACY: {test_acc:.2f}%")
    LOG.log(f"TEST MACRO F1: {test_f1:.4f}")

    # CONFUSION MATRIX

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    LOG.separator()
    LOG.log("Training + Evaluation Complete")
    LOG.separator()

    return best_val_acc



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="data_split.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt_out", type=str, default="complex_best.pth")
    parser.add_argument("--log_file", type=str, default="complex_log.txt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=6)

    args = parser.parse_args()

    train(args)
