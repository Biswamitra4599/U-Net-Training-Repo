import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dataset import S1SLCDataset
from models import ComplexEndToEndModel
from complex_utils import complex_abs
from label_mappings import CLASS_NAMES


# Utils

def normalize(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)

def attn_to_map(attn):
    attn = attn[0].cpu().numpy()
    N = attn.shape[0]
    s = int(np.sqrt(N))
    center = (s//2)*s + (s//2)
    return attn[center].reshape(s, s)

from scipy.ndimage import zoom

def overlay_map(img, attn):
    """
    Overlay attention map on input image.

    img  : (H, W)
    attn : (h, w)   (typically 25x25)

    Output:
        RGB overlay (H, W, 3)
    """

    img_n = normalize(img)

    # --- upsample attention to match input resolution ---
    H, W = img.shape
    h, w = attn.shape

    scale_h = H / h
    scale_w = W / w

    attn_up = zoom(attn, (scale_h, scale_w), order=1)

    # normalize attention map
    attn_up = normalize(attn_up)

    heat = plt.cm.jet(attn_up)[..., :3]
    img_rgb = np.stack([img_n]*3, axis=-1)

    overlay = 0.5 * img_rgb + 0.5 * heat

    return overlay


# MAIN

def run_analysis(data_path, split_file, checkpoint, out_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = S1SLCDataset.from_root(data_path)
    split = torch.load(split_file, map_location="cpu")

    all_test_indices = split["test"]
    random.seed(42)
    test_indices = random.sample(all_test_indices, 300)
    subset = Subset(dataset, test_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    model = ComplexEndToEndModel(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # create folders
    os.makedirs(out_dir, exist_ok=True)
    class_dirs = {}
    for c, name in enumerate(CLASS_NAMES):
        d = os.path.join(out_dir, f"class_{c}_{name.replace(' ', '_')}")
        os.makedirs(d, exist_ok=True)
        class_dirs[c] = d

    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # stats storage
    num_classes = len(CLASS_NAMES)
    per_class_gates = [[] for _ in range(num_classes)]
    per_class_correct = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for i, batch in enumerate(loader):

            r = batch["raw"].to(device)
            f = batch["fourier"].to(device)
            w = batch["wavelet"].to(device)
            y = batch["label"].item()

            logits = model(r, f, w)
            pred = logits.argmax(dim=1).item()

            fusion = model.fusion

            zr = fusion.last_zr[0]
            zf = fusion.last_zf[0]
            zw = fusion.last_zw[0]

            gr = fusion.last_gr[0]
            gf = fusion.last_gf[0]
            gw = fusion.last_gw[0]

            gates = fusion.last_gates[0].cpu().numpy()

            mean_raw, mean_fourier, mean_wavelet = gates.mean(axis=(1,2))

            # accumulate stats
            per_class_gates[y].append([mean_raw, mean_fourier, mean_wavelet])
            per_class_correct[y].append(int(pred == y))

            # maps
            enc_r = complex_abs(zr).cpu().numpy()[0]
            enc_f = complex_abs(zf).cpu().numpy()[0]
            enc_w = complex_abs(zw).cpu().numpy()[0]

            attn_r = attn_to_map(model.ar.last_attn)
            attn_f = attn_to_map(model.af.last_attn)
            attn_w = attn_to_map(model.aw.last_attn)

            out_r = complex_abs(gr * zr).cpu().numpy()[0]
            out_f = complex_abs(gf * zf).cpu().numpy()[0]
            out_w = complex_abs(gw * zw).cpu().numpy()[0]

            img_r = complex_abs(r)[0,0].cpu().numpy()
            img_f = complex_abs(f)[0,0].cpu().numpy()
            img_w = complex_abs(w)[0,0].cpu().numpy()

            # MAIN GRID

            fig, axes = plt.subplots(3,5, figsize=(20,12))

            def show(ax, data, title):
                im = ax.imshow(data, cmap="viridis")
                ax.set_title(title)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # RAW
            show(axes[0,0], img_r, "Raw Input")
            show(axes[0,1], enc_r, "Encoder+Attn")
            show(axes[0,2], attn_r, "Self-Attn")
            show(axes[0,3], gates[0], f"Gate ({mean_raw:.3f})")
            show(axes[0,4], out_r, "Gate×Output")

            # FOURIER
            show(axes[1,0], img_f, "Fourier Input")
            show(axes[1,1], enc_f, "Encoder+Attn")
            show(axes[1,2], attn_f, "Self-Attn")
            show(axes[1,3], gates[1], f"Gate ({mean_fourier:.3f})")
            show(axes[1,4], out_f, "Gate×Output")

            # WAVELET
            show(axes[2,0], img_w, "Wavelet Input")
            show(axes[2,1], enc_w, "Encoder+Attn")
            show(axes[2,2], attn_w, "Self-Attn")
            show(axes[2,3], gates[2], f"Gate ({mean_wavelet:.3f})")
            show(axes[2,4], out_w, "Gate×Output")

            fig.suptitle(f"{CLASS_NAMES[y]} | Pred: {CLASS_NAMES[pred]}",
                         color=("green" if pred==y else "red"),
                         fontsize=16)

            save_base = os.path.join(class_dirs[y], f"sample_{i:04d}")
            plt.tight_layout()
            plt.savefig(save_base + "_main.png")
            plt.close()

            # OVERLAY

            fig2, ax2 = plt.subplots(1,3, figsize=(12,4))
            ax2[0].imshow(overlay_map(img_r, attn_r))
            ax2[1].imshow(overlay_map(img_f, attn_f))
            ax2[2].imshow(overlay_map(img_w, attn_w))
            for a in ax2: a.axis("off")
            plt.savefig(save_base + "_overlay.png")
            plt.close()

            # RGB COMPOSITE

            fig3, ax3 = plt.subplots(1,4, figsize=(16,4))
            ax3[0].imshow(enc_r, cmap="inferno")
            ax3[1].imshow(enc_f, cmap="inferno")
            ax3[2].imshow(enc_w, cmap="inferno")

            rgb = np.stack([
                normalize(enc_r),
                normalize(enc_f),
                normalize(enc_w)
            ], axis=-1)

            ax3[3].imshow(rgb)
            for a in ax3: a.axis("off")

            plt.savefig(save_base + "_rgb.png")
            plt.close()

            # GATE HISTOGRAM

            plt.figure(figsize=(6,4))
            plt.hist(gates[0].flatten(), bins=50, alpha=0.5, label="Raw")
            plt.hist(gates[1].flatten(), bins=50, alpha=0.5, label="Fourier")
            plt.hist(gates[2].flatten(), bins=50, alpha=0.5, label="Wavelet")
            plt.legend()
            plt.savefig(save_base + "_hist.png")
            plt.close()

    # SUMMARY ANALYSIS

    class_means = []
    for c in range(num_classes):
        arr = np.array(per_class_gates[c])
        mean = arr.mean(axis=0)
        class_means.append(mean)

        plt.figure()
        plt.bar(["Raw","Fourier","Wavelet"], mean)
        plt.title(CLASS_NAMES[c])
        plt.savefig(os.path.join(summary_dir, f"class_{c}_mean.png"))
        plt.close()

    # heatmap
    heat = np.array(class_means)
    plt.imshow(heat, cmap="viridis")
    plt.xticks([0,1,2], ["Raw","Fourier","Wavelet"])
    plt.yticks(range(num_classes), CLASS_NAMES)
    plt.colorbar()
    plt.title("Class vs Domain Importance")
    plt.savefig(os.path.join(summary_dir, "class_domain_heatmap.png"))
    plt.close()

    print("✔ Full analysis complete. Results saved to:", out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    run_analysis(
        data_path=args.data_path,
        split_file=args.split_file,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir
    )
