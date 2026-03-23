import argparse
import gc
import glob
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import updated_dataloader as udl


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def free_gpu_memory(*tensors):
    """Explicitly delete tensors and flush GPU + Python memory."""
    for t in tensors:
        del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Generate GT RSS images and sensitivity maps for all .npy files in a directory."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/biswamitra/health/knee_data/val/deconstructed_val",
        help="Directory containing source .npy k-space files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If not set, uses --data-dir.",
    )
    parser.add_argument(
        "--espirit-device",
        type=int,
        default=0,
        help="SigPy device for ESPIRiT (-1 for CPU, 0 for first GPU).",
    )
    parser.add_argument("--calib-width", type=int, default=24)
    parser.add_argument("--thresh", type=float, default=0.02)
    parser.add_argument("--kernel-width", type=int, default=6)
    parser.add_argument("--crop", type=float, default=0.95)
    parser.add_argument(
        "--show-pbar-espirit",
        action="store_true",
        help="Enable ESPIRiT internal progress bar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default, existing pairs are skipped.",
    )
    parser.add_argument(
        "--gpu-clear-every",
        type=int,
        default=50,
        help="Force a full GPU memory clear every N samples (default: 50).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir if args.out_dir is not None else data_dir

    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if len(paths) == 0:
        raise SystemExit(f"No .npy files found in: {data_dir}")

    print(f"Input directory  : {data_dir}")
    print(f"Output directory : {out_dir}")
    print(f"Found files      : {len(paths)}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory (free): {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    dataset = udl.Custom_FMRI_DataLoader_nil(
        data_paths=paths,
        input_req=[0, 0, 0, 0, 0],
        output_req=[0, 0, 1, 0],  # only full_rss_combined
        methods_flags=[0, 1],     # only sensitivity maps via ESPIRiT
        espirit_params=[
            args.calib_width,
            args.thresh,
            args.kernel_width,
            args.crop,
            args.show_pbar_espirit,
        ],
        espirit_device=args.espirit_device,
    )

    mapping_path = os.path.join(out_dir, "generated_index_map.csv")
    write_header = not os.path.exists(mapping_path)

    processed = 0
    skipped = 0
    failed = 0
    failed_indices = []

    t_start = time.perf_counter()

    with open(mapping_path, "a", encoding="utf-8") as fmap:
        if write_header:
            fmap.write("x,source_file,imgGT_file,sensitivity_file\n")

        for i in tqdm(range(len(dataset)), desc="Generating GT + Sensitivity maps"):
            x = i + 1
            img_name = f"imgGT_1_{x}.npy"
            sens_name = f"SensitivityMaps_1_{x}.npy"
            img_path = os.path.join(out_dir, img_name)
            sens_path = os.path.join(out_dir, sens_name)

            src = os.path.basename(paths[i])

            # --- Skip logic: write CSV entry even for skipped files ---
            if (not args.overwrite) and os.path.exists(img_path) and os.path.exists(sens_path):
                fmap.write(f"{x},{src},{img_name},{sens_name}\n")
                skipped += 1
                continue

            try:
                sample = dataset[i]
                gt = to_numpy(sample["full_rss_combined"])
                sens = to_numpy(sample["sensitivity_maps"])

                np.save(img_path, gt)
                np.save(sens_path, sens)

                fmap.write(f"{x},{src},{img_name},{sens_name}\n")
                fmap.flush()  # flush after every write so CSV is not lost on crash
                processed += 1

            except Exception as exc:
                failed += 1
                failed_indices.append((i, paths[i], str(exc)))
                tqdm.write(f"[ERROR] index={i}, file={paths[i]} -> {exc}")

            finally:
                # --- Always free GPU memory after each sample ---
                sample = None  # noqa: F821 (may not be defined if exception before assignment)
                gt = None
                sens = None
                free_gpu_memory()

            # --- Periodic hard GPU flush every N samples ---
            if (i + 1) % args.gpu_clear_every == 0:
                free_gpu_memory()
                if torch.cuda.is_available():
                    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
                    tqdm.write(f"[GPU] After sample {i+1}: {free_mb:.0f} MB free")

    t_end = time.perf_counter()

    print("\n" + "=" * 50)
    print("Done")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Elapsed   : {t_end - t_start:.2f} s")
    print(f"Map file  : {mapping_path}")

    if failed_indices:
        fail_log = os.path.join(out_dir, "failed_samples.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write("index,path,error\n")
            for idx, fpath, err in failed_indices:
                f.write(f"{idx},{fpath},{err}\n")
        print(f"Failed log: {fail_log}")


if __name__ == "__main__":
    main()