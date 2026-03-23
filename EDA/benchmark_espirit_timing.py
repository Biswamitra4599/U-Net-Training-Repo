import argparse
import glob
import os
import time

import numpy as np
import sigpy as sp

import updated_dataloader as udl


def main():
    parser = argparse.ArgumentParser(description="Benchmark ESPIRiT sensitivity map generation time.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/biswamitra/health/knee_data/val/deconstructed_val",
        help="Directory containing .npy k-space files.",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to benchmark.")
    parser.add_argument("--calib-width", type=int, default=24)
    parser.add_argument("--thresh", type=float, default=0.02)
    parser.add_argument("--kernel-width", type=int, default=6)
    parser.add_argument("--crop", type=float, default=0.95)
    parser.add_argument(
        "--espirit-device",
        type=int,
        default=0,
        help="SigPy device for ESPIRiT (-1 for CPU, 0 for first GPU).",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    print(f"Found files: {len(paths)} in {args.data_dir}")
    if len(paths) == 0:
        raise SystemExit("No .npy files found.")

    n_samples = min(args.num_samples, len(paths))
    subset = paths[:n_samples]

    print(f"SigPy version       : {sp.__version__}")
    print(f"CuPy enabled in SigPy: {sp.config.cupy_enabled}")
    if sp.config.cupy_enabled:
        try:
            print(f"SigPy Device(0)      : {sp.Device(0)}")
        except Exception as exc:
            print(f"Could not init Device(0): {exc}")

    ds = udl.Custom_FMRI_DataLoader_nil(
        data_paths=subset,
        input_req=[0, 0, 0, 0, 0],
        output_req=[0, 0, 0, 0],
        methods_flags=[0, 1],  # only ESPIRiT
        espirit_params=[args.calib_width, args.thresh, args.kernel_width, args.crop, False],
        espirit_device=args.espirit_device,
    )

    # Warmup for fair timing.
    print("Running warmup...")
    t0 = time.perf_counter()
    _ = ds[0]["sensitivity_maps"]
    t1 = time.perf_counter()
    print(f"Warmup time: {t1 - t0:.3f} s")

    sample_times = []
    for i in range(n_samples):
        s0 = time.perf_counter()
        sm = ds[i]["sensitivity_maps"]
        s1 = time.perf_counter()
        dt = s1 - s0
        sample_times.append(dt)
        print(f"Sample {i}: {dt:.3f} s | shape={tuple(sm.shape)}")

    arr = np.asarray(sample_times, dtype=np.float64)
    print("\nSummary")
    print(f"mean   : {arr.mean():.3f} s/sample")
    print(f"median : {np.median(arr):.3f} s/sample")
    print(f"min    : {arr.min():.3f} s/sample")
    print(f"max    : {arr.max():.3f} s/sample")


if __name__ == "__main__":
    main()
