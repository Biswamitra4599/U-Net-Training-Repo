#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mri_sigpy_gpu"
PY_VER="3.11"

echo "[1/5] Creating conda env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip

echo "[2/5] Installing PyTorch CUDA 12.8 wheels"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

echo "[3/5] Installing CuPy and SigPy"
conda run -n "${ENV_NAME}" python -m pip install cupy-cuda12x sigpy

echo "[4/5] Installing MRI stack dependencies"
conda run -n "${ENV_NAME}" python -m pip install fastmri pygrappa torchmetrics h5py tqdm matplotlib

echo "[5/5] Running GPU diagnostics"
conda run -n "${ENV_NAME}" python /home/biswamitra/health/knee_data/EDA/Complex_NN_code/check_gpu_sigpy_support.py

echo "Done. Activate with: conda activate ${ENV_NAME}"
