#!/usr/bin/env bash
# Train 2-layer erf network with d=2, P=6, N=20
# temperature=2.0, chi=N=20, lr=5e-6, epochs=100,000,000
# Uses CUDA:0 if available, otherwise CPU.

set -euo pipefail

DEVICE="cuda:1"
if ! python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
then
  DEVICE="cpu"
fi

python train_fcn2_erf.py \
  --d 150 \
  --P 600 \
  --N 700 \
  --epochs 100000000 \
  --temperature 0.1 \
  --chi 700 \
  --lr 5e-06 \
  --device "$DEVICE"
