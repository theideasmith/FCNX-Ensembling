#!/usr/bin/env bash
# =============================================================================
# Langevin MF FCN2-erf at d=50, 50M *wall* epochs.
#
# Schedule (--schedule-divisors 2,3,5): equal wall thirds, NO stretch.
#   wall = --epochs = 50M
#
#     phase   lr          wall span           wall end
#     -----   ---------   ------------------  --------
#       1     lr0 / 2     [0,  50M/3)         ~16.67M
#       2     lr0 / 3     [50M/3, 100M/3)     ~33.33M
#       3     lr0 / 5     [100M/3, 50M]       50M
#
# Overrides via env: P, N, CHI, T, EPS, LR, DEVICE, EPOCHS, SEED
# Default hyps match the d=50 κ=0.1 (T=0.2) 50k P-sweep: N=χ=400.
# =============================================================================
set -euo pipefail

ROOT="/home/akiva/FCNX-Ensembling"
PY="${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}"
TRAIN="$ROOT/milestones/fcn2_erf_hidden_kernel/train_fcn2_erf.py"

D="${D:-50}"
P="${P:-2000}"
N="${N:-400}"
CHI="${CHI:-$N}"
LR="${LR:-0.01}"
T="${T:-0.2}"
EPS="${EPS:-0.5}"
S0="${S0:-1.0}"
SEED="${SEED:-42}"
ENS="${ENS:-1}"
EPOCHS="${EPOCHS:-50000000}"
LOG_INTERVAL="${LOG_INTERVAL:-500000}"
DEVICE="${DEVICE:-cuda:0}"
DIVS="${DIVS:-2,3,5}"

NAME="d${D}_P${P}_N${N}_chi_${CHI}_lr_${LR}_T_${T}_seed_${SEED}_eps_${EPS}_schedule_2_3_5"
OUT="$ROOT/milestones/fcn2_erf_hidden_kernel/red_robin_d${D}_T${T}_P${P}_N${N}_chi${CHI}_eps${EPS}_lr${LR}_ep50M_schedule_2_3_5"
MODEL_DIR="$OUT/models/$NAME"
TB_DIR="$OUT/tensorboard/$NAME"
LOG_DIR="$OUT/launcher_logs"

mkdir -p "$MODEL_DIR" "$TB_DIR" "$LOG_DIR"

echo "============================================================"
echo "d=$D  P=$P  N=$N  chi=$CHI  lr0=$LR  T=$T  eps=$EPS  (kappa≈T/2=$(python3 -c "print($T/2)"))"
echo "wall epochs=$EPOCHS  (no stretch)"
echo "schedule-divisors=$DIVS → equal thirds: lr0/2, lr0/3, lr0/5"
echo "  wall ends: ~16.67M / ~33.33M / 50M"
echo "device=$DEVICE"
echo "output: $OUT"
echo "============================================================"

cd "$ROOT/milestones/fcn2_erf_hidden_kernel"
exec "$PY" -u "$TRAIN" \
  --d "$D" --P "$P" --N "$N" --chi "$CHI" \
  --eps "$EPS" --ens "$ENS" \
  --lr "$LR" --temperature "$T" --s0 "$S0" \
  --dataset-seed "$SEED" \
  --epochs "$EPOCHS" --log-interval "$LOG_INTERVAL" \
  --schedule-divisors "$DIVS" \
  --device "$DEVICE" --classic \
  --output-dir "$MODEL_DIR" \
  --tensorboard-dir "$TB_DIR" \
  2>&1 | tee "$LOG_DIR/${NAME}.log"
