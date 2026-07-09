#!/bin/bash
# Train FCN2 erf network for the mini-grokking sweep.
# Usage: ./train_minigrokking.sh [device]
# Example: ./train_minigrokking.sh cuda:0

set -euo pipefail

# Hyperparameters
D=50
P=400
TEMPERATURE=0.2
LR=1e-5
ENSEMBLE_SIZE=10
EPOCHS=50000000
EPS=0.03

# Sweep over chi values with N = chi
CHI_VALUES=(60 120) # 180 240 600)
SEEDS=(0 1)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep launcher logs separate from training run directories
LOG_BASE_DIR="$SCRIPT_DIR/minigrokking_logs_D${D}_P${P}_T${TEMPERATURE}_lr${LR}_ens${ENSEMBLE_SIZE}"
mkdir -p "$LOG_BASE_DIR"

echo "=========================================="
echo "Training FCN2 ERF Network - MiniGrokking Sweep"
echo "=========================================="
echo "Hyperparameters:"
echo "  D=$D, P=$P"
echo "  T=$TEMPERATURE, lr=$LR, ens=$ENSEMBLE_SIZE"
echo "  epochs=$EPOCHS, eps=$EPS"
echo "  chi/N values: ${CHI_VALUES[*]}"
echo "  seeds per chi: ${SEEDS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Starting seed=$SEED"
    echo "=========================================="

    for CHI in "${CHI_VALUES[@]}"; do
        N=$CHI
        echo "Launching chi=$CHI (N=$N) for seed=$SEED"

        LOG_FILE="$LOG_BASE_DIR/d${D}_P${P}_N${N}_chi${CHI}_T${TEMPERATURE}_seed${SEED}.log"

        python "$SCRIPT_DIR/train_fcn2_erf.py" \
            --d "$D" \
            --P "$P" \
            --N "$N" \
            --epochs "$EPOCHS" \
            --log-interval 10000 \
            --lr "$LR" \
            --temperature "$TEMPERATURE" \
            --chi "$CHI" \
            --dataset-seed "$SEED" \
            --ens "$ENSEMBLE_SIZE" \
            --eps "$EPS" \
            > "$LOG_FILE" 2>&1 &

        echo "  log -> $LOG_FILE"
    done

done

wait

echo "=========================================="
echo "All mini-grokking training runs completed!"
echo "Launcher logs saved to: $LOG_BASE_DIR"
echo "Training outputs are written by train_fcn2_erf.py under the script directory."
echo "=========================================="