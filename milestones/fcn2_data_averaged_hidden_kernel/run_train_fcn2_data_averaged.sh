#!/usr/bin/env bash
set -e

# Train 3 ensembles over 20 datasets for 30,000,000 epochs
# Hyperparameters: d=50, P=200, N=200, lr=5e-6, chi=N, temperature=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/train_fcn2_data_averaged.py"

D=50
P=200
N=200
DATASETS=20
ENSEMBLES=3
EPOCHS=30000000
LR=5e-6
TEMP=1.0
CHI=$N
DEVICE="cuda:0"

python "$PYTHON_SCRIPT" \
  --d "$D" --P "$P" --N "$N" \
  --datasets "$DATASETS" --ensembles "$ENSEMBLES" \
  --epochs "$EPOCHS" --lr "$LR" --temperature "$TEMP" \
  --chi "$CHI" --device "$DEVICE"

# After training, plot outputs
RUN_DIR="$SCRIPT_DIR/dataavg_d${D}_P${P}_N${N}_D${DATASETS}_Q${ENSEMBLES}_chi_${CHI}_lr_${LR}_T_${TEMP}"
python "$SCRIPT_DIR/plot_train_fcn2_data_averaged_outputs.py" --run-dir "$RUN_DIR" --d "$D" --P "$P" --N "$N" --chi "$CHI"
