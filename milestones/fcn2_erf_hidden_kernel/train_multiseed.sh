#!/bin/bash
# Train FCN2 erf network on multiple dataset seeds (up to 10)
# Usage: ./train_multiseed.sh [num_seeds] [device]
# Example: ./train_multiseed.sh 10 cuda:0

NUM_SEEDS=${1:-10}
DEVICE=${2:-cuda:1}

# Hyperparameters
D=50
P=200
N=200
CHI=200.0
LR=5e-7
TEMPERATURE=2.0
EPOCHS=2python milestones/fcn2_erf_hidden_kernel/plot_multi_seed_lJ.py \
  --base-pattern milestones/fcn2_erf_hidden_kernel/d50_P200_N200_chi_200.0_lr_5e-07_T_2.0_seed_{seed} \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --output multi_seed_lJ.png \
  --json-out multi_seed_lJ.jsonpython milestones/fcn2_erf_hidden_kernel/plot_multi_seed_lJ.py \
  --base-pattern milestones/fcn2_erf_hidden_kernel/d50_P200_N200_chi_200.0_lr_5e-07_T_2.0_seed_{seed} \
  --seeds 1 2 3 4 5 6 7 8 9 \
  --output multi_seed_lJ.png \
  --json-out multi_seed_lJ.json000000

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Training FCN2 ERF Network - Multi-Seed"
echo "=========================================="
echo "Number of seeds: $NUM_SEEDS"
echo "Device: $DEVICE"
echo "Hyperparameters:"
echo "  D=$D, P=$P, N=$N"
echo "  chi=$CHI, lr=$LR, T=$TEMPERATURE"
echo "  epochs=$EPOCHS"
echo ""

# Create base directory for runs
BASE_DIR="$SCRIPT_DIR/multiseed_runs"
mkdir -p "$BASE_DIR"

# Loop over seeds
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "=========================================="
    echo "Starting training for seed $seed / $((NUM_SEEDS - 1))"
    echo "=========================================="
    
    RUN_DIR="$BASE_DIR/seed_$seed"
    mkdir -p "$RUN_DIR"
    
    # Run training in background to allow parallel execution if desired
    # To run sequentially, remove '&' at the end
    python "$SCRIPT_DIR/train_fcn2_erf.py" \
        --d "$D" \
        --P "$P" \
        --N "$N" \
        --epochs "$EPOCHS" \
        --log-interval 10000 \
        --lr "$LR" \
        --temperature "$TEMPERATURE" \
        --chi "$CHI" \
        --device "$DEVICE" \
        --dataset-seed "$seed" 
    
    echo "Seed $seed completed at $(date)"
    echo ""
done

echo "=========================================="
echo "All training runs completed!"
echo "Results saved to: $BASE_DIR"
echo "=========================================="
