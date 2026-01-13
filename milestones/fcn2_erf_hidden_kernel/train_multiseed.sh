#!/bin/bash
# Train FCN2 erf network on multiple dataset seeds (up to 10)
# Usage: ./train_multiseed.sh [num_seeds] [device]
# Example: ./train_multiseed.sh 10 cuda:0

NUM_SEEDS=${1:-3}
DEVICE=${2:-cuda:1}

# Hyperparameters
D=150
P=600
N=700
CHI=$N
LR=3e-6
TEMPERATURE=2.0
EPOCHS=100000000

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
BASE_DIR="$SCRIPT_DIR/dataset_averaging_D${D}_P${P}_N${N}_chi${CHI}_T${TEMPERATURE}_lr${LR}_epochs${EPOCHS}"
mkdir -p "$BASE_DIR"

# Loop over seeds
for seed in $(seq  0 $((NUM_SEEDS - 1))); do
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
        --dataset-seed "$seed" & 
    
    echo "Seed $seed run at $(date)"
    echo ""
done


DEVICE="cuda:0"
N=1400
CHI=$N
BASE_DIR="$SCRIPT_DIR/dataset_averaging_D${D}_P${P}_N${N}_chi${CHI}_T${TEMPERATURE}_lr${LR}_epochs${EPOCHS}"
mkdir -p "$BASE_DIR"
for seed in $(seq  0 $((NUM_SEEDS - 1))); do
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
        --dataset-seed "$seed" & 
    
    echo "Seed $seed run at $(date)"
    echo ""
done

wait
echo "=========================================="
echo "All training runs completed!"
echo "Results saved to: $BASE_DIR"
echo "=========================================="
