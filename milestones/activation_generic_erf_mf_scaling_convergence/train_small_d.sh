#!/bin/bash

# Train five networks in parallel with Ramanujan partition seeds: 1,2,3,5,7
# Parameters: d=50, P=600, N=800, chi=80, kappa=1.0, ens=10, device=cuda:0

SEEDS=(1 2 3 5 7)
EPOCHS=10000000
PIDS=()

# Function to kill all child processes
cleanup() {
    echo "Killing child processes..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null
    done
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

for SEED in "${SEEDS[@]}"; do
  python -u d_sweep.py \
    --P 600 \
    --d 50 \
    --N 800 \
    --chi 80 \
    --kappa 1.0 \
    --lr 3e-4 \
    --epochs $EPOCHS \
    --device cuda:0 \
    --seed $SEED \
    --ens 10 \
    > train_small_d_seed${SEED}.log 2>&1 &
  PIDS+=($!)
done

wait
echo "All training jobs finished."
