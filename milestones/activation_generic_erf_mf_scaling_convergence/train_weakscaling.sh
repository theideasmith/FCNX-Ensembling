#!/bin/bash

# Train five networks in parallel with Ramanujan partition seeds: 1,2,3,5,7
# Parameters: d=150, lr=5e-6, P=1200, kappa=2.0, chi=40, N=800, ens=10, device=cuda:0

SEEDS=(1 5 7)
EPOCHS=50000000
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
    --P 1200 \
    --d 100 \
    --N 1000 \
    --chi 10 \
    --kappa 2.0 \
    --lr 1e-4 \
    --epochs $EPOCHS \
    --device cuda:0 \
    --seed $SEED \
    --ens 10 \
    --eps 0.03 \
    > train_seed${SEED}.log 2>&1 &
  PIDS+=($!)
done

wait
echo "All training jobs finished."
