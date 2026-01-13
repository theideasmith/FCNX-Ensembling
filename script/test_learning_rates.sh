#!/bin/bash

# Script to train FCN3 linear networks with varying dataset dimensions
# Sweeps d from 10 to 60 with step 10
# N = 4*d, P = 6*d, ens = 3
# Trains on 10 different datasets using --data_averaged_num for reproducibility
# Results stored in experiment_dirname: d_sweep_linear_network

EXPERIMENT_DIR="d_sweep_linear_network"
KAPPA=1.0
ENS=1
NUM_DATASETS=1
LR_AS=(1.5e-6 1.5e-3)
EPOCHS=25000000
# Array of dataset dimensions to sweep
D_VALUES=(2)

# Loop over each dataset dimension
for dataset_idx in $(seq 0 $((NUM_DATASETS - 1))); do
    for LR_A in "${LR_AS[@]}"; do
        for d in "${D_VALUES[@]}"; do
            # Calculate N and P based on d
            N=$((4 * d))
            P=$((6 * d))
            
            echo "=========================================="
            echo "Training with d=$d, N=$N, P=$P"
            echo "=========================================="
        
        # Train on NUM_DATASETS different datasets
            echo "Training on dataset $((dataset_idx + 1))/$NUM_DATASETS..."
            
            echo "Executing command: python $COMMAND"
            python /home/akiva/FCNX-Ensembling/script/ensembling_fcn3_linear.py \
                --d "$d" \
                --N "$N" \
                --P "$P" \
                --chi "$((N / 2))" \
                # --headless \
                --ens "$ENS" \
                --kappa "$KAPPA" \
                --experiment_dirname "$EXPERIMENT_DIR" \
                --data_averaged_num "$dataset_idx" \
                --lrA "$LR_A" \
                --epochs "$EPOCHS" 
            if [ $? -ne 0 ]; then
                echo "Error training with d=$d, dataset_idx=$dataset_idx"
                exit 1
            fi
        done
    done
done

echo "=========================================="
echo "All training runs completed!"
echo "Results stored in: /home/akiva/exp/$EXPERIMENT_DIR"
echo "=========================================="
