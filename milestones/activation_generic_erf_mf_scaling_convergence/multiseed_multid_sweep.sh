#!/bin/bash

# Define the list of d values
d_values=(30 50 100 150 200)

# Fixed hyperparameters
N=800
chi=80
kappa=0.1
ens=10
num_seeds=10
epochs=1200000
lr="1e-3"
device="cuda:0"
base_seed=42

echo "Starting sweep for d = ${d_values[*]}"
echo "--------------------------------------"

for d in "${d_values[@]}"
do
    # Calculate P = 10 * d
    P=$(( 10 * d ))
    
    echo "Running: d=$d, P=$P, kappa=$kappa, seeds=$num_seeds"
    
    # Execute the python script
    python d_sweep_seeds.py \
        --P "$P" \
        --d "$d" \
        --N "$N" \
        --chi "$chi" \
        --kappa "$kappa" \
        --lr "$lr" \
        --epochs "$epochs" \
        --device "$device" \
        --num_seeds "$num_seeds" \
        --base_seed "$base_seed" \
        --ens "$ens"

    echo "Finished d=$d. Moving to next model..."
    echo "--------------------------------------"
done

echo "All training runs complete."