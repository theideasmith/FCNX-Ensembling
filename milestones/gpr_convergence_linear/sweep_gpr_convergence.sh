#!/bin/bash
# sweep_gpr_convergence.sh
# Sweep over widths: 32, 64, 256, 1024

P=100
d=4

echo "========================================"
echo "GPR Convergence Study: chi=1, d=$d, P=$P"
echo "========================================"

for N in 1028 256 64; do
    for d in 2 4; do
        for P in $((d+1)) $((d * 3)) 100; do
            echo ""
            echo "Training N=$N..."
            python train_gpr_convergence.py $P $N $d 1.0
            echo "Done: N=$N"
        done
    done
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"

