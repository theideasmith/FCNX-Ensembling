#!/usr/bin/env bash
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
O=runs/lam/d150_s2
W="/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $O --iters 400 --burn 100"
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py seed --out $O --lams 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.15 0.1 0.05 0.0 --iters 40 --device cuda:1 > runs/lam/seed_out.out 2>&1
$W --side out --label out --start start_out_lLAM.pt --lams 1.0 0.8 0.6 0.4 0.2 0.1 0.0 --device cuda:0 --seed 2003 > runs/lam/out_a.out 2>&1 &
$W --side out --label out --start start_out_lLAM.pt --lams 0.9 0.7 0.5 0.3 0.15 0.05 --device cuda:1 --seed 2004 > runs/lam/out_b.out 2>&1 &
wait
echo "$(date +%H:%M) out side done"
