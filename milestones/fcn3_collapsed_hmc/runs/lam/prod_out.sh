#!/usr/bin/env bash
# team side of both runs, soft wall (lam_count.K_WALL)
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
A=runs/lam/d150_s2; B=runs/lam/d150_s2_member
W="/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --iters 400 --burn 100"
L1="1.0 0.8 0.6 0.4 0.2 0.1 0.0"; L2="0.9 0.7 0.5 0.3 0.15 0.05"
$W --out $B --side out --label out --lams $L1 --device cuda:0 --seed 5003 > runs/lam/B_out1.out 2>&1 &
$W --out $B --side out --label out --lams $L2 --device cuda:1 --seed 5004 > runs/lam/B_out2.out 2>&1 &
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py seed --out $A --lams 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.15 0.1 0.05 0.0 --iters 40 --device cuda:0 --seed 5005 > runs/lam/A_seed.out 2>&1
$W --out $A --side out --label out --start start_out_lLAM.pt --lams $L1 --device cuda:0 --seed 5006 > runs/lam/A_out1.out 2>&1 &
$W --out $A --side out --label out --start start_out_lLAM.pt --lams $L2 --device cuda:1 --seed 5007 > runs/lam/A_out2.out 2>&1 &
wait
echo "$(date +%H:%M) team sides done"
