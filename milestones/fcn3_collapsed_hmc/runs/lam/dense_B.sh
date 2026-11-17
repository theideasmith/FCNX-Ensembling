#!/usr/bin/env bash
# run B (tagged member, 26 others), team side, dense lam near 1, started as member (m) and dissolved (d)
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
D=runs/lam/d150_s2_member/dense
W="/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $D --side out --iters 400 --burn 100 --grid 0.9 0.92 0.94 0.96 0.97 0.98 0.99 1.0"
$W --label m --start start_mem.pt  --lams 1.0 0.98 0.96 0.92 --device cuda:0 --seed 6001 > runs/lam/dB_m1.out 2>&1 &
$W --label m --start start_mem.pt  --lams 0.99 0.97 0.94 0.9 --device cuda:1 --seed 6002 > runs/lam/dB_m2.out 2>&1 &
$W --label d --start start_diss.pt --lams 0.9 0.94 0.97 0.99 --device cuda:0 --seed 6003 > runs/lam/dB_d1.out 2>&1 &
$W --label d --start start_diss.pt --lams 0.92 0.96 0.98 1.0 --device cuda:1 --seed 6004 > runs/lam/dB_d2.out 2>&1 &
wait
echo "$(date +%H:%M) dense B done"
