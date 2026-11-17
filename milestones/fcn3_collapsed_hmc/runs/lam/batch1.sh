#!/usr/bin/env bash
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
O=runs/lam/d150_s2
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $O --lams 0.0 0.2 0.5 0.8 --device cuda:0 --seed 1001 > runs/lam/b1_p0.out 2>&1 &
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $O --lams 0.05 0.3 0.6 0.9 --device cuda:0 --seed 1002 > runs/lam/b1_p1.out 2>&1 &
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $O --lams 0.1 0.4 0.7 1.0 --device cuda:1 --seed 1003 > runs/lam/b1_p2.out 2>&1 &
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --out $O --lams 0.15 --device cuda:1 --seed 1004 > runs/lam/b1_p3.out 2>&1 &
wait
echo "$(date +%H:%M) batch1 done"
