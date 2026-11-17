#!/usr/bin/env bash
# d = 150, data seed 2, P = 3000. Two tagged neurons from the same state:
#   d150_s2        an ordinary neuron, 27 other outer neurons  -> P(28)/P(27)
#   d150_s2_member a team member,     26 other outer neurons  -> P(27)/P(26)
# Each: confined windows at 13 lam values on the ordinary side and on the team side.
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
A=runs/lam/d150_s2; B=runs/lam/d150_s2_member
W="/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py window --iters 400 --burn 100"
L1="0.0 0.1 0.2 0.4 0.6 0.8 1.0"; L2="0.05 0.15 0.3 0.5 0.7 0.9"
# natural starts
$W --out $A --side in  --label in  --lams $L1 --device cuda:0 --seed 4001 > runs/lam/A_in1.out 2>&1 &
$W --out $A --side in  --label in  --lams $L2 --device cuda:1 --seed 4002 > runs/lam/A_in2.out 2>&1 &
$W --out $B --side out --label out --lams $L1 --device cuda:0 --seed 4003 > runs/lam/B_out1.out 2>&1 &
$W --out $B --side out --label out --lams $L2 --device cuda:1 --seed 4004 > runs/lam/B_out2.out 2>&1 &
# constructed starts, then their windows
( /home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py seed --out $A --lams 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.15 0.1 0.05 0.0 --iters 40 --device cuda:0 --seed 4005 > runs/lam/A_seed.out 2>&1
  $W --out $A --side out --label out --start start_out_lLAM.pt --lams $L1 --device cuda:0 --seed 4006 > runs/lam/A_out1.out 2>&1 &
  $W --out $A --side out --label out --start start_out_lLAM.pt --lams $L2 --device cuda:1 --seed 4007 > runs/lam/A_out2.out 2>&1 &
  wait ) &
( /home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py seed_in --out $B --lams 0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --iters 40 --device cuda:1 --seed 4008 > runs/lam/B_seed.out 2>&1
  $W --out $B --side in --label in --start start_in_lLAM.pt --lams $L1 --device cuda:0 --seed 4009 > runs/lam/B_in1.out 2>&1 &
  $W --out $B --side in --label in --start start_in_lLAM.pt --lams $L2 --device cuda:1 --seed 4010 > runs/lam/B_in2.out 2>&1 &
  wait ) &
wait
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py analyze --out $A > runs/lam/analysis_A.out 2>&1
/home/akiva/miniconda3/envs/ml_env/bin/python lam_count.py analyze --out $B > runs/lam/analysis_B.out 2>&1
echo "$(date +%H:%M) all lam windows done"
