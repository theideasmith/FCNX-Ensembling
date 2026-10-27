#!/usr/bin/env bash
# Multi-chain convergence check at the β=100 steepwell point (d=50, P=11998, N=3162).
# Four independent chains from very different starts; agreement in U, L3, w0² => converged.
# Chains are spread round-robin over $DEVICES in tmux session $SESSION. Compare with:
#   python runs.py compare --group $GROUP
#
#   GROUP=b100_gibbs EXTRA="--gibbs-every 3 --iters 150 --burn 60" ./launch_b100_multichain.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
DEVICES=${DEVICES:-"cuda:0"}
GROUP=${GROUP:-b100_multichain}
SESSION=${SESSION:-chmc_${GROUP}}
EXTRA=${EXTRA:-}
LANGEVIN=../fcn2_erf_hidden_kernel/red_robin_alpha_beta_invariant_steepwell_a01_g1.25_P0160_betamax100_nu0.25_N01000_sa00.03_ens1_Asnap_eps0.5_lr0.01_ep60M_sched2_3_5/models/invariant_beta100.000_alpha316.23_d50_P11998_N3162_sa00.0009_kappa0.0024_seed0

COMMON="--d 50 --P 11998 --N 3162 --T 0.004743 --sa0 0.0009 --eps 0.5 --seed 0 \
  --dtype float32 --leapfrog 50 --step0 0.03 --target-acc 0.8 --iters 200 --burn 50 \
  --obs-every 5 --save-every 10 --chains 1 \
  --group $GROUP --tag b100 --langevin-dir $LANGEVIN"

read -r -a devs <<< "$DEVICES"
mkdir -p runs/logs
tmux new-session -d -s "$SESSION" -n main
i=0
for spec in "prior 1" "prior 2" "small 1" "snapshot 1"; do
  set -- $spec
  dev=${devs[$((i % ${#devs[@]}))]}
  win="${1}_r${2}"
  log=runs/logs/${GROUP}_${win}.log
  cmd="$PY train_collapsed_hmc.py $COMMON --device $dev --init $1 --rng-seed $2 $EXTRA > $log 2>&1"
  if [ $i -gt 0 ]; then tmux new-window -t "$SESSION" -n "$win"; else tmux rename-window -t "$SESSION:main" "$win"; fi
  tmux send-keys -t "$SESSION:$win" "$cmd" C-m
  i=$((i + 1))
done
echo "launched $i chains in tmux session $SESSION on [$DEVICES]; logs in runs/logs/"
