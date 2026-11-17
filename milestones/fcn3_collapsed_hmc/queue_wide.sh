#!/usr/bin/env bash
# Coupling between the read-in and hidden layers as the width grows, at the onset point P = 750, kappa = 0.015, d = 150
# (theory L3 0.29, N = 1500 sampler L3 0.20-0.22; onset runs r10-r12 are the N = 1500 baseline on the same datasets).
# N1 = N2 = 6000 = 40 d with chi = 4, and 3000 with chi = 2: the theory is the same at every N. Everything else as
# queue_onset.sh. Prior start, 1000 iterations (300 burn), read-in snapshots every 5, full network every 50 after burn-in.
#
# One worker on cuda:0 (cuda:1 belongs to another user). N = 6000 dataset 0 first, then N = 3000 dataset 0, then the rest.
#   tmux new -d -s fcn3_wide ./queue_wide.sh
#   tail -f runs/wide/queue.log
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/akiva/miniconda3/envs/ml_env/bin/python
D=runs/wide; Q=$D/queue.txt; LOCK=$D/queue.lock; LOG=$D/queue.log
mkdir -p $D
if [ ! -s $Q ]; then
  printf "6000 0\n3000 0\n6000 1\n6000 2\n3000 1\n3000 2\n" > $Q
fi
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
pop() { ( flock 9; head -n 1 $Q; sed -i '1d' $Q ) 9>$LOCK; }
GPU=0
while true; do
  read -r N S <<< "$(pop)"
  [ -z "${N:-}" ] && break
  CHI=$(( N / 1500 ))
  log "start N=$N chi=$CHI dataset=$S on cuda:$GPU"
  $PY run_fcn3.py --d 150 --P 750 --N1 $N --N2 $N --chi $CHI --kappa 0.015 --eps -0.7348469228349535 \
    --sigA 0.07302967433402215 --sigB 1.25540121081682 --sigw 0.7071067811865476 \
    --iters 1000 --burn 300 --count-every 0 --snap-every 5 --net-every 50 --seed $S --rng-seed $((10 + S)) \
    --group wide_d150 --tag wide --init prior --device cuda:$GPU --smi-index $GPU > $D/N${N}_s${S}.out 2>&1
  log "done  N=$N dataset=$S (exit $?)"
done
log "all done"
