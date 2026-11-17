#!/usr/bin/env bash
# Onset of learnability in the FCN3 as P grows at FIXED kappa = 0.015 (the FCN3 version of the FCN2 theory plot
# ../fcn2_collapsed_hmc/figures/theory_action_vs_P_d150_k0.015.png). Everything else is the kappa0 = 0.015 ray point
# at d = 150: N1 = N2 = 1500, chi = 1, sigA^2 = 8/1500, sigB = 1.2554, sigw^2 = 0.5, eps = -0.3 sqrt(6).
# Prior start only, 1000 iterations (300 burn), read-in snapshots every 5, full network every 50 after burn-in (nets/). summary.json carries the theory at the same P.
# P = 3000 is the ray point (also run as groups ray_k0p015_d150, ray_hist_d150, without full networks); rerun here with them.
#
# Queue: dataset 0 at every P first, then datasets 1 and 2; within each, low and high P alternate. One worker per GPU
# pulls the next job, so both GPUs run at once. Re-running the script resumes the jobs still in the queue.
#   tmux new -d -s fcn3_onset ./queue_onset.sh
#   tail -f runs/onset/queue.log
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/akiva/miniconda3/envs/ml_env/bin/python
D=runs/onset; Q=$D/queue.txt; LOCK=$D/queue.lock; LOG=$D/queue.log
mkdir -p $D
if [ ! -s $Q ]; then
  for s in 0 1 2; do for P in 750 48000 1500 24000 3000 12000 6000; do echo "$P $s"; done; done > $Q
fi
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
pop() { ( flock 9; head -n 1 $Q; sed -i '1d' $Q ) 9>$LOCK; }
worker() {  # $1 GPU index
  while true; do
    read -r P S <<< "$(pop)"
    [ -z "${P:-}" ] && break
    log "start P=$P dataset=$S on cuda:$1"
    $PY run_fcn3.py --d 150 --P $P --N1 1500 --N2 1500 --chi 1 --kappa 0.015 --eps -0.7348469228349535 \
      --sigA 0.07302967433402215 --sigB 1.25540121081682 --sigw 0.7071067811865476 \
      --iters 1000 --burn 300 --count-every 0 --snap-every 5 --net-every 50 --seed $S --rng-seed $((10 + S)) \
      --group onset_d150 --tag onset --init prior --device cuda:$1 --smi-index $1 > $D/P${P}_s${S}.out 2>&1
    log "done  P=$P dataset=$S on cuda:$1 (exit $?)"
  done
  log "worker cuda:$1 finished"
}
worker 0 & sleep 5; worker 1 & wait
log "all done"
