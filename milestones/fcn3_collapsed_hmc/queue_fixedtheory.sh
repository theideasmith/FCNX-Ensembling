#!/usr/bin/env bash
# Is the large-P disagreement between the FCN3 theory and the sampler on the onset line (queue_onset.sh: d = 150,
# kappa = 0.015) a finite-training-set effect?  Hold the theory fixed at an onset point and grow P with kappa ∝ P
# (kappa/P, and so every parameter of the theory, unchanged; the training set relative to the population improves).
#   point B = onset P = 24000:  P = 48000 (kappa 0.03), 96000 (kappa 0.06)
#   point A = onset P = 12000:  P = 24000 (kappa 0.03), 48000 (kappa 0.06)
# Dataset 0, prior start, 1000 iterations (300 burn), read-in snapshots every 5, full networks every 50 after burn-in.
# GPU 0 only (GPU 1 is in use by another user). Re-running resumes the jobs left in runs/ptheory/queue.txt.
#   tmux new -d -s fcn3_ptheory ./queue_fixedtheory.sh ; tail -f runs/ptheory/queue.log
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/akiva/miniconda3/envs/ml_env/bin/python
D=runs/ptheory; Q=$D/queue.txt; LOG=$D/queue.log
mkdir -p $D
if [ ! -s $Q ]; then
  printf "48000 0.03 B\n24000 0.03 A\n96000 0.06 B\n48000 0.06 A\n" > $Q
fi
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
while read -r P K PT <<< "$(head -n 1 $Q)"; [ -n "${P:-}" ]; do
  sed -i '1d' $Q
  log "start P=$P kappa=$K (theory point $PT) on cuda:0"
  $PY run_fcn3.py --d 150 --P $P --N1 1500 --N2 1500 --chi 1 --kappa $K --eps -0.7348469228349535 \
    --sigA 0.07302967433402215 --sigB 1.25540121081682 --sigw 0.7071067811865476 \
    --iters 1000 --burn 300 --count-every 0 --snap-every 5 --net-every 50 --seed 0 --rng-seed 10 \
    --group ptheory_${PT} --tag ptheory --init prior --device cuda:0 --smi-index 0 > $D/P${P}_k${K}.out 2>&1
  log "done  P=$P kappa=$K (exit $?)"
  P=""
done
log "all done"
