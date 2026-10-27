#!/usr/bin/env bash
# Fixed-d P-scan at the κ0=0.015 ray point d=150: κ ∝ P keeps λ = κ/(P sa0) and Ĉ = s0 sa0 P²/(N d κ²) fixed,
# so the read-in theory is identical at every P while the finite-P leakage (∝ d^k/P) shrinks.
# Three chains per P (prior, small, theory-sampled start), points run one after another on one GPU.
#   PS="3000 6000 12000 24000" ./launch_pscan_d150.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
PS=${PS:-"3000 6000 12000 24000"}
ITERS=${ITERS:-800}
INIT=runs/inits/ray_k0p015_both_b1_theory_init.pt
for P in $PS; do
  T=$($PY -c "print(2 * 0.015 * $P / 3000)")
  GROUP=pscan_d150_P$P
  COMMON="--d 150 --P $P --N 1500 --T $T --s0 0.5 --sa0 0.005333333333333333 --eps -0.7348469228349533 --seed 0 \
    --dtype float32 --leapfrog 24 --step0 0.08 --target-acc 0.8 --iters $ITERS --burn 200 --obs-every 5 --save-every 4 \
    --chains 1 --gibbs-every 6 --gibbs-mode w0 --gibbs-graph --group $GROUP --tag pscan_P$P --device cuda:0"
  echo "$(date +%H:%M:%S) P=$P κ=$($PY -c "print($T/2)") start"
  $PY train_collapsed_hmc.py $COMMON --init prior --rng-seed 1 > runs/logs/${GROUP}_prior.log 2>&1 &
  $PY train_collapsed_hmc.py $COMMON --init small --rng-seed 1 > runs/logs/${GROUP}_small.log 2>&1 &
  $PY train_collapsed_hmc.py $COMMON --init file --init-file $INIT --rng-seed 1 > runs/logs/${GROUP}_file.log 2>&1 &
  wait
  echo "$(date +%H:%M:%S) P=$P done: $(grep -l Traceback runs/logs/${GROUP}_*.log | wc -l) failed"
done
echo "pscan finished"
