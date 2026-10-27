#!/usr/bin/env bash
# Large-d limit at the fixed theory point of the κ0=0.015 ray (λ = 9.375e-4, Ĉ = 474.07; only β = d/2σ_w² grows):
#   N = 1500 fixed, P = 3000 (d/150)², κ = 0.015 (d/150), sa0 = (8/1500)(150/d), s0 = 0.5, ε = -0.3√6.
# P ∝ d² makes the dominant training-set leakage (∝ d/P) fall as 1/d. d=150 is the existing pscan_d150_P3000 group.
# Two chains per d (prior, theory-sampled start), d values one after another on one GPU.
#   DS="300 600 1200" ./launch_large_d.sh        analysis: python analyze_large_d.py
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
DS=${DS:-"300 600 1200"}
ITERS=${ITERS:-600}
for D in $DS; do
  read -r P T SA0 <<< "$($PY -c "d=$D; print(3000*(d//150)**2, repr(2*0.015*d/150), repr((8/1500)*150/d))")"
  GROUP=larged_d$D
  COMMON="--d $D --P $P --N 1500 --T $T --s0 0.5 --sa0 $SA0 --eps -0.7348469228349533 --seed 0 \
    --dtype float32 --leapfrog 24 --step0 0.08 --target-acc 0.8 --iters $ITERS --burn 200 --obs-every 5 --save-every 4 \
    --chains 1 --gibbs-every 6 --gibbs-mode w0 --gibbs-graph --group $GROUP --tag larged_d$D --device cuda:0"
  echo "$(date '+%m-%d %H:%M:%S') d=$D P=$P κ=$($PY -c "print($T/2)") start"
  $PY train_collapsed_hmc.py $COMMON --init prior --rng-seed 1 > runs/logs/${GROUP}_prior.log 2>&1 &
  $PY train_collapsed_hmc.py $COMMON --init file --init-file runs/inits/larged_d${D}_theory_init.pt --rng-seed 1 > runs/logs/${GROUP}_file.log 2>&1 &
  wait
  echo "$(date '+%m-%d %H:%M:%S') d=$D done: $(grep -l Traceback runs/logs/${GROUP}_*.log | wc -l) failed"
done
echo "large-d finished"
