#!/usr/bin/env bash
# κ0=0.015 invariant ray (the paper's α/β scaling) at larger d with the fast sampler settings:
#   d = 150 b^{1/2}, N = 1500 b^{1/4}, P = 3000 b^{15/16}, κ = 0.015 b^{3/16}, sa0 = (8/1500) b^{-3/4}.
# Two chains per b (prior, theory-sampled start), b values sequential on one GPU.
#   BS="8 16" ./launch_ray_fast.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
BS=${BS:-"8 16"}
ITERS=${ITERS:-600}
for b in $BS; do
  read -r D N P T SA0 <<< "$($PY -c "
b=$b; al=b**1.25
print(round(150*b**.5), round(1500*b**.25), round(3000*al**.75), repr(2*0.015*(al/b)**.75), repr((8/1500)*b**-.75))")"
  GROUP=rayfast_b$b
  COMMON="--d $D --P $P --N $N --T $T --s0 0.5 --sa0 $SA0 --eps -0.7348469228349533 --seed 0 \
    --dtype float32 --leapfrog 24 --step0 0.08 --target-acc 0.8 --iters $ITERS --burn 200 --obs-every 5 --save-every 4 \
    --chains 1 --gibbs-every 6 --gibbs-mode w0 --gibbs-graph --group $GROUP --tag rayfast_b$b --device cuda:0"
  echo "$(date '+%m-%d %H:%M:%S') b=$b d=$D N=$N P=$P start"
  $PY train_collapsed_hmc.py $COMMON --init prior --rng-seed 1 > runs/logs/${GROUP}_prior.log 2>&1 &
  $PY train_collapsed_hmc.py $COMMON --init file --init-file runs/inits/rayfast_b${b}_theory_init.pt --rng-seed 1 > runs/logs/${GROUP}_file.log 2>&1 &
  wait
  echo "$(date '+%m-%d %H:%M:%S') b=$b done: $(grep -l Traceback runs/logs/${GROUP}_*.log | wc -l) failed"
done
echo "ray-fast finished"
