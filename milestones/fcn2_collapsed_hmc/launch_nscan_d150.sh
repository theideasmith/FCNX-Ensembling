#!/usr/bin/env bash
# N-scan at the κ0=0.015 ray point d=150, P=3000 with χ = N/1500 (sa0 fixed): λ and Ĉ, hence the read-in theory,
# are identical at every N; only finite-N effects change. Two chains per N (prior, theory-sampled start).
#   NS="750 3000" ./launch_nscan_d150.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
NS=${NS:-"750 3000"}
ITERS=${ITERS:-800}
for N in $NS; do
  CHI=$($PY -c "print($N / 1500)")
  GROUP=nscan_d150_N$N
  COMMON="--d 150 --P 3000 --N $N --chi $CHI --T 0.03 --s0 0.5 --sa0 0.005333333333333333 --eps -0.7348469228349533 --seed 0 \
    --dtype float32 --leapfrog 24 --step0 0.08 --target-acc 0.8 --iters $ITERS --burn 200 --obs-every 5 --save-every 4 \
    --chains 1 --gibbs-every 6 --gibbs-mode w0 --gibbs-graph --group $GROUP --tag nscan_N$N --device cuda:0"
  echo "$(date +%H:%M:%S) N=$N χ=$CHI start"
  $PY train_collapsed_hmc.py $COMMON --init prior --rng-seed 1 > runs/logs/${GROUP}_prior.log 2>&1 &
  $PY train_collapsed_hmc.py $COMMON --init file --init-file runs/inits/nscan_d150_N${N}_theory_init.pt --rng-seed 1 > runs/logs/${GROUP}_file.log 2>&1 &
  wait
  echo "$(date +%H:%M:%S) N=$N done: $(grep -l Traceback runs/logs/${GROUP}_*.log | wc -l) failed"
done
echo "nscan finished"
