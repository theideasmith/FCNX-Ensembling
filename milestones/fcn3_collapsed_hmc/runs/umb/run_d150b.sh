#!/usr/bin/env bash
# d = 150 umbrella: pull (cmax 0.36), production in 3 processes per profile, then WHAM analysis.
set -uo pipefail
cd /home/akiva/FCNX-Ensembling/milestones/fcn3_collapsed_hmc
PY=/home/akiva/miniconda3/envs/ml_env/bin/python
SRC=runs/20261109-024057_countval_d150_P3000_N1500x1500_prior_r31
prof() {   # $1 tag, $2 device
  O=runs/umb/d150b_$1
  rm -rf $O
  $PY umbrella_count.py pull --from $SRC --tag $1 --out $O --cmax 0.36 --dc 0.015 --k 15000 --device $2 > runs/umb/d150b_$1_pull.out 2>&1
  for p in 0 1 2 3; do
    $PY umbrella_count.py prod --out $O --windows $(seq $p 4 24) --iters 1000 --burn 300 --device $2 \
      > runs/umb/d150b_$1_prod$p.out 2>&1 &
  done
  wait
  $PY umbrella_count.py analyze --out $O | tee runs/umb/d150b_$1_analysis.out
}
prof outer cuda:0 &
prof inner cuda:1 &
wait
echo "$(date '+%H:%M') all done"
