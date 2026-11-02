#!/usr/bin/env bash
# Large-d sequence of the κ0 = 0.015 ray with the joint (w0, q_perp) move (--gibbs-mode w0q), which mixes the aligned
# fraction at d >= 300 (the w0-only move of launch_large_d.sh does not). Theory point fixed: N = 1500,
# P = 3000 (d/150)², κ = 0.015 (d/150), sa0 = (8/1500)(150/d), s0 = 0.5, ε = -0.3√6.
# Two chains per d, run one after another: a prior start and an over-aligned start (aligned fraction ~0.065, above
# the equilibrium ~0.04), so agreement between them shows convergence from both sides. Groups: w0q_d<d>.
# One chain at a time on GPU $DEV under the thermal governor (gpu_governor.py), which pauses it at >= 76 °C.
#   DS="600 800" DEV=cuda:1 ./launch_large_d_w0q.sh        analysis: chains.py, plot_large_d_actions.py
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
DS=${DS:-"300 600 800 1200"}
DEV=${DEV:-cuda:1}
ITERS=${ITERS:-600}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs runs/inits
LOG=runs/logs/launch_large_d_w0q.log
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

overaligned_init() {   # theory-sampled start with extra neurons moved into the outer wells
  $PY - "$1" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np, torch
from ray_driver import theory_init
d = int(sys.argv[1]); r = d / 150
p = dict(N=1500, d=d, P=round(3000 * r * r), T=2 * 0.015 * r, sa0=(8 / 1500) / r)
base = Path(f"runs/inits/larged_d{d}_theory_init.pt")
if not base.exists():
    theory_init(p, base)
W = torch.load(base)["W0"].double().numpy().copy()
N = W.shape[0]; thr = 2.5 * np.sqrt(0.5 / d); q0 = 0.5 * (d - 1) / d
rng = np.random.default_rng(11 + d)
al = np.abs(W[:, 0]) > thr
need = int(0.065 * N) - al.sum()
idx = rng.choice(np.where(~al)[0], need, replace=False)
W[idx, 0] = rng.choice([-1, 1], need) * rng.normal(1.15, 0.05, need)
W[idx, 1:] *= np.sqrt(0.4 * q0 / (W[idx, 1:] ** 2).sum(1))[:, None]
torch.save({"W0": torch.tensor(W)}, f"runs/inits/w0q_d{d}_overaligned_init.pt")
PYEOF
}

run_chain() {   # d init [init-file] tag
  local d=$1 INIT=$2 FILE=$3 TAG=$4 G=w0q_d$1
  read -r P T SA0 <<< "$($PY -c "r=$d/150; print(round(3000*r*r), repr(2*0.015*r), repr((8/1500)/r))")"
  local extra=""; [ -n "$FILE" ] && extra="--init-file $FILE"
  say "start $G $TAG (P=$P)"
  $PY train_collapsed_hmc.py --d $d --P $P --N 1500 --T $T --s0 0.5 --sa0 $SA0 --eps -0.7348469228349533 --seed 0 \
    --dtype float32 --leapfrog 24 --step0 0.08 --target-acc 0.8 --iters $ITERS --burn 200 --obs-every 5 --save-every 4 \
    --chains 1 --gibbs-every 6 --gibbs-mode w0q --group $G --tag ${G}_$TAG --device $DEV \
    --init $INIT $extra --rng-seed 1 > runs/logs/${G}_${TAG}.log 2>&1 &
  local pid=$!
  $PY gpu_governor.py --pids $pid --hot 76 --cool 68 --poll 0.1 --log runs/logs/${G}_governor.log > /dev/null 2>&1 &
  wait $pid; wait
  if grep -q Traceback runs/logs/${G}_${TAG}.log; then say "FAILED $G $TAG"; else say "done $G $TAG"; fi
}

for d in $DS; do
  [ -f runs/inits/w0q_d${d}_overaligned_init.pt ] || overaligned_init $d
  run_chain $d prior "" prior
  run_chain $d file runs/inits/w0q_d${d}_overaligned_init.pt overaligned
done
say "finished"
