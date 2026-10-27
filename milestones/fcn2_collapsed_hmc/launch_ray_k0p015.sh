#!/usr/bin/env bash
# Invariant α/β ray at high learnability (theory L3 ≈ 0.40-0.41 all along), climbing in d:
#   d = 150 b^{1/2}, N = 1500 b^{1/4}, sa0 = (8/1500) b^{-3/4}, P = 3000 α^{3/4} with α = b^{5/4},
#   κ = 0.015 (α/b)^{3/4}, s0 = 0.5, ε = -0.3√6 (paper H3 coefficient -0.3 on normalized He3).
# λ = κ/(P sa0) and Ĉ = s0 sa0 P²/(N d κ²) are exactly constant; only β = d/(2 s0) grows.
#
#   B=1 DTYPE=float32 DEVICES="cuda:0 cuda:1" ./launch_ray_k0p015.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
B=${B:-1}
DTYPE=${DTYPE:-float32}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
SPECS=${SPECS:-"prior 1,small 1"}  # comma-separated "init seed" chains
EXTRA=${EXTRA:-}
read -r D N P T SA0 EPS <<< "$($PY -c "
b = $B; al = b ** 1.25
print(round(150 * b ** 0.5), round(1500 * b ** 0.25), round(3000 * al ** 0.75),
      repr(2 * 0.015 * (al / b) ** 0.75), repr((8 / 1500) * b ** -0.75), repr(-0.3 * 6 ** 0.5))")"
BTAG=$(echo "$B" | tr . p)
GROUP=${GROUP:-ray_k0p015_b${BTAG}_${DTYPE}}
SESSION=${SESSION:-chmc_${GROUP}}

COMMON="--d $D --P $P --N $N --T $T --s0 0.5 --sa0 $SA0 --eps $EPS --seed 0 \
  --dtype $DTYPE --leapfrog 50 --step0 0.03 --target-acc 0.8 --iters 400 --burn 100 \
  --obs-every 5 --save-every 2 --chains 1 --gibbs-every 3 --gibbs-mode w0 \
  --group $GROUP --tag ray_b${BTAG}"

read -r -a devs <<< "$DEVICES"
mkdir -p runs/logs
tmux new-session -d -s "$SESSION" -n main
IFS=, read -r -a specs <<< "$SPECS"
i=0
for spec in "${specs[@]}"; do
  set -- $spec
  dev=${devs[$((i % ${#devs[@]}))]}
  win="${1}_r${2}"
  log=runs/logs/${GROUP}_${win}.log
  cmd="$PY train_collapsed_hmc.py $COMMON --device $dev --init $1 --rng-seed $2 $EXTRA > $log 2>&1"
  if [ $i -gt 0 ]; then tmux new-window -t "$SESSION" -n "$win"; else tmux rename-window -t "$SESSION:main" "$win"; fi
  tmux send-keys -t "$SESSION:$win" "$cmd" C-m
  i=$((i + 1))
done
echo "group $GROUP: d=$D N=$N P=$P T=$T sa0=$SA0 eps=$EPS $DTYPE; $i chains in tmux $SESSION on [$DEVICES]"
