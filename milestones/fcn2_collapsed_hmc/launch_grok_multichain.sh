#!/usr/bin/env bash
# 4-chain collapsed HMC + Gibbs at the grokking-paper teacher-student point
# (d=150, P=3000, N=700, σ_w²=0.5, σ_a²=8/N, y = He1 - 0.3 H3 with unnormalized H3, i.e. eps = -0.3√6).
# KAPPA is the noise variance σ² (T = 2κ). EPS overrides the He3 coefficient (normalized He3, as in
# readin_action_theory.py); EPS=-0.3 is the RubinRegime point. S0 is σ_w²; SPECS lists "init seed" chains.
# D, NP (=P) and SA0 override the base point (α/β ray).
# N with CHI=N/700 (sa0 fixed at 8/700) holds the mean-field theory fixed: a finite-N series. Compare with:
#   python runs.py rhat --group $GROUP
#
#   KAPPA=0.1 DEVICES="cuda:0 cuda:1" ./launch_grok_multichain.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
DEVICES=${DEVICES:-"cuda:0"}
KAPPA=${KAPPA:-0.1}
KTAG=$(echo "$KAPPA" | tr . p)
GROUP=${GROUP:-grok_k${KTAG}}
SESSION=${SESSION:-chmc_${GROUP}}
EXTRA=${EXTRA:-}
SPECS=${SPECS:-"prior 1,prior 2,small 1,small 2"}  # comma-separated "init seed" chains
T=$(python -c "print(2*$KAPPA)")
EPS=${EPS:-$(python -c "print(-0.3*6**0.5)")}
S0=${S0:-0.5}
N=${N:-700}
CHI=${CHI:-1.0}
SPECS=${SPECS:-"prior 1,prior 2,small 1,small 2"}
ITERS=${ITERS:-400}
BURN=${BURN:-100}
SA0=${SA0:-$(python -c "print(8/700)")}
D=${D:-150}
NP=${NP:-3000}

COMMON="--d $D --P $NP --N $N --chi $CHI --T $T --s0 $S0 --sa0 $SA0 --eps $EPS --seed 0 \
  --dtype float64 --leapfrog 50 --step0 0.03 --target-acc 0.8 --iters $ITERS --burn $BURN \
  --obs-every 5 --save-every 2 --chains 1 --gibbs-every 3 --gibbs-mode w0 \
  --group $GROUP --tag grok_k${KTAG}"

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
echo "launched $i chains in tmux session $SESSION on [$DEVICES]; logs in runs/logs/"
