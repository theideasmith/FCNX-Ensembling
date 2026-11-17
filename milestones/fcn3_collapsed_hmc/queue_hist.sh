#!/usr/bin/env bash
# Read-in histogram chains on the kappa0 = 0.015 ray, N1 = N2 = 1500, on new datasets (data seeds 1-3), prior start,
# with read-in snapshots every 5 iterations (snaps.npz). Count moves off (as in the original ray chains).
#   phase 1 (now):            d = 150 x3 then d = 300 seed 3 on cuda:0;  d = 300 seeds 1, 2 on cuda:1
#   phase 2 (after N = 3000): d = 600 seeds 1, 3 on cuda:0;  d = 600 seed 2 on cuda:1
#   phase 3:                  umbrella production at d = 150 (runs/umb/d150b_*), 2 processes per GPU
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/akiva/miniconda3/envs/ml_env/bin/python
WAIT0=${WAIT0:-}; WAIT1=${WAIT1:-}
declare -A SIGB=([150]=1.25540121081682 [300]=1.254358125197598 [600]=1.2538362420680749)
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a runs/queue_hist.log; }
chain() {  # $1 d, $2 data seed, $3 device, $4 smi index
  read -r P KAP SA <<< "$($PY -c "r=$1/150; print(round(3000*r*r), repr(0.015*r), repr(((8/1500)/r)**0.5))")"
  log "start d=$1 data seed=$2 on $3"
  $PY run_fcn3.py --d $1 --P $P --N1 1500 --N2 1500 --chi 1 --kappa $KAP --eps -0.7348469228349535 \
    --sigA $SA --sigB ${SIGB[$1]} --sigw 0.7071067811865476 --iters 1000 --burn 300 --count-every 0 \
    --snap-every 5 --seed $2 --group ray_hist_d$1 --tag ray --init prior --device $3 --smi-index $4 \
    --rng-seed $((10 + $2)) > runs/hist_d$1_s$2.out 2>&1
  log "done d=$1 data seed=$2 (exit $?)"
}
waitpid() { while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done; }
umb() {   # $1 tag, $2 device, $3 lane
  $PY umbrella_count.py prod --out runs/umb/d150b_$1 --windows $(seq $3 2 24) --iters 1000 --burn 300 --device $2 \
    > runs/umb/d150b_$1_prod$3.out 2>&1
}
lane0() {
  chain 150 1 cuda:0 0; chain 150 2 cuda:0 0; chain 150 3 cuda:0 0; chain 300 3 cuda:0 0
  waitpid "$WAIT0"; chain 600 1 cuda:0 0; chain 600 3 cuda:0 0
  log "umbrella outer start"; umb outer cuda:0 0 & umb outer cuda:0 1 & wait
  $PY umbrella_count.py analyze --out runs/umb/d150b_outer > runs/umb/d150b_outer_analysis.out 2>&1; log "umbrella outer done"
}
lane1() {
  chain 300 1 cuda:1 1; chain 300 2 cuda:1 1
  waitpid "$WAIT1"; chain 600 2 cuda:1 1
  log "umbrella inner start"; umb inner cuda:1 0 & umb inner cuda:1 1 & wait
  $PY umbrella_count.py analyze --out runs/umb/d150b_inner > runs/umb/d150b_inner_analysis.out 2>&1; log "umbrella inner done"
}
lane0 & lane1 & wait
log "all done"
