#!/usr/bin/env bash
# FCN3 chains along the kappa0 = 0.015 ray (same theory point as fcn2_collapsed_hmc FINDINGS):
#   N1 = N2 = 1500, P = 3000 (d/150)^2, kappa = 0.015 (d/150), sigA^2 = (8/1500)(150/d), sigw^2 = 0.5, eps = -0.3 sqrt6,
#   chi = 1; sigB set so that sigB^2 J11(prior) = sigw^2 / d (hidden He1 prior variance = FCN2 read-in prior).
# One chain type per GPU (prior on cuda:0, aligned on cuda:1), d values run in sequence on each, after WAIT_PID exits.
#   WAIT0=<pid> WAIT1=<pid> DS="75 300" ./queue_ray_k0p015.sh
# N (default 1500) sets N1 = N2 = N with chi = N/1500: the theory point (r_B, g~ = sigA^2 chi/N2, N2/chi) is unchanged,
# so the theory prediction is the same and only the finite-N corrections of the sampler change.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/akiva/miniconda3/envs/ml_env/bin/python}
DS=${DS:-"75 300"}
ITERS=${ITERS:-1000}
BURN=${BURN:-300}
N=${N:-1500}
CHI=$($PY -c "print($N / 1500)")
declare -A SIGB=([75]=1.2574845533917596 [150]=1.25540121081682 [300]=1.254358125197598 [600]=1.2538362420680749)
mkdir -p runs
lane() {   # $1 init, $2 device, $3 smi index, $4 rng seed, $5 pid to wait for
  while [ -n "$5" ] && kill -0 "$5" 2>/dev/null; do sleep 30; done
  for d in $DS; do
    read -r P KAP SA <<< "$($PY -c "r=$d/150; print(round(3000*r*r), repr(0.015*r), repr(((8/1500)/r)**0.5))")"
    echo "$(date '+%m-%d %H:%M') start d=$d N=$N chi=$CHI $1 on $2 (P=$P kappa=$KAP sigA=$SA sigB=${SIGB[$d]})"
    $PY run_fcn3.py --d $d --P $P --N1 $N --N2 $N --chi $CHI --kappa $KAP --eps -0.7348469228349535 \
      --sigA $SA --sigB ${SIGB[$d]} --sigw 0.7071067811865476 --iters $ITERS --burn $BURN \
      --group ray_k0p015_d${d}_N$N --tag ray --init $1 --device $2 --smi-index $3 --rng-seed $4 \
      > runs/ray_d${d}_N${N}_$1.out 2>&1
    echo "$(date '+%m-%d %H:%M') done d=$d $1 (exit $?)"
  done
}
lane prior cuda:0 0 1 "${WAIT0:-}" &
lane aligned cuda:1 1 2 "${WAIT1:-}" &
wait
