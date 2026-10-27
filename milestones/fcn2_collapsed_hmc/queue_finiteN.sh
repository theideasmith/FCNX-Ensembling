#!/usr/bin/env bash
# Finite-N series at the RubinRegime point (κ=0.07, σ_w²=0.5, ε=-0.3), theory held fixed:
# sa0 = 8/700, χ = N/700. Waits for the κ sweep to free the GPUs, then runs N=1400 and N=2800 in turn.
set -euo pipefail
cd "$(dirname "$0")"
until [ "$(ls runs/*rubin_em0p3_k0p*/summary.json 2>/dev/null | wc -l)" -ge 10 ]; do sleep 30; done
for n in 1400 2800; do
  chi=$(python -c "print($n/700)")
  G=rubin_k0p07_em0p3_N${n}
  KAPPA=0.07 EPS=-0.3 S0=0.5 N=$n CHI=$chi SPECS="prior 1,small 1" ITERS=100 BURN=30 GROUP=$G \
    DEVICES="cuda:0 cuda:1" EXTRA="--tag $G" ./launch_grok_multichain.sh
  until [ "$(ls runs/*_${G}_*/summary.json 2>/dev/null | wc -l)" -ge 2 ]; do
    grep -q Traceback runs/logs/${G}_*.log 2>/dev/null && { echo "$G crashed"; exit 1; }
    sleep 30
  done
  echo "$G done"
done
