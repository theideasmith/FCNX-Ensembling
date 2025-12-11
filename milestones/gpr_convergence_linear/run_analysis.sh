#!/usr/bin/env bash
# run_analysis.sh
# Loop over all trained runs in gpr_convergence_linear and generate plots
# using analyze_gpr_results.py, then copy them into a central plots/ folder.
#
# Usage:
#   bash run_analysis.sh
#
# Requirements:
#   - train_gpr_convergence.py outputs run directories named:
#       P{P}_N{N}_d{d}_k{k}_chi{chi}
#     containing config.txt and model.pt
#   - analyze_gpr_results.py is in the same folder as this script.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOTS_DIR="$ROOT_DIR/plots"
ANALYZER="$ROOT_DIR/analyze_gpr_results.py"

mkdir -p "$PLOTS_DIR"

if [[ ! -x $(command -v python3) ]]; then
  echo "python3 not found" >&2
  exit 1
fi

if [[ ! -f "$ANALYZER" ]]; then
  echo "Analyzer script not found: $ANALYZER" >&2
  exit 1
fi

echo "Starting analysis..."
shopt -s nullglob
for run_dir in "$ROOT_DIR"/P*_N*_d*_k*_chi*; do
  [[ -d "$run_dir" ]] || continue
  base="$(basename "$run_dir")"
  echo "Processing $base"
  python3 "$ANALYZER" "$run_dir"
  if [[ -f "$run_dir/analysis_plot.png" ]]; then
    cp "$run_dir/analysis_plot.png" "$PLOTS_DIR/${base}_analysis.png"
  else
    echo "Warning: analysis_plot.png missing in $run_dir" >&2
  fi
done

echo "All done. Plots collected in $PLOTS_DIR"
