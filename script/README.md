# Scripts Overview

Quick guide to the scripts in this folder.

## Training / Sweeps
- `ensembling_fcn3_erf_cubic.py` – Train FCN3 ensemble with erf activation (He1 + cubic target).
- `ensembling_fcn3_linear.py` – Train FCN3 ensemble with linear activation (He1 target only).
- `fcn3_erf_cubic_sweep.py` – Sweep utility for erf/cubic experiments.
- `d_sweep_fcn3_linear.sh` – Bash sweep launcher for linear runs (various d, N, P).
- `train_and_analyze_d_scale_10_15.py` – Training/analysis workflow for specific d scales.
- `train_and_compare_gpr_d2.py` – Train FCN3Erf on d=2 and compare with exact GPR.

## Analysis / Diagnostics
- `analyze_d_sweep.py` – Post-hoc analysis for erf/cubic d-sweep experiments.
- `analyze_d_sweep_linear.py` – Post-hoc analysis for linear d-sweep experiments.
- `diagnostics_langevin_runtime.py` – Runtime diagnostics for Langevin training.
- `plot_eigenspectra_vs_P.py` – Plot eigenspectra vs sample count P.
- `experiment_analyzer.py` – Helpers to load and summarize experiment outputs.
- `experiment_collections.py` – Defines experiment groupings/collections.

## Utilities / Logs
- `d_sweep_fcn3_linear.log`, `nohup.out`, `logs/`, `plots/` – Output artifacts from long-running sweeps.
