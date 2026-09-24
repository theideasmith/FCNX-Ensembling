---
name: fcn2-research-context
description: Cached FCN2 VGA / read-in bimodality research context for this repo. Use when a cloud agent or user asks about W0 histograms, bimodal read-in weights, readout A, VGA offdiag, steepwell invariant sweeps, A_snapshots, kappa, or comparing trained FCN2 erf nets to julia_lib theory.
---

# FCN2 research context

Read [reference.md](reference.md) before exploring the repo for VGA or weight-distribution questions.

## Do this, not a full crawl

1. Check tmux `steepwell_invariant` and `nvidia-smi` if GPU/run status matters.
2. Compare empirical `W0[..., 0]` to `julia julia_lib/fcn2_vga_erf.jl ... --laplace --matrix --sa0 <sa0>`. The matrix Laplace saddle is the current best model; `--offdiag` is the older VGA.
3. Prefer `A_snapshots/` when present; otherwise `model.pt` is a single optimizer state, not a posterior.

## Do not

- Kill training processes or occupy the 4090 without being asked.
- Call VGA for `train_fcn2_erf_sigma_a` runs without `--sa0`.
- Equate readout `A` bimodality with the VGA `w₀` mixture (they couple in sign, but the ansatz is on `W0`).
