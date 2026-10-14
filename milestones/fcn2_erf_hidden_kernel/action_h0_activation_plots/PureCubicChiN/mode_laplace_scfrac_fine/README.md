# Fine ρ learnability around 0.03

## Matrix continuum (primary)

Non-specialized multiwell matrix Laplace (`tmp_multiwell_pure_cubic.jl --rho-deep`,
χ=N, seed-π=0.17) on the fine grid:

`0.08, 0.06, 0.05, 0.045, 0.04, 0.035, 0.032, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.015, 0.012`

Result: **L₃ is strictly monotonic** (rises as ρ falls). No bump at 0.03.

Artifact: `summary_multiwell_fine.csv`, plot `L3_fine_vs_rho.png`.

## Mode-Laplace overlay

Specialist-fraction mode-space Laplace (slower). The old 4-pt “bump” came from a
suspicious ρ=0.015 root (`f*≈0.77`); the continuum theory has no such feature.

## Reproduce

```bash
julia --project=../../julia_lib tmp_multiwell_pure_cubic.jl --rho-deep \
  --d 16 --n1 200 --chi 200 --P 1000 --sa0 0.03 --eps 1 --seed-pi 0.17 \
  --rhos 0.08,0.06,0.05,0.045,0.04,0.035,0.032,0.03,0.028,0.025,0.022,0.02,0.018,0.015,0.012 \
  --outdir action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine \
  --out action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine/summary_multiwell_fine.csv

python -u plot_mode_laplace_fine_rho.py
```
