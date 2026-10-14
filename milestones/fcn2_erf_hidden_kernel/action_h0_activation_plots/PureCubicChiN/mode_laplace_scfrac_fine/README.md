# Fine mode-Laplace ρ scan around the L₃ bump

Question: is the non-monotonic L₃ peak near ρ≈0.03 in the 4-point mode-Laplace
curve a real local max, or a solver/basin artifact?

## Status

Dense scan running: ρ ∈ {0.08, 0.06, 0.05, 0.045, 0.04, 0.035, 0.032, 0.03,
0.028, 0.025, 0.022, 0.02, 0.018, 0.015, 0.012} with `ITERS=40 N_BISECT=5`.

Interim plot overlays:
- mode-Laplace points (fine scan as they finish; seeded with the old 4-pt caches)
- multiwell χ=N from `fo_rho_deep_l3/summary_deep_equilibrium.csv` (already dense; **monotonic**)

## Reproduce

```bash
OUT_DIR=action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine \
RHOS=0.08,0.06,0.05,0.045,0.04,0.035,0.032,0.03,0.028,0.025,0.022,0.02,0.018,0.015,0.012 \
ITERS=40 N_BISECT=5 SKIP_EXISTING=1 \
python -u tmp_mode_laplace_pure_cubic_rho.py

python -u plot_mode_laplace_fine_rho.py
```

## Note

Multiwell Laplace (χ=N) has **no bump** through ρ=0.03 — L₃ rises smoothly
0.08→0.01. The bump in the three-way plot is mode-space Laplace only.
