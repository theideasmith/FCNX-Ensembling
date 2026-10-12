# Pure-cubic FO ρ-ray at min_π F(π)

Working phase-transition ray from the self-consistent multiwell matrix Laplace
solver (`tmp_multiwell_pure_cubic.jl --rho-ray-actions` / `--rho-branch-scan`).

## Setup

- teacher y = (0, ε = 1)
- d = 16, N = χ = 200, P = 10³, sa0 = 0.03
- control ρ = κ / (sa0 P)
- equilibrium = global min_π F(π)

## Figures

- `L3_vs_rho.png` — continuous π-tracks (colored) + global min F (black, gaps at FO)
- `learnability_and_wells_vs_rho.png` — L₃, wells, dV / π_out on the same tracks
- `actions_V_vs_w.png` — raw cavity action V(w) at the F minimum
- `summary_equilibrium.csv` — global-eq numeric summary
- `branch_local_mins.csv` — warm-started local F(π) minima vs ρ

## Continuity

Global argmin F is discontinuous at the FO (basin switch near ρ ≈ 0.078–0.076).
Colored tracks are continuous metastable branches stitched by π proximity from
`--rho-branch-scan`. Track 1 is the outer-deepening branch through the V_out
crossing (ρ ≈ 0.095→0.09); track 2 is the high-π post-FO winner.

Regenerate plots:

```bash
python3 plot_pure_cubic_rho_ray_learnability.py \
  --summary action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin/summary_equilibrium.csv \
  --branches action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin/branch_local_mins.csv \
  --outdir action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin
```
