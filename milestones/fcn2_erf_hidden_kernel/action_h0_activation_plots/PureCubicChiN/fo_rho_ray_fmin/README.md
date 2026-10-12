# Pure-cubic FO ρ-ray at min_π F(π)

Working phase-transition ray from the self-consistent multiwell matrix Laplace
solver (`tmp_multiwell_pure_cubic.jl --rho-ray-actions --modes equilibrium`).

## Setup

- teacher y = (0, ε = 1)
- d = 16, N = χ = 200, P = 10³, sa0 = 0.03
- control ρ = κ / (sa0 P)
- equilibrium = global min_π F(π)

## Figures

- `L3_vs_rho.png` — cubic learnability alone (dense ρ)
- `learnability_and_wells_vs_rho.png` — L₃ + wells + dV / π_out
- `actions_V_vs_w.png` — raw cavity action V(w) at the F minimum
- `summary_equilibrium.csv` — numeric summary

FO crossing (outer wells below inner at the F minimum): between ρ = 0.095 and ρ = 0.09.
The mild dip in L₃ across that window is the outer-branch equilibrium, not sampling noise.
