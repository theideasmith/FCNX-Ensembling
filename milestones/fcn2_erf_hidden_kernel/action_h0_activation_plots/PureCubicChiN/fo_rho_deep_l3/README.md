# Deep ρ-ray: self-consistent L₃ → 1

Warm-started post-FO high-π branch of the pure-cubic multiwell matrix Laplace
(`tmp_multiwell_pure_cubic.jl --rho-deep --seed-pi 0.17`).

## Setup

- y = (0, ε = 1), d = 16, N = χ = 200, P = 10³, sa0 = 0.03
- ρ = κ / (sa0 P) from 0.07 down to 10⁻⁵
- continuation of the FO high-π basin (not the deep-outer metastable)

## What happens

| ρ | L₃ | π_out | notes |
|---|-----|-------|-------|
| 0.07 | 0.064 | 0.18 | post-FO start |
| 0.01 | 0.39 | 0.30 | |
| 0.006 | 0.51 | 0.30 | L₃ crosses ½ |
| 7×10⁻⁴ | 0.90 | 0.31 | |
| 5×10⁻⁵ | 0.99 | 0.31 | |
| 10⁻⁵ | 0.998 | 0.31 | |

- Implied He3 kernel λ ≈ L₃ ρ / (1−L₃) saturates at ≈ 0.00621.
- π_out saturates near 0.31; μ_out grows to ≈ 8.2; wells become extremely deep
  (V ~ −10³), with V_in slightly below V_out (dV_raw > 0).
- Two-well ansatz survives all the way; no collapse to nwells = 0.

## Files

- `summary_deep_equilibrium.csv`
- `L3_deep_vs_rho.png`
- `tmp_rho_deep.log`
