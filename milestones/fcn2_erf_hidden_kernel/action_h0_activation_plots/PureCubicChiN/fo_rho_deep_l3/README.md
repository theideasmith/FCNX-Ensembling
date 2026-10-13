# Deep ρ-ray: χ=N vs NNGP learnability

## Setup

- y = (0, ε = 1), d = 16, N = 200, P = 10³, sa0 = 0.03
- ρ = κ / (sa0 P) from 0.07 → 10⁻⁵
- ridge always ρ' = κ/(sa0 P) (no N on κ or sa0)

## Curves

- **χ=N self-consistent** — post-FO high-π multiwell branch (`summary_deep_equilibrium.csv`). L₃ → 0.998 at ρ=10⁻⁵.
- **NNGP** — prior-kernel learnability at χ=1 amplitude (`summary_deep_nngp_frozen.csv`). Same ridge; no N-scaling. L₃ → 0.99 at ρ=10⁻⁵.
- **χ=1 self-consistent** (optional) — center/unimodal basin stays at tiny L₃ (no FO high-π condensate).

## Figures

- `L3_deep_vs_rho.png` — L₃ / well depth / π_out vs ρ (χ=N + NNGP overlay)
- `actions_V_vs_w.png` — raw cavity action V(w) on the χ=N deep branch, curves colored by ρ (viridis). Outer wells drift out to |w|≈8 as ρ→10⁻⁵ while both wells deepen to V≈−2500.
