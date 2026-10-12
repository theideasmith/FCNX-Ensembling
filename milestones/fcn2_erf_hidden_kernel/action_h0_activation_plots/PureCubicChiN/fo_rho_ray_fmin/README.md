# Pure-cubic FO ρ-ray at \(\min_\pi F(\pi)\)

Working phase-transition ray from the self-consistent multiwell matrix Laplace
solver (`tmp_multiwell_pure_cubic.jl --rho-ray-actions --modes equilibrium`).

## Setup

- teacher \(y=(0,\varepsilon=1)\)
- \(d=16\), \(N=\chi=200\), \(P=10^3\), \(\mathrm{sa0}=0.03\)
- control \(\rho=\kappa/(\mathrm{sa0}\,P)\)
- equilibrium = global \(\min_\pi F(\pi)\) (hardened π-walk + free-solve fallback)

## Figures

- `learnability_and_wells_vs_rho.png` — \(L_3\), well depths, \(dV\) and \(\pi_{\mathrm{out}}\) vs \(\rho\)
- `actions_V_vs_w.png` — raw cavity action \(V(w)\) at the free-energy minimum
- `summary_equilibrium.csv` — numeric summary

FO crossing (outer wells below inner at the \(F\) minimum): between \(\rho=0.10\) and \(\rho=0.09\).
