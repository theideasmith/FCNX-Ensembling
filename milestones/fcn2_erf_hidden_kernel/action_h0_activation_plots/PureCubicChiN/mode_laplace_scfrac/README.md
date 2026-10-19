# Pure-cubic χ=N mode-space Laplace (self-consistent f*)

Solver: `tmp_mode_laplace.py` `solve_sc_root` with all 154 modes (He1/3/5/7 + perps),
`S=I`, `SELF=span`, teacher `y=(0,ε)`, `σ_A²=sa0/(Nχ)`, ridge `ρ'=κ/(χ P)`.

The χ-scaled ridge is required for χ≠1: with bare `κ/P` the cubic drive is ~χ× too weak
and no wells form. (For steepwell χ=1 this reduces to the original `κ/P`.)

Params: d=16 N=200 P=1000 χ=200 sa0=0.03 ε=1 he1=0.

| ρ | f* | μ_bulk | μ_spec | emp f (\|w₀\|>1) |
|---|---|---|---|---|
| 0.15 | 0.052 | 0.00 | 1.40 | 0.060 |
| 0.08 | 0.163 | 0.25 | 2.07 | 0.148 |
| 0.03 | 0.269 | 0.51 | 2.99 | 0.065 |
| 0.015 | 0.702 | 0.54 | 2.08 | 0.051 |

At high/mid ρ, f* and specialist location track the A_snapshot posterior. At low ρ the
thermodynamic f* grows large (most neurons outer) while the trained nets still show a
small specialist fraction — same stall pattern as steepwell large-d.

Scripts:
- `tmp_mode_laplace_pure_cubic_rho.py` — solve + dump
- `plot_pure_cubic_mode_laplace_actions.py` — emp overlay

Figure: `empirical_vs_mode_laplace_scfrac.png`
