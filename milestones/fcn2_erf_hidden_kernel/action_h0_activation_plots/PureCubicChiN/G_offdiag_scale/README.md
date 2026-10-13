# G-matrix off-diagonal diagnosis (pure-cubic ρ-ray)

Question: on the experimental χ=N ρ-ray (`d=16`, `N=200`, `P=10³`, `sa0=0.03`, `ε=1`),
are the He1–He3 amplitude off-diagonals `G₁₃` comparable to the diagonals — the coupling
that widens the central/inner mode? And does proportional large-`d` scaling close them?

## Experimental ray (`d=16`)

| ρ | \|r_G\| | \|G₁₃\|/\|G₁₁\| | \|G₁₃\|/\|G₃₃\| | σ_in | L₃ |
|---|--------|----------------|----------------|------|-----|
| 0.15 | 1.13 | 51 | 0.025 | 0.25 | 0.007 |
| 0.08 | 1.00 | 8.2 | 0.12 | 0.20 | 0.047 |
| 0.03 | 1.00 | 4.4 | 0.23 | 0.048 | 0.17 |
| 0.015 | 1.00 | 4.0 | 0.25 | 0.027 | 0.29 |

`r_G = G₁₃ / √|G₁₁ G₃₃|`. Values ≈ ±1 mean the 2×2 amplitude is **fully correlated**
(rank-1 dominated by the −χ² vvᵀ piece of `amp`, with teacher `y=(0,ε)`).

In the weak-FL window (ρ≈0.03–0.08):
- \|G₁₃\|/\|G₃₃\| is **O(0.1–0.25)** — same order as the cubic diagonal entry.
- \|G₁₃\|/\|G₁₁\| ≫ 1 because the pure-cubic drive leaves `G₁₁` small; the off-diagonal
  *dominates* the linear channel.

So this ray is **not** in a decoupled-diagonal regime. Central-mode structure should feel
the He1↔He3 mix.

## Large-`d` proportional scale (hold ρ and Finv)

Steepwell exponents from the `d=16` anchor:

`β=(d/16)²`, `N∝β^{1/4}`, `sa0∝β^{-3/4}`, `P∝α^{3/4}`, `κ∝(α/β)^{3/4}` with `α=β^{5/4}`.

This keeps `ρ=κ/(sa0 P)` and `Finv∝1/(sa0 d N)` fixed. Sweep `d=16…512`.

**Result: off-diagonals do not close.**

- `|r_G|` stays pinned at **1.0** for all `d` in the FO/deep window.
- `|G₁₃|/|G₃₃|` **increases** with `d` (e.g. ρ=0.08: 0.12 → 0.23 at `d=512`).
- `|G₁₃|/|G₁₁|` decreases slowly but remains ≫ 1.
- Inner width `σ_in` shrinks (wells sharpen); `L₃` and `π_out` drift up along the scale.

So Finv-flat large-`d` scaling does **not** open a diagonal / weak-coupling window for
staircase-free weak central FL. Killing `G₁₃` likely needs a different ansatz or teacher
(e.g. suppress induced He1 / change χ or the ridge structure), not just `d→hundreds` on
this ray.

## Reproduce

```bash
julia tmp_multiwell_pure_cubic.jl --G-scale \
  --ds 16,32,64,128,256,512 --rhos 0.15,0.08,0.03,0.015 --seed-pi 0.17
python plot_G_offdiag_scale.py
```

## Artifacts

- `G_offdiag_vs_d.png` — four-panel summary
- `G_offdiag_vs_d_from_log.csv` / `log_dscale.txt` — numbers
- `G_offdiag_vs_d_d16only.csv` — dense ρ scan at the experimental anchor
