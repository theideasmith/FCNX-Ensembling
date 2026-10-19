# Fine ρ learnability around 0.03

## Matrix continuum (primary) — no bump

Non-specialized multiwell matrix Laplace (`tmp_multiwell_pure_cubic.jl --rho-deep`,
χ=N, seed-π=0.17), 15-point grid `0.08→0.012`:

**L₃ strictly monotonic** (0.047 → 0.347 as ρ falls). No local max.

See `summary_multiwell_fine.csv`.

## Mode-Laplace specialist ansatz — apparent bump is a basin jump

Fine `solve_sc_root` scan on the same grid (`summary_fine.csv` / `rho*_scfrac.npz`):

| ρ window | f* | L₃ behavior |
|---|---|---|
| 0.15 → 0.02 | 0.05 → 0.26 | L₃ rises smoothly to ~0.19 |
| ≤ 0.018 | jumps to 0.60–0.77 | L₃ **drops** (0.14 → 0.12 → 0.12) |

So the old 4-pt “bump” (peak at 0.03, dip at 0.015) was the start of this
**high-f* basin** at low ρ, not a physical local max. Continuum theory (and
network / particles) stay monotonic through the same window.

Plot: `L3_fine_vs_rho.png` (continuum primary, mode overlay).

## Reproduce

```bash
# continuum
julia --project=../../julia_lib tmp_multiwell_pure_cubic.jl --rho-deep \
  --d 16 --n1 200 --chi 200 --P 1000 --sa0 0.03 --eps 1 --seed-pi 0.17 \
  --rhos 0.08,0.06,0.05,0.045,0.04,0.035,0.032,0.03,0.028,0.025,0.022,0.02,0.018,0.015,0.012 \
  --outdir action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine \
  --out action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine/summary_multiwell_fine.csv

# mode-Laplace
OUT_DIR=action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine \
RHOS=0.08,0.06,0.05,0.045,0.04,0.035,0.032,0.03,0.028,0.025,0.022,0.02,0.018,0.015,0.012 \
ITERS=40 N_BISECT=5 SKIP_EXISTING=1 \
python -u tmp_mode_laplace_pure_cubic_rho.py

python -u plot_mode_laplace_fine_rho.py
```
