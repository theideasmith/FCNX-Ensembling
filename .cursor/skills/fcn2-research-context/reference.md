# FCN2 VGA and empirical W0 notes (2026-09-18)

## Theory

`FCS2_VGA.jl` variationalizes teacher-axis read-in as a symmetric 2-Gaussian. Offdiag (`residuals_fcn2_offdiag`) diagonalizes the 2×2 Hermite kernel `K_H` and rotates teacher `(1,ε)→δ'`. Entropy default is Hershey–Olsen; `--advanced` is exact GMM entropy.

`χ_readout = χ_train / a0` with `a0=sa0`. Steepwell uses `χ_train=1` and small `sa0`, i.e. large `χ_readout` (deep wells at large β).

## Live steepwell ray

Launcher: `milestones/fcn2_erf_hidden_kernel/red_robin_alpha_beta_invariant_steepwell.py`

`d=5 β^{1/2}`, `N=1000 β^{1/4}`, `P=160 α^{3/4}`, `α=β^{5/4}`, `sa0=0.03 β^{-3/4}`, `κ=0.001 (α/β)^{3/4}`, `ε=0.5`, `ens=1`, `base_lr=0.01`, schedule 2,3,5 on 60M wall.

Offdiag VGA along this ray is a **double well at every β**. μ stays ≈0.36–0.41; σ shrinks (0.35→0.22) so μ/σ and the barrier grow with β. A single `fcn2_vga_erf.jl` solve with default μ₀=0.1 collapses onto the μ=0 saddle (lWT almost unchanged). Use the multi-seed scan in `plot_invariant_vga_actions_alpha_scale.jl` (μ₀∈{0.25,…,0.50}).

| β | d | μ | σ | μ/σ | lWT | epoch-2M empirical |
|---|---|---|---|---|---|---|
| 1 | 5 | 0.413 | 0.345 | 1.20 | 0.289 | var≈0.32, looks unimodal (theory p(0)/p(μ)≈0.93) |
| 3.16 | 9 | 0.377 | 0.300 | 1.26 | 0.232 | var≈0.23, still overlapping |
| 10 | 16 | 0.363 | 0.269 | 1.35 | 0.204 | not launched |
| 31.6 | 28 | 0.362 | 0.243 | 1.49 | 0.190 | not launched |
| 100 | 50 | 0.369 | 0.215 | 1.71 | 0.183 | two peaks ≈±0.18; wells still filling |

## Finished Langevin (best posterior)

`red_robin_d50_T0.2_P3000_..._weight_snaps`: 80 late snaps, empirical `W0[:,0]` peaks ≈±0.37, 50/50, var≈0.27. Offdiag VGA `κ=T/2=0.1`: μ≈0.52, lWT≈0.31. Shape matches; theory wells slightly farther out.

`P=16000` same family: still two-peaked but **asymmetric** (VGA assumes ± symmetry).

`InvariantPfixed1500` β=1 (sa0=1, κ≈2.8): empirically unimodal (var≈0.022); VGA μ≈0.11, lWT≈0.023. Weak wells.

Perp `W0[...,1:]` unimodal in all of the above.

Readout `A` on d50 P=3000 snaps is marginally bimodal at ≈±0.03 and tracks `sign(W0[:,0])` (`corr≈0.44`). That is joint (A,w₀) structure, not the VGA 1D ansatz.

## Pitfalls

- `plot_action_h0_activation.py` `get_vga_for_model` omits `--sa0`.
- Diagonal HO (no `--offdiag`) gives larger μ (steepwell d=50 μ≈0.88 vs 0.375 offdiag). Quote which residual you used.
- `μ=0` saddle: `fcn2_vga_erf.jl` default init μ₀=0.1 is too small for the steepwell ray and reports UNIMODAL. Multi-seed as in `plot_invariant_vga_actions_alpha_scale.jl`. See also `diagnose_vga_bimodality.jl`.
- Do not confuse journal `self_consistent_kappa_solver.jl` leftovers with the steepwell train.

## Analysis entry points

- Histograms / −log p overlays: `plot_action_h0_activation.py` (`--vga-offdiag`, groups `LangevinD50Kap0p1Long`, `InvariantPfixed1500Asnap`)
- Theory-only μ(β) along invariant ray: `julia_lib/plot_invariant_vga_actions_alpha_scale.jl`
