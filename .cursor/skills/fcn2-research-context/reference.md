# FCN2 VGA and empirical W0 notes (2026-09-18)

## Theory

`FCS2_VGA.jl` variationalizes teacher-axis read-in as a symmetric 2-Gaussian. Offdiag (`residuals_fcn2_offdiag`) diagonalizes the 2×2 Hermite kernel `K_H` and rotates teacher `(1,ε)→δ'`. Entropy default is Hershey–Olsen; `--advanced` is exact GMM entropy.

`A = a0 · ℓ_T(χ/a0, κ/a0)/(n1 χ)` (`training_amplitude`): χ and the ridge are both rescaled by a0 inside ℓ_T. Steepwell uses `χ=1`, small `sa0`, and at β=1 `κ = 1/N`, `sa0 ≈ 14/N` (chosen to mimic χ=N). Without the a0 rescaling, steepwell A₁ ≈ −0.01 (prior-only); with it, A₁ ≈ 1.26·A_crit along the whole ray.

## Laplace saddle (2026-09-24) — best bimodal model

`residuals_fcn2_laplace` in `FCS2_VGA.jl`: no entropy; `V(w)=(d/2s0)w² + energy(w)`, μ from `V'(μ)=0`, σ from `σ²V''(μ)=1`. Why not VGA: KL/entropy leaves μ weakly constrained; a saddle ties μ to feature-learning strength.

- **Linear channel** (`A₁ λ₁(w)`) fails structurally: at a nontrivial well `V''(μ) = (d/s0)·8μ²/(T+2μ²) < d` for μ² < T/6, so wells are always *wider* than the prior. Measured wells are ~0.65× the prior width (V'' ≈ 2–2.5 d). Full solve collapses to μ=0 except β=100; σ=0 (`--laplace-mean`) gives μ≈0.41–0.42 (too far out).
- **Matrix channel** (`--laplace --matrix`): energy `½ c(w)ᵀGc(w)`, `G = a0/(n1χ)[−χ'²δ vvᵀ + χ'Q⁻¹]`, `v=(Q+ρ'I)⁻¹(1,ε)`, χ'=χ/a0, ρ'=κ/(a0P). The He1–He3 kernel coupling λ₁₃ is independent of ε: the ±μ neurons emit He1 and He3 locked together (Q nearly rank 1, λ₁₃²/(λ₁₁λ₃₃)≈0.7–0.86), so v₃/v₁ ≈ 2–4.5 even at ε≈0. `v·c(w)` partially cancels as |w| grows, which confines neurons beyond μ.

Steepwell ε=0.03 (N30d) matrix Laplace vs empirical bimodal fit:

| β | d | μ th | σ th | μ emp | σ emp | L3 matrix | L3 emp |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.322 | 0.349 | ~0.35 | poor fit | −1.74 | ≈−1.9 |
| 3.16 | 9 | 0.341 | 0.295 | 0.302 | 0.21 | −1.81 | ≈−2.4 |
| 10 | 16 | 0.366 | 0.228 | 0.315 | 0.16 | −1.72 | ≈−1.8 |
| 31.6 | 28 | 0.383 | 0.157 | 0.321 | 0.12 | −1.53 | ≈−1.5 |
| 100 | 50 | 0.377 | 0.092 | 0.340 | 0.10 | −1.29 | ≈−1.35 |

L3 here is `f₃/ε` with `f = Q(Q+ρ'I)⁻¹y` on the solved (μ,σ). The negative empirical cubic learnability is the network's spurious He3 output from λ₁₃, not failure to learn ε. With the bare ridge κ/P the matrix L3 stays positive (≈+0.6–0.9), so ρ'=κ/(a0P) is the right ridge. With `matrix=true`, `populate_solution_fcn2` reports these (`matrix_learnabilities`), and `plot_action_h0_activation.py` plots them when the Julia JSON has `parameters.matrix`. Non-matrix modes keep the diagonal `λ/(λ+κ/P)`.

Remaining gaps are ansatz limits: the barrier top at w=0 (Gaussian mixture is cusp-like there) and non-Gaussian walls at small d.

Measured perp norm `|w⊥|²` is 10–20% below the prior `(d−1)s0/d`, smaller for neurons in the wells than near w∥=0. That is a secondary effect (few % on μ, 7–17% on V'').

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

- Plots must use κ_eff (default in `plot_action_h0_activation.py`), never bare κ.
- Laplace root finding: the χ-anneal tracks the μ=0 root. Use `use_anneal=false` from a μ₀ (×σ₀) grid and keep the largest converged μ (done in `fcn2_vga_erf.jl`).
- Diagonal HO (no `--offdiag`) gives larger μ (steepwell d=50 μ≈0.88 vs 0.375 offdiag). Quote which residual you used.
- `μ=0` saddle: `fcn2_vga_erf.jl` default init μ₀=0.1 is too small for the steepwell ray and reports UNIMODAL. Multi-seed as in `plot_invariant_vga_actions_alpha_scale.jl`. See also `diagnose_vga_bimodality.jl`.
- Do not confuse journal `self_consistent_kappa_solver.jl` leftovers with the steepwell train.

## Analysis entry points

- Histograms / −log p overlays: `plot_action_h0_activation.py` (`--vga-offdiag`, `--vga-laplace [--vga-matrix]`, `--vga-laplace-mean`; groups `LangevinD50Kap0p1Long`, `InvariantPfixed1500Asnap`, `SteepwellInvariantEps0p03N30dAsnap`)
- Theory-only μ(β) along invariant ray: `julia_lib/plot_invariant_vga_actions_alpha_scale.jl`
