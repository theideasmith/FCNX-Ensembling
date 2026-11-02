# FCN2 read-in theory vs exact posterior (collapsed HMC): findings and handoff

Status as of 2026-11-02. Start here when resuming. The README covers the sampler itself.
Supersedes the 2026-10-27 version (commit 7fffdfe), several of whose numbers came from a coefficient bug and from
chains that had not mixed (see "Claims that turned out wrong" below).

## Question
Does the mean-field read-in theory reproduce the equilibrium posterior of a two-layer erf network in the learnable
regime, and in which limit is it exact?

## Bottom line
The theory with **a per-neuron perp norm** and **the x0 Hermite tower to He11** (`readin_action_theory_q.py`,
`analyse_q(..., ks=(1,3,5,7,9,11))`) matches the exact posterior along the κ0 = 0.015 ray, and the agreement
improves with d. With converged HMC (joint (w0, q) move, finding 1) at d = 600–1200: ⟨w0²⟩ within 0.2–1.5%,
L3 within 0.5%, the outer well within 0.02, the outer-well gap within error. The leading-order (d-independent)
normalized action has degenerate wells, and both theory and HMC show gap/β ∝ 1/d. The theory fails at small P
(d = 75, P = 750 < N): a population theory needs a training set that looks like the population.

## Setting used throughout
- Target y = He1(x0) + ε He3(x0), with normalized He3 and **ε = −0.3√6** (the paper's −0.3 on unnormalized H3).
- σ_w² = s0 = 0.5. Readout per-weight variance sa0/(Nχ), so the theory's σ_A² = sa0. κ = T/2. χ = 1.
- **κ0 = 0.015 ray**: κ/(P σ_A²) = 9.375e-4, Ĉ = σ_w² σ_A² χ P²/(N d κ²) = 474.07. Base case d = 150, N = 1500,
  P = 3000, κ = 0.015, sa0 = 8/1500. Large d at a fixed theory point: N = 1500, P = 3000 (d/150)², κ = 0.015 (d/150),
  sa0 = (8/1500)(150/d). Points: d = 75 (P = 750), 150, 300, 600, 800, 1200 (P = 192000).
- **κ0 = 0.1 ray** (grokking paper): κ/(P σ_A²) = 2.92e-3, Ĉ = 49; d = 150 (N = 700, `grok_k0p1`), 212, 424
  (`raypaper_b2`, `raypaper_b8`), with the paper's α/β scaling (N ∝ b^{1/4}, P ∝ b^{15/16}).
- Other scalings that keep the theory point fixed: P at fixed d: κ ∝ P; N: χ = N/1500.
- HMC learnabilities L1, L3 and the residuals Δ_n = y_n − f̄_n are exact population He_n(x0) coefficients of the
  posterior-mean predictor, from the sampled weights (closed-form c_n; no test set). Error bars: standard error
  across chains (`chains.py`).

## Headline result (κ0 = 0.015 ray, theory at He11)
| d | chains | ⟨w0²⟩ HMC / theory | L3 HMC / theory | L1 HMC / theory | outer well HMC / theory | gap (nats) HMC / theory |
|---|---|---|---|---|---|---|
| 75 | 3 | 0.0381 / 0.0429 (+12.6%) | 0.318 / 0.391 (+23%) | 1.051 / 1.014 | 1.07 / 1.15 | 4.11 / 3.91 |
| 150 | 3 | 0.0382 / 0.0388 (+1.7%) | 0.370 / 0.387 (+4.7%) | 1.025 / 1.014 | 1.09 / 1.16 | 3.95 / 3.94 |
| 300 | 3 (w0q) | 0.0360 / 0.0367 (+1.9%) | 0.376 / 0.385 (+2.4%) | 1.023 / 1.014 | 1.13 / 1.17 | 3.97 / 3.96 |
| 600 | 2 (w0q) | 0.0354 / 0.0356 (+0.5%) | 0.381 / 0.383 (+0.4%) | 1.013 / 1.015 | 1.15 / 1.17 | 3.96 / 3.96 |
| 800 | 2 (w0q) | 0.0353 / 0.0353 (−0.2%) | 0.382 / 0.382 (0.0%) | 1.012 / 1.015 | 1.16 / 1.17 | 3.92 / 3.97 |
| 1200 | 1 (w0q) | 0.0345 / 0.0350 (+1.5%) | 0.382 / 0.382 (0.0%) | 1.010 / 1.015 | 1.175 / 1.173 | 3.95 / 3.97 |

Groups: `larged_d75`, `pscan_d150_P3000`, `mix_d300`, `w0q_d600`, `w0q_d800`, `w0q_d1200`. Per-chain errors are
≤ 0.0003 in ⟨w0²⟩ and ≤ 0.001 in L1, L3. d = 1200 has one chain only (the over-aligned chain was lost when the 4080
fell off the bus), so it carries no convergence check. Aligned perp norm q_al/q_prior: HMC 0.55–0.59 at d ≥ 150,
theory 0.56–0.58.
Figures: `figures/large_d_actions.png`, `figures/gap_scaling.png`, `figures/gap_lineup.png`, `figures/tower_cutoff.png`.

## Findings

### 1. The sampler is exact; at d ≥ 300 it needs the joint (w0, q) move to mix
`train_collapsed_hmc.py`: the readout is integrated out exactly, HMC moves W0, and exact per-neuron moves (Schur
complement conditional, multiple-try with a state-independent proposal; `neuron_gibbs.py`) resample neurons.
- **The w0-only move (`--gibbs-mode w0`, graph-captured with `--gibbs-graph`) does not mix the aligned fraction at
  d ≥ 300.** Switching wells needs w0 and q = |w_perp|² to change together (aligned neurons have q ≈ 0.35–0.6 of
  the prior), and HMC moves q too slowly. Chains froze at their own aligned fraction, 10–25% apart between prior and
  theory starts (`larged_d300/600/800/1200`; don't use those groups for d ≥ 300 numbers).
- **Fix: `--gibbs-mode w0q`** (`NeuronGibbs` + `JointProposal`, not available with `--gibbs-graph`): resamples
  (w0, q) jointly, keeping w_perp's direction (w_perp → √(q'/q) w_perp). Proposal: 70% the theory's own p(w0, q),
  30% broad (w0 ~ N(0, 0.6²), q log-uniform on [0.05, 2]·q_prior). The weight carries the polar volume q^{(d−3)/2}.
  Exactness checks (CPU, d = 20): with no data it reproduces the prior (w0 sd 0.1588 vs exact 0.1581; q quantiles to
  ~0.005); with weak data it matches the `full` move in every quantile of |w0| and q.
- **Mixing check** (`mix_d300`): chains from the prior (aligned fraction 0.036), theory (0.034) and an over-aligned
  start (0.065) all converge to 0.0384 ± 0.0002, ⟨w0²⟩ 0.0360 (agree to 0.2%); the over-aligned chain within 100
  iterations. At d = 600 and 800, prior vs over-aligned agree in ⟨w0²⟩ to 1.4% and in aligned fraction to 3–6%
  (was 20–25% with the w0 move). `mix_d150` matches the well-mixed w0-move chains at d = 150.
- Settings: `--dtype float32 --leapfrog 24 --gibbs-every 6 --gibbs-mode w0q`, 600 iterations, burn-in 200. On the
  power-capped 4080: ~8 min per chain at d = 300, ~19 min at 600, ~32 min at 800, ~77 min at 1200.
  Trajectories with L ≤ 6 fail (w_perp random-walks). Block-MH Gibbs (`--gibbs-block`) is exact but mixes worse.
- Pooling chains that have not mixed biases histogram-based quantities: averaging peaks at slightly different
  positions flattens them (the pooled outer-well gap at d = 600 was 4.17 nats vs 3.84 and 3.97 per chain). Compute
  such quantities per chain (`chains.py`).

### 2. The original theory's error is real and no finite-size limit removes it
The original theory (`readin_action_theory.py`: frozen perp norm, He1–He3) puts the aligned neurons too far out:
⟨w0²⟩ +40% / +39% / +36% at d = 150 / 300 / 600, outer well 1.52–1.54 (HMC 1.09–1.15), L3 +11% at d = 150. The
gap is set by learnability (κ ≥ 0.2 agrees within 1–2%) and does not shrink with N (750–3000), P (3k–24k at fixed
theory point) or d.

### 3. Two corrections, both needed
(a) **The perp norm is a per-neuron variable.** Integrating out the direction of w_perp ∈ R^{d−1} gives the volume
q^{(d−3)/2}, and the single-neuron action is S(w0, q) = β[w0² + q − (D·c(w0, q))²] − ((d−3)/2) log q, with
β = d/(2σ_w²), D = √Ĉ (y − f̄). Every term is O(d), so the saddle q*(w0) is d-independent: large d makes q
deterministic given w0, not frozen at the prior. Unaligned neurons sit at the prior (1.001–1.005 of q_prior,
with exactly the prior's spread, which shrinks as 1/√d); aligned ones at 0.55–0.59 (in-well ~0.33–0.39), flat in d.

(b) **The x0 Hermite tower goes to He11.** Erf neurons carry He5, He7, … of x0, so f̄ has spurious f̄5, f̄7, …
(HMC at d = 150: f̄ = (1.025, −0.272, 0.116, −0.057, 0.030, −0.016) for n = 1, 3, …, 11). The truncation
criterion is the mixing J_15, J_35 vs λ, not J_55: J is nearly rank one (J_15/√(J11 J55) = 0.92) and
J_15 = 1.3e-3 > λ = 9.4e-4 while J_55 = 1.4e-4 < λ. Converged by He11 (He13 changes L3 by < 0.2%).
Truncation error in the theory's L3 relative to He13 (`figures/tower_cutoff.png`, `plot_tower_cutoff.py`):

| cutoff | κ0 = 0.015 ray (d = 150 / 300 / 600) | κ0 = 0.1 ray (d = 150 / 212 / 424) |
|---|---|---|
| He3 | +17.3% / +16.2% / +15.4% | +7.4% / +7.1% / +6.8% |
| He5 | +5.2% / +5.1% / +5.1% | +1.5% / +1.5% / +1.5% |
| He7 | +1.7% / +1.7% / +1.8% | +0.4% / +0.4% / +0.3% |
| He9 | +0.6% | +0.1% |
| He11 | +0.2% | 0.0% |

At high learnability Δ1 ≈ 0.01–0.03 and the spurious modes carry ~97% of the non-He3 residual power (d = 150);
on the paper ray Δ1 ≈ 0.11–0.15 dominates (~18%). The tower also decays more slowly at high learnability.

Ablation at d = 150 / 300 / 600 (⟨w0²⟩ gap vs HMC, theory at He7): frozen + He3 +40/+39/+36%; frozen + He7
+25/+24/+22%; per-neuron q + He3 +18/+16/+12%; per-neuron q + He7 +3/+1/−1%. Neither correction alone suffices.

### 4. Large-d limit: the minima line up, gap/β ∝ 1/d
In the normalized action −log p(w0)/β, the leading-order action Ŝ_∞(w0) is d-independent and its two wells are
degenerate (a finite gap would make the aligned fraction go to 0 or 1 exponentially in d; self-consistency needs
~4%). The raw gap −log p(w_out) + log p(0) is O(1): theory 3.94 → 3.97 nats from d = 150 to 4800, fitted as
A + B/d with A = 3.973, B = −4.7; gap/β ∝ d^{−0.997}, and the intercept of gap/β vs 1/d is (4 ± 2)×10⁻⁶. HMC
(per chain): 3.95 / 3.97 / 3.96 / 3.92 / 3.95 nats at d = 150 / 300 / 600 / 800 / 1200, within error of the theory.
`plot_gap_scaling.py` → `figures/gap_scaling.png`, `figures/gap_lineup.png`.

### 5. The theory's drive is correct at the population level
For aligned neurons moved to a test w0 at their own w_perp, the exact overlap with the cavity residual over the
input distribution is 1.04 / 1.02 / 1.00 / 0.99 × the theory's β(√Ĉ Δ·c)² at w0 = 0.7 / 1.0 / 1.3 / 1.6, with HMC's
own residual (`diagnostics/drive_shape.py`, d = 150). Rotation-invariant perp modes (functions of x0 and
|x⊥|²/(d−1)) are negligible: share −0.0010 (`diagnostics/radial_and_action.py`, `readin_action_theory_radial.py`).

## Remaining discrepancies
- **Small P (finite dataset).** d = 75 (P = 750 < N = 1500): L3 +23%, L1 overshoots to 1.05 (the posterior mean
  fits the training points). At d = 150 a small residual remains (L3 +4.7%, outer well +0.07): the exact per-neuron
  conditional action puts the well at 1.10, the theory's drive form with HMC's own residual at 1.22, the theory with
  its own residual at 1.18 (`diagnostics/radial_and_action.py`). The training-set drive is tilted toward small w0
  relative to the population drive (×1.39 → ×0.92 from w0 = 0.7 to 1.6) and the training residual is 19% smaller in
  mean square than on fresh inputs (`diagnostics/residual_insample.py`). All of it shrinks with d along the ray
  (P ∝ d²). Candidate explanation, untested: in-sample explaining-away, which a population theory misses; it would
  need a cavity/TAP treatment of the training-set kernel.
- **κ0 = 0.1 ray at d = 424**: HMC L1 0.850 vs theory 0.877, while d = 150 / 212 agree (0.887 / 0.884,
  0.875 / 0.881). Those runs used the w0-only move; rerun with w0q before reading anything into it.
- The isotropic effective-ridge correction (`effective_ridge_theory.py`, built on the original theory) and the
  Onsager (χR⁻¹) term are small.

## Claims that turned out wrong (don't reuse)
- **Erf Hermite coefficients for n ≥ 5** were wrong in five files: (−1)^m (n−1)!! instead of (−2)^m (n−2)!!
  (n = 2m+1; equal for n = 1, 3; c5 was 1.5× and c7 2.5× too small). Fixed 2026-10-29; `verify_hermite_coeffs.py`
  checks every copy against quadrature. Numbers that came from the bug: "corrected theory ⟨w0²⟩ gap +10.5 / +8.8 /
  +6.2%", "outer well 1.30", "He9 and He11 change nothing", "the theory overestimates Δ5 and Δ7 by 1.3–1.4×"
  (correct: 1.03–1.16×), and the "remaining outer-well offset 1.30 vs 1.10" (old open problem 5).
- **"Chains agree to 1–2% at every point"**: true only up to d = 150 with the w0 move (finding 1).
- **HMC gaps 4.10 / 4.17 nats and a non-monotonic HMC gap in d**: artifacts of pooling unmixed chains.
- "Radial perp modes cost −6% of the drive": Monte Carlo artifact (exact −0.0010).
- "The training residual has 2× the He5 of the population": least-squares artifact on a finite sample.
- "The He7 residual in the theory is 3× too small": Monte Carlo noise.

## Theory-only result
At d = 150 the original theory opens an inner double well (centre splits to ±0.15–0.19) between L3 ≈ 0.41 and 0.61
as P grows (`plot_theory_action_vs_P.py`, `figures/theory_action_vs_P_d150_k0.015.png`). Not rechecked with the
corrected theory.

## Hardware (zohar-gpu-2)
- GPUs fell off the PCIe bus (Xid 79) on 2026-10-27 (4080 at 07:30; 4090 at 13:41, ~89 °C) and on 2026-11-01 20:58
  (4080, at 65–77 °C under the governor, 20 min into a d = 1200 chain). The last one was not thermal: suspect power
  delivery (PSU, cables, riser/slot). Check the hardware before long full-load runs.
- Under full load the 4090 peaks ~82 °C even at its 150 W minimum cap (about 10 °C hotter than the 4080 at every
  load; likely its cooling). The 4080 at 180 W is usable. Caps (`sudo nvidia-smi -i <idx> -pl <W>`) reset on reboot.
- `gpu_governor.py` pauses a run's processes (SIGSTOP) at ≥ 76 °C and resumes them at ≤ 68 °C; it cannot stop
  short spikes (GPU work already queued keeps running) and causes rapid power swings. `gputemp.sh` shows live temps.
- nvidia-smi index 0 = 4080, 1 = 4090; torch cuda:0 = 4090, cuda:1 = 4080. Training data is always drawn on cuda:0,
  so chains on either GPU share the dataset. Don't restrict CUDA_VISIBLE_DEVICES to one GPU for a chain.

## How to continue on a new machine
1. **Run data is not in git.** The repo ignores `*.json`, `*.csv`, `*.pt`, `plots/`, `runs/logs/` and
   `runs/inits/`; only `runs/registry.jsonl` is committed. The run folders live on zohar-gpu-2 under
   `milestones/fcn2_collapsed_hmc/runs/`. Otherwise rerun with the launchers:
   - `launch_large_d_w0q.sh` (κ0 = 0.015 ray, d = 300–1200, w0q move, prior + over-aligned starts; use this for d ≥ 300);
   - `launch_pscan_d150.sh` (P-scan at d = 150), `launch_nscan_d150.sh` (N-scan);
   - `launch_large_d.sh` (w0 move; fine for d ≤ 150, does not mix at d ≥ 300);
   - `launch_grok_multichain.sh` (κ sweep, N = 700), `launch_ray_paper.sh` (κ0 = 0.1 ray).
   Theory-sampled starts: `ray_driver.theory_init` (the launchers build them if missing).
2. **Training data** (`collapsed_hmc_fcn2.make_data`, seed 0) is drawn on cuda:0, an RTX 4090. CUDA randn depends on
   the GPU model; without a GPU, `make_data` reproduces the 4090 stream on the CPU (`cuda_randn_emulator.py`). On a
   different GPU model run with `CUDA_VISIBLE_DEVICES=""` or add the model to `cuda_randn_emulator.GPUS`, otherwise
   the data differ and the runner refuses to resume.
3. **Open items**: the second (over-aligned) chain at d = 1200; the κ0 = 0.1 ray with the w0q move; the small-P
   correction (Remaining discrepancies).

## File map
| file | role |
|---|---|
| `train_collapsed_hmc.py`, `collapsed_hmc_fcn2.py`, `neuron_gibbs.py`, `fast_moves.py`, `common.py` | sampler (`neuron_gibbs.JointProposal`: w0q proposal) |
| `bookkeeping.py`, `runs.py` | run folders, registry, `runs.py list/compare/rhat` |
| `chains.py` | per-chain observables and standard errors; cached per-chain predictor coefficients |
| `readin_action_theory_q.py` | **the theory**: per-neuron perp norm, `ks` = x0 Hermite tower (`KernelQ`, `solve_loop_q`, `analyse_q`) |
| `readin_action_theory.py` | original theory (frozen perp, He1–He3) and `reduced_params` |
| `readin_action_theory_radial.py` | plus rotation-invariant perp modes (negligible) |
| `effective_ridge_theory.py` | isotropic effective-ridge correction on the original theory (small) |
| `verify_theory_solution.py`, `verify_hermite_coeffs.py` | checks of the solver and of every c_n copy |
| `plot_large_d_actions.py` | action plot along both rays, learnability inset, per-chain error bars |
| `plot_gap_scaling.py` | outer-well gap vs d; minima line-up figure |
| `plot_tower_cutoff.py` | truncation error vs cutoff and residual spectrum, both rays |
| `plot_readin_action*.py`, `plot_pscan.py`, `plot_theory_action_vs_P.py`, `plot_ray_actions.py` | other action plots |
| `analyze_large_d.py` | HMC vs theories along the large-d sequence (`hmc_exact`) |
| `diag_cavity_action.py`, `diag_effective_coupling.py`, `plot_cavity_drive.py` | exact per-neuron action and drive |
| `diagnostics/` | drive shape, radial modes, in-sample residual, leakage, L3 variance split |
| `gpu_governor.py`, `gputemp.sh` | thermal governor; live GPU temperatures |
| `cuda_randn_emulator.py` | CPU reproduction of the 4090's CUDA randn |

## Figures (committed copies in `figures/`; scripts write to the gitignored `plots/`)
- `large_d_actions.png`: −log p(w0)/β along both rays, HMC (per-chain error bars) vs theory, learnability inset.
- `gap_scaling.png`, `gap_lineup.png`: outer-well gap vs d; the normalized minima lining up as d → ∞.
- `tower_cutoff.png`: truncation error vs cutoff and the residual spectrum |Δ_n|, both rays.
- `pscan_d150.png`: fixed-d P-scan (original theory).
- `readin_action_sweep_grok_d150.png`, `readin_action_sweep_grok_cavity.png`: κ sweep at d = 150, original theory.
- `cavity_drive_grok_k0p07.png`: measured per-neuron drive vs the theory's β(D·c)².
- `theory_action_vs_P_d150_k0.015.png`: original theory only; the inner double well opens as P grows.
