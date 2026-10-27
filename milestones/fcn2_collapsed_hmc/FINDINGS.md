# FCN2 read-in theory vs exact posterior (collapsed HMC): findings and handoff

Status as of 2026-10-27. Start here when resuming. The README covers the sampler itself.

## Question
Does the mean-field read-in theory (`readin_action_theory.py`: action β[w² − (D·c(w))²], He1–He3 block,
perp norm frozen at T = 1 + 2σ_w²(d−1)/d, loop (I + J/λ)D = √Ĉ y) reproduce the equilibrium posterior of a
two-layer erf network in the learnable regime? If it doesn't, which limit or correction closes the gap?

## Setting used throughout
- Target y = He1(x0) + ε He3(x0), with normalized He3 and **ε = −0.3√6** (the paper's −0.3 on unnormalized H3).
- σ_w² = s0 = 0.5. Readout per-weight variance sa0/(Nχ), so the theory's σ_A² = sa0. κ = T/2.
- Main theory point, the "κ0 = 0.015 ray": λ = κ/(P sa0) = 9.375e-4, Ĉ = σ_w² sa0 χ P²/(N d κ²) = 474.07,
  theory L3 ≈ 0.41. Base case d = 150, N = 1500, P = 3000, κ = 0.015, sa0 = 8/1500.
- The theory point is held fixed (λ, Ĉ unchanged) when scaling:
  - P at fixed d: κ ∝ P;
  - N: χ = N/1500;
  - d (large-d runs): N = 1500 fixed, P = 3000 (d/150)², κ = 0.015 (d/150), sa0 = (8/1500)(150/d).

## Findings

### 1. The sampler is exact, equilibrates, and is fast
`train_collapsed_hmc.py`: the readout is integrated out exactly, HMC moves W0, exact per-neuron Gibbs jumps
move w0 (the conditional is a Schur complement; `neuron_gibbs.py`; graph-captured fast version
`fast_moves.GraphedNeuronGibbs`).
- Chains from the prior, small and theory-sampled starts agree to 1–2% at every point. A start drawn from
  the theory's own distribution relaxes to the HMC state within about 10–20 iterations.
- Recommended settings, chosen by measured ESS: `--dtype float32 --leapfrog 24 --gibbs-every 6 --gibbs-mode w0
  --gibbs-graph`, 600–800 iterations, burn-in 200. That's about 1 minute per chain at d = 150 on a 4090.
  Trajectories with L ≤ 6 fail (w_perp does a random walk). Block-MH Gibbs (`--gibbs-block`) is exact but
  mixes worse. CUDA graphs on the HMC gradient give no gain.

### 2. The original theory's error is real and no finite-size limit removes it
The theory puts the aligned ("outer-well") neurons too far out:

| d = 150, κ0 = 0.015 | ⟨w0²⟩ | outer well | L3 |
|---|---|---|---|
| HMC | 0.0382 | ±1.10 | 0.369 |
| original theory | 0.0533 (+40%) | ±1.52 | 0.412 (+12%) |

The gap is set by learnability (it appears with feature learning; κ ≥ 0.2 agrees within 1–2%) and is
insensitive to:
- **N** (750 / 1500 / 3000 at a fixed theory point): no change beyond 2%;
- **P** (3k → 24k at a fixed theory point): no change, even though the training-set "leakage" (mixed modes
  He2(x0)·x_j aliasing onto He3(x0) on the finite sample) falls 21% → 6%;
- **d** (150 → 600): no change (+40% → +37%).

Plots: `figures/readin_action_sweep_grok_d150.png` (κ sweep at d = 150, N = 700), `figures/pscan_d150.png`,
`figures/large_d_actions.png`.

### 3. Two corrections to the theory, both verified against HMC
(a) **The perp norm must be a per-neuron variable.** Aligned neurons shrink |w_perp|² to about 0.56 of the
prior on average (in-well about 0.37), at every d. Because the drive and the perp entropy both scale with d,
this is an O(1) effect at any d. The single-neuron action becomes
S(w0, q) = β[w0² + q − (D·c(w0, T = 1 + 2q))²] − ((d−3)/2) log q, integrated over q.

(b) **The x0 Hermite tower must go beyond He3.** Erf neurons carry He5 and He7 of x0, so the posterior-mean f
has spurious f5 and f7 and the residual inherits −f5 and −f7. Use c = (c1, c3, c5, c7), a 4×4 J and
y = (1, ε, 0, 0). He9 and He11 change nothing.

Both are in `readin_action_theory_q.py` (`analyse_q(..., ks=(1,3,5,7))`).

| | ⟨w0²⟩ gap vs HMC | outer well (HMC ≈ 1.10–1.14) | aligned q/q0 (HMC 0.54–0.59) |
|---|---|---|---|
| original, d = 150 / 300 / 600 | +40% / +39% / +37% | 1.52–1.54 | 1 (assumed) |
| corrected, d = 150 / 300 / 600 | **+10.5% / +8.8% / +6.2%** | 1.30–1.31 | 0.57 / 0.58 / 0.60 |

The corrected theory's ⟨w0²⟩ error shrinks with d (roughly d^−0.8). The centre well matches exactly at every d.
Plot: `figures/large_d_actions.png` (all d on one axis: HMC points, original dashed, corrected solid).

### 4. With these corrections, the theory's drive is correct at the population level
For aligned neurons moved to a test w0, keeping their own w_perp, the exact overlap with the cavity residual
over the input distribution matches the theory's β(√Ĉ Δ·c(w0, q))² to within 2–5% for w0 ≥ 0.7, using HMC's
own residual (`diagnostics/drive_shape.py`). The rotation-invariant perp modes (functions of x0 and
|x⊥|²/(d−1)) are negligible: their exact share is −0.0008, and including them in the theory
(`readin_action_theory_radial.py`) changes nothing.

## Open problem (5): the remaining outer-well offset (theory 1.30 vs HMC 1.10) comes from the training set
- The exact per-neuron conditional action at its own w_perp, averaged over aligned neurons, puts the outer
  well at 1.10 (HMC). The theory's drive form fed with HMC's own residual puts it at 1.46
  (`diagnostics/radial_and_action.py`).
- The drive the neurons feel is computed on the **training** points. Relative to the population drive it is
  ×1.10 at w0 = 0.7, ×0.95 at 1.0, ×0.85 at 1.6, and nearly the same at P = 24000 (fixed λ). It tilts toward
  small w0 and pulls the wells in.
- On the training points the residual is about 20% smaller in mean square than on fresh inputs.
- Hypothesis, untested: in-sample explaining-away. The posterior-mean readout satisfies (2/T) Φᵀe = A/σ_A² on
  the training set, so a candidate feature resembling existing aligned features overlaps the training
  residual less than the population residual. A population mean-field theory misses this. It probably needs
  a cavity/TAP-style treatment of the training-set kernel: work it out on paper first.
- A uniform rescaling of the drive cannot fix the well: ×0.7 moves it only to 1.24 and spoils ⟨w0²⟩ and the
  aligned fraction. The missing piece has to be w0-dependent.

### Claims made during the investigation that turned out wrong (don't reuse)
- "Radial perp modes cost −6% of the drive": a Monte Carlo decomposition artifact (exact value −0.0008).
- "The training residual has 2× the He5 of the population": a least-squares artifact (high-order Hermite fits
  on a finite sample are ill-conditioned). Plain sample averages agree with fresh inputs.
- "The He7 residual in the theory is 3× too small": Monte Carlo noise. Exactly, the theory overestimates Δ5
  and Δ7 by about 1.3–1.4×.
- The isotropic effective-ridge correction and the Onsager (χR⁻¹) term are both far too small (1e-4 of the drive).

## Theory-only result
At d = 150, the original theory opens an inner double well (centre minimum splits to ±0.15–0.19) between
L3 ≈ 0.41 and 0.61 as P grows (`plot_theory_action_vs_P.py`, `figures/theory_action_vs_P_d150_k0.015.png`).
The κ0 = 0.015 ray sits just before that transition.

## How to continue on a new machine
1. **Run data is not in git.** The repo ignores `*.json`, `*.csv`, `*.pt` and `plots/`, so only
   `runs/registry.jsonl` (the run index, with parameters and headline results per run) is committed. The run
   folders (samples, checkpoints, configs, summaries) live on zohar-gpu-2 under
   `milestones/fcn2_collapsed_hmc/runs/`; copy them over with rsync if that machine is reachable. Otherwise,
   rerun a point with the launchers:
   - `launch_pscan_d150.sh` (P-scan at d = 150);
   - `launch_nscan_d150.sh` (N-scan);
   - `launch_large_d.sh` (d = 300/600/1200, P ∝ d²; d = 1200 needs about 9 GB and about 35 min per chain);
   - `launch_grok_multichain.sh` (κ sweep, N = 700).

   Theory-sampled starting weights go in `runs/inits/` (gitignored). Rebuild them with the snippet in
   `launch_large_d.sh`'s header, or `ray_driver.theory_init`.
2. **Training data is drawn on the GPU** (`collapsed_hmc_fcn2.make_data`, seed 0). CUDA randn depends on the GPU
   model. These runs used an RTX 4090 (`cuda:0` on zohar-gpu-2). Without a GPU, `make_data` reproduces the 4090
   stream on the CPU (`cuda_randn_emulator.py`; it matches the stored `data_fingerprint` to about 1e-5,
   last-bit float32). On a different GPU model, run with no GPU visible
   (`CUDA_VISIBLE_DEVICES=""`) or add the model to `cuda_randn_emulator.GPUS` and route through it. Otherwise
   the training data differs and the runner refuses to resume.
3. **Hardware note**: on 2026-10-27 both GPUs on zohar-gpu-2 fell off the PCIe bus (Xid 79) under sustained
   load: the 4080 at 07:30, the 4090 at 13:41 while at about 89 °C. Check power and cooling, and limit
   concurrent chains.
4. **Next step for (5)**: derive the training-set cavity correction. Then test it once with
   `diagnostics/drive_shape.py` and `diagnostics/radial_and_action.py` at d = 150 (they work on CPU).

## File map
| file | role |
|---|---|
| `train_collapsed_hmc.py`, `collapsed_hmc_fcn2.py`, `neuron_gibbs.py`, `fast_moves.py`, `common.py` | sampler |
| `bookkeeping.py`, `runs.py` | run folders, registry, `runs.py list/compare/rhat` |
| `readin_action_theory.py` | original theory (frozen perp, He1–He3) |
| `readin_action_theory_q.py` | corrected theory: per-neuron perp norm, `ks` = x0 Hermite tower |
| `readin_action_theory_radial.py` | plus rotation-invariant perp modes (negligible; d ≥ 600 quadrature needs fixing) |
| `effective_ridge_theory.py` | isotropic effective-ridge correction (small) |
| `verify_theory_solution.py` | independent checks of the theory solver |
| `plot_readin_action*.py`, `plot_large_d_actions.py`, `plot_pscan.py`, `plot_theory_action_vs_P.py` | action plots |
| `analyze_large_d.py` | HMC vs both theories along the large-d sequence |
| `diag_cavity_action.py`, `diag_effective_coupling.py`, `plot_cavity_drive.py` | exact per-neuron action and drive from samples |
| `diagnostics/` | scripts behind findings 4 and 5 and the retractions (radial modes, drive shape, train-vs-fresh residual, leakage, L3 variance split) |
| `cuda_randn_emulator.py` | CPU reproduction of the 4090's CUDA randn |

## Figures (committed copies in `figures/`; scripts write to the gitignored `plots/`)
- `figures/large_d_actions.png`: all d on one axis. HMC points, original theory dashed, corrected theory solid.
- `figures/pscan_d150.png`: fixed-d P-scan; the theory is identical at every P, HMC barely moves.
- `figures/readin_action_sweep_grok_d150.png`: κ sweep at d = 150 (N = 700); HMC histograms vs original theory.
- `figures/readin_action_sweep_grok_cavity.png`: the same with HMC's exact per-neuron action (resolves the barriers).
- `figures/cavity_drive_grok_k0p07.png`: measured per-neuron drive vs the theory's β(D·c)².
- `figures/theory_action_vs_P_d150_k0.015.png`: theory only; the inner double well opens as P grows.
