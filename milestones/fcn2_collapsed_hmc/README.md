# Fast equilibrium sampling for FCN2 erf nets

> **Resuming the HMC-vs-theory work? Read [FINDINGS.md](FINDINGS.md) first.** It covers the results, the open
> problem, recommended sampler settings and how to rerun on a new machine. This README describes the sampler
> as of 2026-10-23.

These tools sample the same posterior as `fcn2_erf_hidden_kernel/train_fcn2_erf_sigma_a.py` (its Langevin stationary distribution), in minutes instead of days, and get past the metastable states that trap plain Langevin at large d.

## Why plain Langevin is slow
1. **Stiffness.** The readout A is about 1e3–1e4× stiffer than the read-in W0. One shared lr is capped by A, so W0 crawls; at the steepwell points, relaxing the W0 prior alone takes about 1e7 steps.
2. **Barriers.** At large d (β=100 steepwell), each neuron sits either unaligned (w0≈0) or aligned with the teacher (|w0|≈μ). Changing the number of aligned neurons means crossing barriers, so a chain keeps whatever count it started with. The 60M-epoch β=100 Langevin run is stuck in one of these states.

## The tools
| File | What it does |
|---|---|
| `collapsed_hmc_fcn2.py` | Integrates A out exactly (an exact Born–Oppenheimer step): U(W0) = \|W0\|²/2σ_W² + ½yᵀK⁻¹y + ½log det K, K = σ_A²ΦΦᵀ + (T/2)I. HMC on W0 with exact Metropolis correction, exact per-neuron sign flips, and exact draws of A given W0. Fixes (1). |
| `neuron_gibbs.py` | Exact per-neuron Gibbs jumps. The conditional of one neuron given all others (the finite-N mean-field single-neuron potential, by a Schur complement on the N×N matrix) is sampled with iterated SIR from a broad proposal, so neurons jump between wells in one move. Fixes (2). |
| `train_collapsed_hmc.py` | Runner: HMC plus Gibbs sweeps (`--gibbs-every`). **Recommended.** |
| `train_langevin_split.py` | Langevin training with per-layer lr (lr_W = c_W/λ_W, lr_A = c_A/λ_A, curvatures re-estimated by power iteration), optional Gibbs sweeps plus exact A redraws. Same target as the original trainer; still has O(lr) step-size bias. |
| `bookkeeping.py`, `common.py` | Shared run folders, CLI groups, initial states, Langevin-snapshot comparison. |
| `runs.py` | Registry CLI: `list`, `show`, `compare --group`, `rhat --group` (Gelman–Rubin across chains). |
| `launch_b100_multichain.sh` | 4-chain convergence check at β=100 (`GROUP`, `DEVICES`, `EXTRA` env vars). |

## Recipes
β=100 steepwell (d=50, P=11998, N=3162), four chains, about 3–5 min to equilibrium per chain:

    GROUP=b100_gibbs DEVICES="cuda:0 cuda:1" \
      EXTRA="--gibbs-every 3 --gibbs-mode w0 --iters 150 --burn 60" ./launch_b100_multichain.sh
    python runs.py rhat --group b100_gibbs

Split-lr Langevin with jumps:

    python train_langevin_split.py --d 50 --P 11998 --N 3162 --T 0.004743 --sa0 0.0009 --eps 0.5 \
      --dtype float32 --steps 40000 --burn 15000 --gibbs-every 2000 --gibbs-mode both --gibbs-sperp 0.16 \
      --final-frac 0.25 --final-scale 0.25 --init prior --tag b100_split --group b100_split_both

## Bookkeeping
Each run gets its own folder, `runs/<stamp>_<tag>_d_P_N_<init>_r<seed>/`, containing:
- `config.json`: arguments, git commit, host, command, `data_fingerprint`
- `trace.csv`: U or E/T, acceptance or lr, Gibbs changed/crossed fractions
- `obs.csv`: observables, burn-in flagged
- `samples/` and `checkpoint.pt`
- `summary.json`: posterior means, plus the Langevin-snapshot comparison

`runs/registry.jsonl` is an append-only event log; statuses are running, resumed, done, failed and invalid. Use `--resume <run_dir>` to continue or extend a run.

## Results (2026-10-23)
**β=1** (d=5, P=160, N=1000): a single basin. Collapsed HMC (100 s) and split-lr Langevin (200k steps) agree: w0² 0.353, w_perp² 0.207, L3 0.326. The 60M-epoch run sits about 2–5% away, consistent with its own step-size bias.

**β=100** (d=50, P=11998, N=3162):
- **HMC without jumps** gives 4 chains in 4 different metastable states, with R̂ 4–80.
- **HMC with Gibbs jumps** gives 4 chains (prior ×2, small, Langevin end state) that agree, with R̂ ≤ 1.09. Each converges in about 10–15 iterations, roughly 3–5 min.

| β=100 | Equilibrium (HMC + jumps) | 60M-epoch Langevin |
|---|---|---|
| L3 | 0.535 | 0.300 |
| w0² | 0.282 | 0.102 |
| fraction aligned (\|w0\| > 2.5σ_W) | 0.358 | 0.101 |
| w0 kurtosis | 4.47 | 18.0 |
| w_perp² | 0.0259 | 0.0417 |
| test MSE | 0.108 | 0.165 |

So the β=100 steepwell Langevin posterior is a metastable state, not equilibrium. At equilibrium about 36% of neurons are aligned, not 10%.

## Pitfalls
- **CUDA randn differs by GPU model for large tensors.** P×d = 600k gives different X on the 4090 (`cuda:0`) and the 4080 (`cuda:1`); P×d ≤ 114k is identical. `make_data` always draws on `cuda:0`, where the Langevin runs drew their data, and runs refuse to resume on different data.
- **HMC trajectory length** (step × leapfrog) must be about 1 or more. Shorter trajectories look converged when they aren't.
- **`--gibbs-mode w0`** keeps w_perp fixed. With Langevin (slow w_perp), use `both`.
