#!/usr/bin/env python3
"""
Diagnostics: measure time per Langevin-training epoch as a function of d

This script benchmarks one Langevin-training epoch for FCN3 models over a
log-spaced sweep of hidden dimension `d` (default 10..1000). For each `d`
we set `P = d**1.5` and `N = 5*d` (as requested) and run a single Langevin
update epoch, repeating to obtain mean/std timings. Outputs:
    - `plots/diagnostics_langevin_runtime.npz` (P, d, times_mean, times_std)
    - `plots/diagnostics_langevin_runtime.png` (log-log errorbar plot)

Defaults: d in [10,1000] log-spaced with 12 points, repeats=3, ens=1.

Usage:
    python3 script/diagnostics_langevin_runtime.py --d-min 10 --d-max 1000 --d-points 12

Notes:
 - This reproduces the Langevin update style from
     `ensembling_fcn3_erf_cubic.py` in a minimal way: params are updated in-place
     with additive Gaussian noise, weight-decay and gradient updates.
 - Runs on CPU by default to avoid CUDA timing/synchronization issues;
     pass `--cuda` to benchmark on GPU if available (results will reflect GPU timings).
 - Large `d` (and thus large `P = d**2`) will use substantial RAM. Reduce
     `--d-points` or upper bound if your machine has limited memory.
"""

import os
import sys
import time
import math
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure local lib is importable
PROJECT_ROOT = os.path.abspath('/home/akiva/FCNX-Ensembling/')
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

try:
    from FCN3Network import FCN3NetworkEnsembleErf
except Exception as e:
    raise RuntimeError('Could not import FCN3NetworkEnsembleErf from lib/FCN3Network.py: ' + str(e))

# Attempt to import juliacall and configure Julia LOAD_PATH to include `julia_lib`
JULIA_AVAILABLE = False
_FCS = None
_jl = None
try:
    import juliacall
    from juliacall import Main as jl
    julia_lib_dir = os.path.join(PROJECT_ROOT, 'julia_lib')
    jl.include('/home/akiva/FCNX-Ensembling/julia_lib/FCS.jl')

    _FCS = jl.FCS
    _jl = jl
    JULIA_AVAILABLE = True
except Exception as e:
    print('juliacall/FCS.jl not available or failed to initialize:', e)
    JULIA_AVAILABLE = False
import torch


def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return 0.5 * torch.sum(diff * diff)


def single_epoch_langevin(model, X, Y, lr, t, weight_decay_vec, langevin_gen, noise_buffer):
    # One forward/backward + Langevin update step mirroring ensembling script
    model.zero_grad()
    outputs = model(X)
    # Ensure outputs and Y shapes align
    if outputs.ndim > 1 and outputs.shape[-1] == 1:
        outputs = outputs.view(-1, 1)
    loss = custom_mse_loss(outputs, Y.unsqueeze(-1))
    loss.backward()

    # Langevin noise scale derived from t and lr (approximate)
    noise_scale = math.sqrt(2 * lr * t)

    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if param.grad is None:
                continue
            # resize noise buffer and sample
            noise_buffer.resize_(param.shape).normal_(0, noise_scale, generator=langevin_gen)
            param.add_(noise_buffer)
            # weight decay: map by index into provided vector (wrap-around if needed)
            wdec = float(weight_decay_vec[i % len(weight_decay_vec)])
            param.add_(param.data, alpha=-wdec * lr)
            param.add_(param.grad, alpha=-lr)


class LazyHeDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that generates He1 + eps*He3 targets on the fly.

    This avoids allocating the full (P, d) matrix in memory by producing
    samples in chunks. It is deterministic when seeded via `seed`.
    """

    def __init__(self, P, d, device='cpu', seed=613, chunk_size=16384, eps=0.03):
        super().__init__()
        self.P = int(P)
        self.d = int(d)
        self.device = device
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)

    def __iter__(self):
        # Create a local RNG so multiple workers / iterations are deterministic
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        produced = 0
        while produced < self.P:
            this = min(self.chunk_size, self.P - produced)
            # generate a chunk of inputs
            Xc = torch.randn((this, self.d), dtype=torch.float32, device=self.device, generator=gen)
            z = Xc[:, 0]
            He1 = z
            He3 = z ** 3 - 3.0 * z
            Yc = (He1 + self.eps * He3)
            # yield row-wise tuples
            for i in range(this):
                yield Xc[i], Yc[i]
            produced += this


def single_epoch_langevin_dataloader(model, dataloader, lr, t, weight_decay_vec, langevin_gen, noise_buffer, device):
    """Run a single epoch by iterating over `dataloader`, accumulating gradients.

    Gradients from each batch are accumulated (no optimizer step per batch).
    After processing all batches we apply the same Langevin update as in the
    full-dataset path.
    """
    model.zero_grad()
    # accumulate gradients over batches (loss is a summed loss over samples)
    for batch in dataloader:
        Xb, Yb = batch
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        outputs = model(Xb)
        if outputs.ndim > 1 and outputs.shape[-1] == 1:
            outputs = outputs.view(-1, 1)
        loss = custom_mse_loss(outputs, Yb.unsqueeze(-1))
        loss.backward()

    # Langevin noise scale derived from t and lr (approximate)
    noise_scale = math.sqrt(2 * lr * t)

    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if param.grad is None:
                continue
            noise_buffer.resize_(param.shape).normal_(0, noise_scale, generator=langevin_gen)
            param.add_(noise_buffer)
            wdec = float(weight_decay_vec[i % len(weight_decay_vec)])
            param.add_(param.data, alpha=-wdec * lr)
            param.add_(param.grad, alpha=-lr)


def run_benchmark(d_min=10, d_max=500, d_points=12, ens=1, repeats=3, use_cuda=False, out_dir=None, batch_size=16384, p_scalings=None):
    device = torch.device('cuda:1' if use_cuda and torch.cuda.is_available() else 'cpu')

    ds = np.unique(np.round(np.logspace(math.log10(d_min), math.log10(d_max), num=d_points))).astype(int)

    # Default two scalings: P = d**2 and P = 10*d
    if p_scalings is None:
        p_scalings = [
            ("d^2", lambda d: int(d ** 1.5)),
            ("10d", lambda d: int(5 * d)),
        ]

    # If outputs already exist on disk for the requested d range and scalings,
    # load them and skip the expensive benchmark runs.
    out_dir = out_dir or os.path.join(PROJECT_ROOT, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    expected_npzs = [os.path.join(out_dir, f'diagnostics_langevin_runtime_{label}_d{d_min}_to_d{d_max}.npz') for label, _ in p_scalings]
    if all([os.path.exists(p) for p in expected_npzs]):
        print('Found existing diagnostics for requested settings — loading and skipping benchmark.')
        all_results = {}
        for path, (label, _) in zip(expected_npzs, p_scalings):
            try:
                data = np.load(path)
                # Reconstruct a minimal results dict similar to what the benchmark produces
                d_arr = data['d'] if 'd' in data else np.array([])
                P_arr = data['P'] if 'P' in data else np.array([])
                N_arr = data['N'] if 'N' in data else np.array([])
                times_mean = data['times_mean'] if 'times_mean' in data else np.array([])
                times_std = data['times_std'] if 'times_std' in data else np.array([])

                res = {
                    'd': list(d_arr.tolist()),
                    'P': list(P_arr.tolist()),
                    'N': list(N_arr.tolist()),
                    'times_mean': list(times_mean.tolist()),
                    'times_std': list(times_std.tolist()),
                    'raw_times': [],
                    'learn1': [float('nan')] * len(d_arr),
                    'learn3': [float('nan')] * len(d_arr),
                }
                all_results[label] = res
                print(f'  loaded {path}')
            except Exception as e:
                print(f'  failed to load {path}:', e)
                # if any load fails, fall back to running the full benchmark
                all_results = None
                break
        # If all files loaded successfully, return the loaded results and skip heavy computation
        if isinstance(all_results, dict) and len(all_results) == len(p_scalings):
            print('All diagnostics loaded from disk; skipping benchmark.')
            return all_results
    else:
        all_results = {}

    all_results = {}

    for scale_label, scale_fn in p_scalings:
        results = {'d': [], 'P': [], 'N': [], 'times_mean': [], 'times_std': [], 'raw_times': [], 'learn1': [], 'learn3': []}
        print(f'Running benchmark for P scaling: {scale_label}')

        for d in ds:
            P = int(scale_fn(d))
            N = int(4 * d)
            print(f'Benchmarking d={d} -> P={P}, N={N} (ens={ens}) ...')

            # Create model: signature FCN3NetworkEnsembleErf(input_size, hidden_size, hidden_size, P, ens=..., device=...)
            model = FCN3NetworkEnsembleErf(d, N, N, P, ens=ens, device=device)
            model.to(device)

            # Data generation (He1 + eps * He3, w=e1)
            eps = 0.03
            use_dataloader = False
            dataloader = None
            try:
                torch.manual_seed(613)
                X = torch.randn((P, d), dtype=torch.float32, device=device)
                z = X[:, 0]
                He1 = z
                He3 = z ** 3 - 3.0 * z
                Y = (He1 + eps * He3)
            except (RuntimeError, MemoryError) as e:
                # Fall back to chunked generation via IterableDataset + DataLoader
                print('Could not allocate full dataset in memory, falling back to chunked DataLoader:', str(e))
                use_dataloader = True
                dataset = LazyHeDataset(P, d, device=device, seed=613, chunk_size=batch_size, eps=eps)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
                X = None
                Y = None

            # Langevin parameters similar to ensembling script defaults
            kappa = 1.0
            chi = 1.0
            t0 = 2 * kappa
            t = t0 / chi
            # weight_decay vector used in the original script was [d, N, N*chi] * t
            weight_decay_vec = torch.tensor([d, N, N * chi], dtype=torch.float32, device=device) * t
            print(device)
            langevin_gen = torch.Generator(device=device)
            noise_buffer = torch.empty(1, device=device, dtype=torch.float32)

            # Choose a small lr for stable step; only timing matters
            lr = 1e-3

            times = []
            # Warm-up iteration (JITs / caches)
            try:
                if not use_dataloader:
                    single_epoch_langevin(model, X, Y, lr, t, weight_decay_vec, langevin_gen, noise_buffer)
                else:
                    single_epoch_langevin_dataloader(model, dataloader, lr, t, weight_decay_vec, langevin_gen, noise_buffer, device)
            except Exception:
                pass

            for r in range(repeats):
                torch.manual_seed(1000 + r)
                start = time.perf_counter()
                if not use_dataloader:
                    single_epoch_langevin(model, X, Y, lr, t, weight_decay_vec, langevin_gen, noise_buffer)
                else:
                    single_epoch_langevin_dataloader(model, dataloader, lr, t, weight_decay_vec, langevin_gen, noise_buffer, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f'  repeat {r+1}/{repeats}: {elapsed:.4f}s')

            times = np.array(times)
            results['d'].append(d)
            results['P'].append(P)
            results['N'].append(N)
            results['times_mean'].append(float(times.mean()))
            results['times_std'].append(float(times.std()))
            results['raw_times'].append(times)

            # Compute learnability via FCS.jl if available (eps=0.03, chi=N)
            learn1 = float('nan')
            learn3 = float('nan')
            if JULIA_AVAILABLE and _FCS is not None:
                try:
                    eps_val = 0.03
                    chi_val = float(N)

                    # Use nlsolve-based flow from julia FCS (matches provided example)
                    d_f = float(d)
                    # initial guess for target-solution (T)
                    i0 = [1 / d_f ** 0.5, 1 / d_f ** (3 / 2), 1 / d_f ** 0.5, 1 / d_f ** (3 / 2)]
                    i0_j = juliacall.convert(jl.Vector[jl.Float64], i0)

                    χ = chi_val
                    n = N
                    ϵ = eps_val
                    π = np.pi
                    δ = 1.0
                    P_f = float(P)
                    lr = 1e-6
                    Tf = 60_000
                    kappa = 1.0

                    lT = jl.FCS.nlsolve_solver(
                        i0_j,
                        chi=χ, d=d_f, kappa=kappa, delta=δ,
                        epsilon=ϵ, n=n, b=4 / (3 * π),
                        P=P_f, lr=lr, max_iter=Tf, verbose=False, anneal=True
                    )

                    # initial guess for prior-solution (P)
                    i0b = [1 / d_f, 1 / d_f ** 3, 1 / d_f, 1 / d_f ** 3]
                    i0b_j = juliacall.convert(jl.Vector[jl.Float64], i0b)
                    lP = jl.FCS.nlsolve_solver(
                        i0b_j,
                        chi=χ, d=d_f, kappa=1.0, delta=1.0,
                        epsilon=ϵ, n=n, b=4 / (3 * π),
                        P=P_f, lr=lr, max_iter=Tf, verbose=False, anneal=True
                    )

                    # Try to compute readout/kernel eigenvalue predictions via Julia FCS
                    try:
                        lK_T = jl.FCS.compute_lK_ratio(lT, P_f, n,  χ, d_f, δ , kappa, ϵ, 4/(3*π))
                        lK_T_py = [float(x) for x in lK_T]
                        if len(lK_T_py) >= 1:
                            learn1 = float(lK_T_py[0])
                        if len(lK_T_py) >= 2:
                            learn3 = float(lK_T_py[1])
                    except Exception as _e:
                        print(f"Warning: compute_lK (T) failed: {_e}")

                    try:
                        lK_P = jl.FCS.compute_lK_ratio(lP, P_f, n,  χ, d_f, 0.0 , kappa, ϵ, 4/(3*π))
                        lK_P_py = [float(x) for x in lK_P]
                        # If compute_lK for P provides values and we don't have learnability from T,
                        # use those as a fallback (prefer target-based values when available).
                        if np.isnan(learn1) and len(lK_P_py) >= 1:
                            learn1 = float(lK_P_py[0])
                        if np.isnan(learn3) and len(lK_P_py) >= 2:
                            learn3 = float(lK_P_py[1])
                    except Exception as _e:
                        print(f"Warning: compute_lK (P) failed: {_e}")

                except Exception as e:
                    print(f'FCS.jl nlsolve-based computation failed for d={d}, P={P}:', e)

            results['learn1'].append(learn1)
            results['learn3'].append(learn3)

            # Free memory
            del model
            if not use_dataloader:
                del X, Y, He1, He3
            else:
                del dataloader, dataset
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # store per-scaling results
        all_results[scale_label] = results

    # Save per-scaling results and prepare combined plotting
    out_dir = out_dir or os.path.join(PROJECT_ROOT, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    combined = {}
    for scale_label, res in all_results.items():
        d_arr = np.array(res['d'])
        P_arr = np.array(res['P'])
        N_arr = np.array(res['N'])
        means = np.array(res['times_mean'])
        stds = np.array(res['times_std'])
        learns1 = np.array(res.get('learn1', [np.nan] * len(d_arr)))
        learns3 = np.array(res.get('learn3', [np.nan] * len(d_arr)))
        epochs_mean = 1.0 / means
        epochs_std = stds / (means ** 2)

        out_npz = os.path.join(out_dir, f'diagnostics_langevin_runtime_{scale_label}_d{d_min}_to_d{d_max}.npz')
        np.savez_compressed(out_npz,
                            d=d_arr,
                            P=P_arr,
                            N=N_arr,
                            times_mean=means,
                            times_std=stds,
                            epochs_mean=epochs_mean,
                            epochs_std=epochs_std)
        print('Saved results to', out_npz)

        combined[scale_label] = dict(d=d_arr, times_mean=means, times_std=stds, epochs_mean=epochs_mean, epochs_std=epochs_std)
        # include learnabilities in combined
        combined[scale_label]['learn1'] = learns1
        combined[scale_label]['learn3'] = learns3

    # Combined plotting: epoch time and epochs/sec with both scalings on same axes
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Epoch time (log-log) combined
    fig, ax = plt.subplots(figsize=(8, 6))
    for (scale_label, vals), c in zip(combined.items(), colors):
        ax.errorbar(vals['d'], vals['times_mean'], yerr=vals['times_std'], fmt='o-', capsize=4, label=scale_label, color=c)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('d (hidden dimension)')
    ax.set_ylabel('Epoch time (s)')
    ax.set_title(f'Langevin epoch runtime comparison (ens={ens})')
    ax.legend(title='P scaling')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    out_png = os.path.join(out_dir, f'diagnostics_langevin_runtime_compare_d{d_min}_to_d{d_max}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print('Saved combined runtime plot to', out_png)

    # Epochs-per-second combined
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for (scale_label, vals), c in zip(combined.items(), colors):
        ax2.errorbar(vals['d'], vals['epochs_mean'], yerr=vals['epochs_std'], fmt='o-', capsize=4, label=scale_label, color=c)
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_xlabel('d (hidden dimension)')
    ax2.set_ylabel('Epochs / second')
    ax2.set_title(f'Epochs per second comparison (ens={ens})')
    ax2.legend(title='P scaling')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    out_png_epochs = os.path.join(out_dir, f'diagnostics_langevin_runtime_epochs_compare_d{d_min}_to_d{d_max}.png')
    plt.tight_layout()
    plt.savefig(out_png_epochs, dpi=200)
    plt.close(fig2)
    print('Saved combined epochs/sec plot to', out_png_epochs)

    # Combined learnability plots (Target H1 and H3)
    fig3, (axl1, axl3) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for (scale_label, vals), c in zip(combined.items(), colors):
        axl1.plot(vals['d'], vals.get('learn1', np.full_like(vals['d'], np.nan)), 'o-', label=scale_label, color=c)
        axl3.plot(vals['d'], vals.get('learn3', np.full_like(vals['d'], np.nan)), 's-', label=scale_label, color=c)

    for ax, ylabel in [(axl1, 'Learnability (H1)'), (axl3, 'Learnability (H3)')]:
        # ax.set_xscale('log')
        ax.set_xlabel('d (hidden dimension)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title='P scaling')
        ax.grid(True, which='both', ls='--', alpha=0.3)

    out_png_learn = os.path.join(out_dir, f'diagnostics_langevin_runtime_learnability_compare_d{d_min}_to_d{d_max}.png')
    plt.tight_layout()
    plt.savefig(out_png_learn, dpi=200)
    plt.close(fig3)
    print('Saved combined learnability plot to', out_png_learn)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnostics: Langevin epoch runtime vs d (P=d**1.5, N=5*d)')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeat timings per d')
    parser.add_argument('--d-min', type=int, default=10, help='Minimum d (log scale)')
    parser.add_argument('--d-max', type=int, default=500, help='Maximum d (log scale)')
    parser.add_argument('--d-points', type=int, default=12, help='Number of log-spaced d values')
    parser.add_argument('--ens', type=int, default=1, help='Ensemble size passed to model')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for plots and results')
    parser.add_argument('--batch-size', type=int, default=16384, help='Batch size for chunked processing if full dataset does not fit in memory')
    args = parser.parse_args()

    run_benchmark(d_min=args.d_min, d_max=args.d_max, d_points=args.d_points,
                  ens=args.ens, repeats=args.repeats, use_cuda=args.cuda, out_dir=args.out_dir,
                  batch_size=args.batch_size)
