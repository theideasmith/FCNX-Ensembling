#!/usr/bin/env python3
"""
Analyze d-sweep experiments for FCN3 Linear networks: generate large datasets, 
compute QB decompositions, compute sandwich (left) eigenvalues for He1, 
and plot results.

This script creates Experiment objects for a sweep of `d` values (matching
`script/d_sweep_fcn3_linear.sh` defaults), generates a large dataset (p_large=5000),
computes a randomized QB for the H-kernel, computes sandwich eigenvalues for Y1,
and saves plots to `script/analysis_outputs/`.

Notes:
- Runs on CUDA by default. Set `USE_CUDA=False` below for CPU.
- Linear networks don't have He3 component, only He1 (linear target).
"""

import os
import sys
import math

# Ensure repository `lib/` is on the import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from Experiment import Experiment

import torch
import matplotlib.pyplot as plt
import numpy as np

# Configuration
USE_CUDA = True
# Sweep parameters — linear network defaults: d = 10, 20, 30, 40, 50, 60
DEFAULT_D_VALUES = [10, 20, 30, 40, 50, 60]

# For model initialization
N_FACTOR = 4  # N = 4 * d
P_FACTOR = 6  # P = 6 * d
ENS = 3

# Large dataset size
P_LARGE = 5000

# QB parameters (default oversampling)
P_OVERSAMPLE = 50

# Experiment directory for linear networks
EXPERIMENT_DIR = '/home/akiva/exp/'
EXPERIMENT_SUBDIR = 'd_sweep_linear_network'

# OUT_DIR is created per-run
OUT_DIR = None


def device_for(use_cuda: bool):
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def run_analysis(d_values=None, use_cuda=USE_CUDA, use_cache: bool = False):
    device = device_for(use_cuda)
    
    records = []

    # Get experiment files from linear network directory
    exp_path = os.path.join(EXPERIMENT_DIR, EXPERIMENT_SUBDIR)
    if os.path.exists(exp_path):
        EXPERIMENT_FILES = [f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))]
    else:
        print(f"Experiment directory {exp_path} not found!")
        EXPERIMENT_FILES = []

    if EXPERIMENT_FILES:
        files_to_process = [os.path.join(exp_path, f) for f in EXPERIMENT_FILES]
    else:
        if d_values is None:
            d_values = DEFAULT_D_VALUES
        files_to_process = None

    iterator = files_to_process if files_to_process is not None else d_values

    if iterator is None or len(iterator) == 0:
        print("No experiments to process!")
        return records

    # Determine d range for naming output folder
    try:
        if files_to_process is not None:
            ds = []
            import re
            for f in EXPERIMENT_FILES:
                m = re.search(r"_D_(\d+)_", f)
                if m:
                    ds.append(int(m.group(1)))
            if not ds:
                ds = DEFAULT_D_VALUES
        else:
            ds = list(d_values)
        d_min = min(ds)
        d_max = max(ds)
    except Exception:
        d_min = 10
        d_max = 60

    # Create project-level plots output directory
    PROJECT_PLOTS = os.path.join(PROJECT_ROOT, 'plots')
    out_dir = os.path.join(PROJECT_PLOTS, f'd_sweep_linear_d{d_min}_to_d{d_max}')
    os.makedirs(out_dir, exist_ok=True)

    for item in iterator:
        # item may be a filename (str) or a d (int)
        if isinstance(item, str):
            fname = item
            base = os.path.basename(fname)
            # parse d, N, P from filename
            import re
            m_d = re.search(r"_D_(\d+)_", base)
            m_N = re.search(r"_N_(\d+)_", base)
            m_P = re.search(r"_P_(\d+)_", base)
            if m_d:
                d = int(m_d.group(1))
            else:
                raise RuntimeError(f"Could not parse D from filename {base}")
            N = int(m_N.group(1)) if m_N else N_FACTOR * d
            P_file = int(m_P.group(1)) if m_P else P_FACTOR * d
            chi = N

            print(f"\n--- Processing experiment file {base} (d={d}, N={N}, P_file={P_file}) ---")
            exp = Experiment(file=fname, N=N, d=d, chi=chi, P=P_file, ens=ENS)
        else:
            d = int(item)
            N = N_FACTOR * d
            chi = N
            P_file = P_FACTOR * d
            print(f"\n--- Processing d={d} ---")
            exp = Experiment(file='', N=N, d=d, chi=chi, P=P_file, ens=ENS)

        # Ensure the Experiment and its model operate on the correct device
        exp.device = device
        try:
            setattr(exp.model, 'device', device)
            exp.model.device = device
        except Exception:
            pass
        try:
            exp.model.to(device)
            exp.model.device = device
        except Exception:
            pass

        # Generate large dataset (linear networks only have Y1)
        try:
            Xinf, Y = exp.large_dataset(p_large=P_LARGE)
        except TypeError:
            Xinf, Y = exp.large_dataset(p_large=P_LARGE)

        if isinstance(Y, tuple) and len(Y) == 2:
            Y1_inf = Y[0]
        else:
            Y1_inf = Y

        # Ensure tensors are on device
        Xinf = Xinf.to(device).to(torch.float32)
        Y1_inf = Y1_inf.to(device).to(torch.float32)

        # Compute QB decomposition of H (randomized)
        k = 3000
        npz_path = os.path.join(out_dir, f'd_{d}_sandwich_eigs.npz')
        
        compute_now = True
        sandwiches1_np = None

        if use_cache and os.path.exists(npz_path):
            try:
                loaded = np.load(npz_path, allow_pickle=True)
                sandwiches1_np = loaded['sandwich_H1']
                compute_now = False
                print(f'Loaded precomputed eigenvalues from {npz_path}')
            except Exception as e:
                print('Failed to load existing npz, will recompute:', e)
        
        if compute_now:
            print(f'Computing randomized QB (H_random_QB) with k={k}')
            try:
                Q, Z = exp.model.H_random_QB(Xinf, k=int(k), p=int(P_OVERSAMPLE))
            except Exception as e:
                print('H_random_QB failed:', e)
                continue

            # Build eigen-decomposition
            with torch.no_grad():
                Z_t = Z.t()
                try:
                    Ut, _S, V = torch.linalg.svd(Z_t)
                except Exception:
                    Ut, _S, V = torch.svd(Z_t)

                m = Z.shape[1]
                n = Z.shape[0]
                k_sv = min(m, n)

                # Map left singular vectors back to sample space
                U_mat = torch.matmul(Q, Ut)

                # Normalize Y column
                Y1_norm = Y1_inf / (torch.norm(Y1_inf, dim=0) + 1e-30)

                Sdiag = torch.diag(_S)

                M1 = torch.matmul(torch.matmul(Y1_norm.t(), U_mat), 
                                 torch.matmul(Sdiag, torch.matmul(U_mat.t(), Y1_norm)))
                sandwiches1 = torch.diagonal(M1) / float(Xinf.shape[0])
                
                sandwiches1_np = sandwiches1.detach().cpu().numpy()

            # Save computed eigenvalues
            try:
                np.savez_compressed(npz_path, sandwich_H1=sandwiches1_np)
                print('Saved eigenvalues to', npz_path)
            except Exception as e:
                print('Could not save eigenvalues to npz:', e)

        # Save a plot of sandwich eigenvalues
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

            sorted_vals1 = np.sort(sandwiches1_np)[::-1]
            ax.bar(np.arange(len(sorted_vals1)), sorted_vals1, color='C0', alpha=0.7)
            ax.set_title(f'd={d} He1 sandwich eigenvalues (Linear Network)')
            ax.set_yscale('log')
            ax.set_xlabel('mode index')
            ax.set_ylabel('value')
            
            plt.tight_layout()
            outfn = os.path.join(out_dir, f'd_{d}_sandwich_eigenvalues.png')
            plt.savefig(outfn, dpi=300)
            plt.close(fig)
            print(f'Saved plot to {outfn}')

        except Exception as e:
            print('Could not generate sandwich eigenvalues plot:', e)

        records.append({
            'd': d,
            'N': N,
            'P': P_file,
            'sandwich_H1': sandwiches1_np,
        })

    return records


def aggregate_and_plot_scaling(records, out_dir):
    """Aggregate per-experiment sandwich eigenvalues and produce a log-log
    scaling plot with best-fit exponent and R^2 for He1.
    """
    if not records:
        print('No records to aggregate for scaling plot.')
        return

    ds = []
    h1_leads = []
    h1_rem_mean = []
    h1_rem_std = []

    for rec in records:
        try:
            d = int(rec['d'])
            s1 = rec.get('sandwich_H1', None)
            if s1 is None:
                continue
            
            s1_sorted = np.sort(s1)[::-1]
            h1_lead = float(s1_sorted[0]) if s1_sorted.size > 0 else np.nan
            
            # remaining modes statistics
            if s1_sorted.size > 1:
                rem1 = s1_sorted[1:]
                h1_rem_mean_val = float(np.mean(rem1))
                h1_rem_std_val = float(np.std(rem1))
            else:
                h1_rem_mean_val = np.nan
                h1_rem_std_val = np.nan
            
            ds.append(d)
            h1_leads.append(h1_lead)
            h1_rem_mean.append(h1_rem_mean_val)
            h1_rem_std.append(h1_rem_std_val)
        except Exception:
            continue

    if len(ds) < 2:
        print('Not enough data points to fit scaling law (need >=2).')
        return

    # Sort by d
    order = np.argsort(ds)
    ds_arr = np.array(ds)[order]
    h1_arr = np.array(h1_leads)[order]

    # Only keep positive entries
    mask1 = (h1_arr > 0) & np.isfinite(h1_arr)
    
    rem1_arr = np.array(h1_rem_mean)[order]
    rem1_std_arr = np.array(h1_rem_std)[order]
    mask1_rem = (rem1_arr > 0) & np.isfinite(rem1_arr)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Fit H1 leading
    if np.sum(mask1) >= 2:
        x1 = np.log(ds_arr[mask1])
        y1 = np.log(h1_arr[mask1])
        p1 = np.polyfit(x1, y1, 1)
        y1_pred = np.polyval(p1, x1)
        ss_res1 = np.sum((y1 - y1_pred) ** 2)
        ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
        r21 = 1.0 - ss_res1 / ss_tot1 if ss_tot1 != 0 else np.nan
        slope1 = float(p1[0])
        
        ax.scatter(ds_arr[mask1], h1_arr[mask1], marker='o', color='C0', label='H1 Leading')
        xs = np.linspace(ds_arr[mask1].min(), ds_arr[mask1].max(), 200)
        ax.plot(xs, np.exp(np.polyval(p1, np.log(xs))), '--', color='C0', alpha=0.7,
                label=f'H1 fit: slope={slope1:.3f}, $R^2$={r21:.3f}')
    else:
        slope1 = np.nan

    # Fit H1 remaining
    if np.sum(mask1_rem) >= 2:
        x1r = np.log(ds_arr[mask1_rem])
        y1r = np.log(rem1_arr[mask1_rem])
        p1r = np.polyfit(x1r, y1r, 1)
        slope1r = float(p1r[0])
        xs_r = np.linspace(ds_arr[mask1_rem].min(), ds_arr[mask1_rem].max(), 200)
        
        ax.scatter(ds_arr[mask1_rem], rem1_arr[mask1_rem], marker='o', 
                  color='C0', alpha=0.6, label='H1 rem mean')
        ax.plot(xs_r, np.exp(np.polyval(p1r, np.log(xs_r))), '-', color='C0', alpha=0.4,
                label=f'H1 rem fit: slope={slope1r:.3f}')
        
        try:
            std_vals = rem1_std_arr[mask1_rem]
            ax.fill_between(ds_arr[mask1_rem],
                           rem1_arr[mask1_rem] - std_vals,
                           rem1_arr[mask1_rem] + std_vals,
                           color='C0', alpha=0.15)
        except Exception:
            pass

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('d (hidden dimension)')
    ax.set_ylabel('leading sandwich eigenvalue')
    ax.set_title('Scaling of sandwich eigenvalues vs d (Linear Network, log-log)')
    ax.legend(loc='best')
    
    plt.tight_layout()
    outfn = os.path.join(out_dir, 'scaling_fit.png')
    try:
        plt.savefig(outfn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('Saved scaling fit plot to', outfn)
    except Exception as e:
        print('Could not save scaling plot:', e)

    # Save summary npz
    try:
        slope1r_val = locals().get('slope1r', np.nan)
        summary_npz = os.path.join(out_dir, 'scaling_summary.npz')
        np.savez_compressed(summary_npz,
                           ds=ds_arr,
                           h1=h1_arr,
                           h1_rem_mean=rem1_arr,
                           h1_rem_std=rem1_std_arr,
                           slope1=slope1,
                           slope1_rem=slope1r_val)
        print('Saved scaling summary to', summary_npz)
    except Exception as e:
        print('Could not save scaling summary:', e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze d-sweep experiments for linear networks')
    parser.add_argument('--d-values', type=int, nargs='+', default=None,
                       help='List of d values to analyze (default: 10 20 30 40 50 60)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--use-cache', action='store_true', 
                       help='Load precomputed caches instead of recomputing')
    args = parser.parse_args()

    d_vals = args.d_values if args.d_values is not None else DEFAULT_D_VALUES
    print('d values:', d_vals)
    recs = run_analysis(d_vals, use_cuda=args.cuda, use_cache=args.use_cache)
    
    # Determine out_dir
    try:
        if recs:
            ds = [rec['d'] for rec in recs]
            d_min = min(ds)
            d_max = max(ds)
        else:
            d_min = 10
            d_max = 60
    except Exception:
        d_min = 10
        d_max = 60

    PROJECT_PLOTS = os.path.join(PROJECT_ROOT, 'plots')
    out_dir = os.path.join(PROJECT_PLOTS, f'd_sweep_linear_d{d_min}_to_d{d_max}')

    # Aggregate and plot scaling fit
    try:
        aggregate_and_plot_scaling(recs, out_dir)
    except Exception as e:
        print('Failed to produce aggregate scaling plot:', e)

    print('\nDone. Saved outputs to project-level plots folder (plots/d_sweep_linear_*)')
