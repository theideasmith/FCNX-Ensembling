#!/usr/bin/env python3
"""
Analyze d-sweep experiments: generate large datasets, compute QB decompositions,
compute sandwich (left) eigenvalues for He1/He3, fetch MF predictions and plot
results.

This script creates Experiment objects for a sweep of `d` values (matching
`script/d_sweep_fcn3_P4d.py` defaults), generates a large dataset (p_large=4000),
computes a randomized QB for the H-kernel, computes sandwich eigenvalues for
Y1/Y3, attempts to obtain MF predictions via `eig_predictions`, and saves plots
to `script/analysis_outputs/`.

Notes:
- Runs on CPU by default to avoid CUDA/Julia precompilation issues. If you
  have CUDA and Julia available and prefer GPU, set `USE_CUDA=True` below.
- Julia-based MF predictions may fail if Julia/FCS.jl isn't configured; the
  script proceeds gracefully in that case.
"""

import os
import sys
import math


# Ensure repository `lib/` is on the import path so local modules (e.g. Experiment)
# can be imported when this script runs from the `script/` directory.
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
# Sweep parameters — mirror defaults from `d_sweep_fcn3_P4d.py`
DEFAULT_START = 20
DEFAULT_STOP = 300
DEFAULT_STEP = 25

# For model initialization
N_FACTOR = 4  # n = n_factor * d
ENS = 5

# Large dataset size
P_LARGE = 4000

# QB parameters (default oversampling)
P_OVERSAMPLE = 50

# If you want to analyze a fixed list of experiment folders in
# `/home/akiva/exp/fcn3erf`, list them here (the script will parse D/N/P from
# the filename). When provided, the script will iterate these files instead
# of performing a simple d-sweep.
EXPERIMENT_DIR = '/home/akiva/exp/fcn3erf'
EXPERIMENT_FILES = [
    'erf_cubic_eps_0.03_P_225_D_45_N_180_epochs_20000000_lrA_4.44e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_475_D_95_N_380_epochs_20000000_lrA_2.11e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_100_D_20_N_80_epochs_20000000_lrA_1.00e-08_time_20251201_220935',
    'erf_cubic_eps_0.03_P_350_D_70_N_280_epochs_20000000_lrA_2.86e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_975_D_195_N_780_epochs_20000000_lrA_1.03e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_850_D_170_N_680_epochs_20000000_lrA_1.18e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_725_D_145_N_580_epochs_20000000_lrA_1.38e-09_time_20251201_220935',
    'erf_cubic_eps_0.03_P_600_D_120_N_480_epochs_20000000_lrA_1.67e-09_time_20251201_220935',
]

# OUT_DIR is created per-run inside `run_analysis` so the folder can include
# the d-range in its name (project-level `plots/d_sweep_d<min>_to_d<max>`).
OUT_DIR = None


def device_for(use_cuda: bool):
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def run_analysis(d_values=None, use_cuda=USE_CUDA, use_cache: bool = False):
    # Use CPU for all tensors/operations to guarantee consistency
    device = torch.device('cuda')
    if use_cuda:
        print('Note: --cuda was passed but this analysis forces CPU for stability.')

    # If explicit EXPERIMENT_FILES are provided, iterate those; otherwise do a d-sweep
    records = []

    if EXPERIMENT_FILES:
        files_to_process = [os.path.join(EXPERIMENT_DIR, f) for f in EXPERIMENT_FILES]
    else:
        if d_values is None:
            d_values = list(range(DEFAULT_START, DEFAULT_STOP + 1, DEFAULT_STEP))
        files_to_process = None

    iterator = files_to_process if files_to_process is not None else d_values

    if iterator is None:
        return records

    # Determine d range for naming output folder
    try:
        if files_to_process is not None:
            # parse all d values from EXPERIMENT_FILES
            ds = []
            import re
            for f in EXPERIMENT_FILES:
                m = re.search(r"_D_(\d+)_", f)
                if m:
                    ds.append(int(m.group(1)))
            if not ds:
                ds = [DEFAULT_START, DEFAULT_STOP]
        else:
            ds = list(d_values)
        d_min = min(ds)
        d_max = max(ds)
    except Exception:
        d_min = DEFAULT_START
        d_max = DEFAULT_STOP

    # Create project-level plots output directory
    PROJECT_PLOTS = os.path.join(PROJECT_ROOT, 'plots')
    out_dir = os.path.join(PROJECT_PLOTS, f'd_sweep_d{d_min}_to_d{d_max}')
    os.makedirs(out_dir, exist_ok=True)

    for item in iterator:
        # item may be a filename (str) from EXPERIMENT_FILES or a d (int)
        if isinstance(item, str):
            fname = item
            base = os.path.basename(fname)
            # parse d, N, P from filename if possible
            import re
            m_d = re.search(r"_D_(\d+)_", base)
            m_N = re.search(r"_N_(\d+)_", base)
            m_P = re.search(r"_P_(\d+)_", base)
            if m_d:
                d = int(m_d.group(1))
            else:
                raise RuntimeError(f"Could not parse D from filename {base}")
            N = int(m_N.group(1)) if m_N else N_FACTOR * d
            P_file = int(m_P.group(1)) if m_P else 5 * d
            chi = N

            print(f"\n--- Processing experiment file {base} (d={d}, N={N}, P_file={P_file}) ---")
            exp = Experiment(file=fname, N=N, d=d, chi=chi, P=P_file, ens=ENS)
        else:
            d = int(item)
            N = N_FACTOR * d
            chi = N
            P_file = 5 * d
            print(f"\n--- Processing d={d} ---")
            exp = Experiment(file='', N=N, d=d, chi=chi, P=P_file, ens=ENS, eps=0.03)

        # Ensure the Experiment and its model operate on the CPU
        exp.device = device
        try:
            # Set model.device attribute and move parameters/buffers to cpu
            setattr(exp.model, 'device', device)
            exp.model.device = device

        except Exception:
            pass
        try:
            exp.model.to(device)
            exp.model.device = device
        except Exception:
            pass

        # Generate large dataset. `large_dataset` may return either
        # (Xinf, (Y1, Y3)) or (Xinf, Y1, Y3) depending on `flat` flag.
        res = None
        try:
            Xinf, Y = exp.large_dataset(p_large=P_LARGE)
        except TypeError:
            Xinf, Y = exp.large_dataset(p_large=P_LARGE)

        if isinstance(Y, tuple) and len(Y) == 2:
            Y1_inf, Y3_inf =  Y[0], Y[1]
        else:
            raise RuntimeError('Unexpected return from large_dataset')

        # Ensure tensors are on device
        Xinf = Xinf.to(device).to(torch.float32)
        Y1_inf = Y1_inf.to(device).to(torch.float32)
        Y3_inf = Y3_inf.to(device).to(torch.float32)

        # Compute QB decomposition of H (randomized)
        # Set per-experiment rank parameter k; increased per request to 5 * d
        k = 5 * d
        npz_path = os.path.join(out_dir, f'd_{d}_sandwich_eigs.npz')
        # Default: recompute each experiment and then cache results.
        # Only load from cache when `use_cache` is True (CLI flag `--use-cache`).
        compute_now = True
        sandwiches1_np = None
        sandwiches3_np = None
        preds = None

        if use_cache and os.path.exists(npz_path):
            try:
                loaded = np.load(npz_path, allow_pickle=True)
                sandwiches1_np = loaded['sandwich_H1']
                sandwiches3_np = loaded['sandwich_H3']
                preds = loaded.get('preds', None)
                compute_now = False
                print(f'Loaded precomputed eigenvalues from {npz_path}')
            except Exception as e:
                print('Failed to load existing npz, will recompute:', e)

        if compute_now:
            print(f'Computing randomized QB (H_random_QB) with k={k} — this may take a while')
            try:
                Q, Z = exp.model.H_random_QB(Xinf, k=int(k), p=int(P_OVERSAMPLE))
            except Exception as e:
                print('H_random_QB failed:', e)
                continue

            # Build eigen-decomposition following the randomized-QB -> SVD -> sandwich recipe
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

                # Normalize Y columns
                Y1_norm = Y1_inf / (torch.norm(Y1_inf, dim=0) + 1e-30)
                Y3_norm = Y3_inf / (torch.norm(Y3_inf, dim=0) + 1e-30)

                Sdiag = torch.diag(_S)

                M1 = torch.matmul(torch.matmul(Y1_norm.t(), U_mat), torch.matmul(Sdiag, torch.matmul(U_mat.t(), Y1_norm)))
                sandwiches1 = torch.diagonal(M1) / float(Xinf.shape[0])

                M3 = torch.matmul(torch.matmul(Y3_norm.t(), U_mat), torch.matmul(Sdiag, torch.matmul(U_mat.t(), Y3_norm)))
                sandwiches3 = torch.diagonal(M3) / float(Xinf.shape[0])

                sandwiches1_np = sandwiches1.detach().cpu().numpy()
                sandwiches3_np = sandwiches3.detach().cpu().numpy()

            # Save computed eigenvalues for future runs
            try:
                np.savez_compressed(npz_path,
                                    sandwich_H1=sandwiches1_np,
                                    sandwich_H3=sandwiches3_np,
                                    preds=preds)
                print('Saved eigenvalues to', npz_path)
            except Exception as e:
                print('Could not save eigenvalues to npz:', e)

        # Attempt to compute MF predictions (may fail if Julia not configured)
        preds = None
        try:
            preds = exp.eig_predictions()
            print('Got MF predictions from Experiment.eig_predictions()')
        except Exception as e:
            print('eig_predictions failed (Julia/FCS might be missing):', e)

        # Save a plot combining predicted horizontal lines and sandwich eigenvalues
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # He1 plot
            ax = axes[0]
            sorted_vals1 = np.sort(sandwiches1_np)[::-1]
            ax.bar(np.arange(len(sorted_vals1)), sorted_vals1, color='C0', alpha=0.7)
            if preds is not None:
                # Predicted target/perp can be scalars
                for key, style, color in [('\u03BB_H1^T', '--', 'k'), ('\u03BB_H1^P', '-', 'r')]:
                    try:
                        if key == '\u03BB_H1^T':
                            val = float(preds.lH1T)
                        else:
                            val = float(preds.lH1P)
                        ax.axhline(y=val, linestyle=style, color=color, label=key)
                    except Exception:
                        pass
            ax.set_title(f'd={d} He1 sandwich eigenvalues')
            ax.set_yscale('log')

            # He3 plot
            ax = axes[1]
            sorted_vals3 = np.sort(sandwiches3_np)[::-1]
            ax.bar(np.arange(len(sorted_vals3)), sorted_vals3, color='C1', alpha=0.7)
            if preds is not None:
                for key, style, color in [('\u03BB_H3^T', '--', 'k'), ('\u03BB_H3^P', '-', 'r')]:
                    try:
                        if key == '\u03BB_H3^T':
                            val = float(preds.lH3T)
                        else:
                            val = float(preds.lH3P)
                        ax.axhline(y=val, linestyle=style, color=color, label=key)
                    except Exception:
                        pass
            ax.set_title(f'd={d} He3 sandwich eigenvalues')
            ax.set_yscale('log')

            for ax in axes:
                ax.set_xlabel('mode index')
                ax.set_ylabel('value')
                ax.legend()

            plt.tight_layout()
            outfn = os.path.join(out_dir, f'd_{d}_sandwich_eigenvalues.png')
            plt.savefig(outfn, dpi=300)
            plt.close(fig)
            print('Saved plot to', outfn)
        except Exception as e:
            print('Plotting failed:', e)

        records.append({
            'd': d,
            'N': N,
            'P': P_file,
            'sandwich_H1': sandwiches1_np,
            'sandwich_H3': sandwiches3_np,
            'predictions': preds,
        })

    return records


def aggregate_and_plot_scaling(records, out_dir):
    """Aggregate per-experiment sandwich eigenvalues and produce a log-log
    scaling plot with a best-fit exponent and R^2 for H1 and H3.
    """
    if not records:
        print('No records to aggregate for scaling plot.')
        return

    ds = []
    h1_leads = []
    h3_leads = []
    # statistics for remaining modes (mean, std)
    h1_rem_mean = []
    h1_rem_std = []
    h3_rem_mean = []
    h3_rem_std = []
    # prediction arrays (may be missing for some records)
    pred_h1T = []
    pred_h1P = []
    pred_h3T = []
    pred_h3P = []

    for rec in records:
        try:
            d = int(rec['d'])
            s1 = rec.get('sandwich_H1', None)
            s3 = rec.get('sandwich_H3', None)
            if s1 is None or s3 is None:
                continue
            # pick the leading (largest) sandwich eigenvalue as the statistic
            s1_sorted = np.sort(s1)[::-1]
            s3_sorted = np.sort(s3)[::-1]
            h1_lead = float(s1_sorted[0]) if s1_sorted.size > 0 else np.nan
            h3_lead = float(s3_sorted[0]) if s3_sorted.size > 0 else np.nan
            # remaining modes statistics (exclude leading)
            if s1_sorted.size > 1:
                rem1 = s1_sorted[1:]
                h1_rem_mean_val = float(np.mean(rem1))
                h1_rem_std_val = float(np.std(rem1))
            else:
                h1_rem_mean_val = np.nan
                h1_rem_std_val = np.nan

            if s3_sorted.size > 1:
                rem3 = s3_sorted[1:]
                h3_rem_mean_val = float(np.mean(rem3))
                h3_rem_std_val = float(np.std(rem3))
            else:
                h3_rem_mean_val = np.nan
                h3_rem_std_val = np.nan
            ds.append(d)
            h1_leads.append(h1_lead)
            h3_leads.append(h3_lead)
            h1_rem_mean.append(h1_rem_mean_val)
            h1_rem_std.append(h1_rem_std_val)
            h3_rem_mean.append(h3_rem_mean_val)
            h3_rem_std.append(h3_rem_std_val)
            # extract predictions if available
            p = rec.get('predictions', None)
            if p is None:
                pred_h1T.append(np.nan)
                pred_h1P.append(np.nan)
                pred_h3T.append(np.nan)
                pred_h3P.append(np.nan)
            else:
                # Try multiple attribute/key patterns
                try:
                    h1T = float(p.lH1T)
                except Exception:
                    h1T = p.get('lH1T', np.nan) if isinstance(p, dict) else np.nan
                try:
                    h1P = float(p.lH1P)
                except Exception:
                    h1P = p.get('lH1P', np.nan) if isinstance(p, dict) else np.nan
                try:
                    h3T = float(p.lH3T)
                except Exception:
                    h3T = p.get('lH3T', np.nan) if isinstance(p, dict) else np.nan
                try:
                    h3P = float(p.lH3P)
                except Exception:
                    h3P = p.get('lH3P', np.nan) if isinstance(p, dict) else np.nan

                pred_h1T.append(h1T)
                pred_h1P.append(h1P)
                pred_h3T.append(h3T)
                pred_h3P.append(h3P)
        except Exception:
            continue

    if len(ds) < 2:
        print('Not enough data points to fit scaling law (need >=2).')
        return

    # Sort by d
    order = np.argsort(ds)
    ds_arr = np.array(ds)[order]
    h1_arr = np.array(h1_leads)[order]
    h3_arr = np.array(h3_leads)[order]

    # Only keep positive entries (log domain)
    mask1 = (h1_arr > 0) & np.isfinite(h1_arr)
    mask3 = (h3_arr > 0) & np.isfinite(h3_arr)
    # remaining means masks
    rem1_arr = np.array(h1_rem_mean)[order]
    rem1_std_arr = np.array(h1_rem_std)[order]
    rem3_arr = np.array(h3_rem_mean)[order]
    rem3_std_arr = np.array(h3_rem_std)[order]
    mask1_rem = (rem1_arr > 0) & np.isfinite(rem1_arr)
    mask3_rem = (rem3_arr > 0) & np.isfinite(rem3_arr)

    # predictions arrays
    pred_h1T_arr = np.array(pred_h1T)[order]
    pred_h1P_arr = np.array(pred_h1P)[order]
    pred_h3T_arr = np.array(pred_h3T)[order]
    pred_h3P_arr = np.array(pred_h3P)[order]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Fit H1
    if np.sum(mask1) >= 2:
        x1 = np.log(ds_arr[mask1])
        y1 = np.log(h1_arr[mask1])
        p1 = np.polyfit(x1, y1, 1)
        y1_pred = np.polyval(p1, x1)
        ss_res1 = np.sum((y1 - y1_pred) ** 2)
        ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
        r21 = 1.0 - ss_res1 / ss_tot1 if ss_tot1 != 0 else np.nan
        slope1 = float(p1[0])
        ax.plot(ds_arr[mask1], h1_arr[mask1], 'o-', color='C0', label='H1 Leading')
        # plot fit line (exponentiated)
        xs = np.linspace(ds_arr[mask1].min(), ds_arr[mask1].max(), 200)
        ax.plot(xs, np.exp(np.polyval(p1, np.log(xs))), '--', color='C0', alpha=0.7,
                label=f'H1 fit: slope={slope1:.3f}, $R^2$={r21:.3f}')
    else:
        slope1 = np.nan

    # Fit H3 (leading)
    if np.sum(mask3) >= 2:
        x3 = np.log(ds_arr[mask3])
        y3 = np.log(h3_arr[mask3])
        p3 = np.polyfit(x3, y3, 1)
        y3_pred = np.polyval(p3, x3)
        ss_res3 = np.sum((y3 - y3_pred) ** 2)
        ss_tot3 = np.sum((y3 - np.mean(y3)) ** 2)
        r23 = 1.0 - ss_res3 / ss_tot3 if ss_tot3 != 0 else np.nan
        slope3 = float(p3[0])
        ax.plot(ds_arr[mask3], h3_arr[mask3], 's-', color='C1', label='H3 Leading')
        xs = np.linspace(ds_arr[mask3].min(), ds_arr[mask3].max(), 200)
        ax.plot(xs, np.exp(np.polyval(p3, np.log(xs))), '--', color='C1', alpha=0.7,
                label=f'H3 fit: slope={slope3:.3f}, $R^2$={r23:.3f}')
    else:
        slope3 = np.nan

    # Plot remaining means and fit them as well
    if np.sum(mask1_rem) >= 2:
        x1r = np.log(ds_arr[mask1_rem])
        y1r = np.log(rem1_arr[mask1_rem])
        p1r = np.polyfit(x1r, y1r, 1)
        slope1r = float(p1r[0])
        xs_r = np.linspace(ds_arr[mask1_rem].min(), ds_arr[mask1_rem].max(), 200)
        ax.plot(ds_arr[mask1_rem], rem1_arr[mask1_rem], 'o--', color='C0', alpha=0.6, label='H1 rem mean')
        ax.plot(xs_r, np.exp(np.polyval(p1r, np.log(xs_r))), '-', color='C0', alpha=0.4,
                label=f'H1 rem fit: slope={slope1r:.3f}')
        # fill between mean +/- std
        try:
            std_vals = rem1_std_arr[mask1_rem]
            # interpolate std to xs_r using simple nearest mapping
            ax.fill_between(ds_arr[mask1_rem],
                            rem1_arr[mask1_rem] - std_vals,
                            rem1_arr[mask1_rem] + std_vals,
                            color='C0', alpha=0.15)
        except Exception:
            pass

    if np.sum(mask3_rem) >= 2:
        x3r = np.log(ds_arr[mask3_rem])
        y3r = np.log(rem3_arr[mask3_rem])
        p3r = np.polyfit(x3r, y3r, 1)
        slope3r = float(p3r[0])
        xs_r3 = np.linspace(ds_arr[mask3_rem].min(), ds_arr[mask3_rem].max(), 200)
        ax.plot(ds_arr[mask3_rem], rem3_arr[mask3_rem], 's--', color='C1', alpha=0.6, label='H3 rem mean')
        ax.plot(xs_r3, np.exp(np.polyval(p3r, np.log(xs_r3))), '-', color='C1', alpha=0.4,
                label=f'H3 rem fit: slope={slope3r:.3f}')
        try:
            std_vals3 = rem3_std_arr[mask3_rem]
            ax.fill_between(ds_arr[mask3_rem],
                            rem3_arr[mask3_rem] - std_vals3,
                            rem3_arr[mask3_rem] + std_vals3,
                            color='C1', alpha=0.12)
        except Exception:
            pass

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('d (hidden dimension)')
    ax.set_ylabel('leading sandwich eigenvalue')
    ax.set_title('Scaling of leading sandwich eigenvalue vs d (log-log)')
    ax.set_title('Scaling of sandwich eigenvalues vs d (log-log)')
    # Place legend underneath the main plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    # Plot prediction points (if available)
    try:
        # H1 predictions
        mask_pred_h1T = (pred_h1T_arr > 0) & np.isfinite(pred_h1T_arr)
        mask_pred_h1P = (pred_h1P_arr > 0) & np.isfinite(pred_h1P_arr)
        if np.any(mask_pred_h1T):
            ax.scatter(ds_arr[mask_pred_h1T], pred_h1T_arr[mask_pred_h1T],
                    marker='X', color='k', label='H1 Pred Target')
        if np.any(mask_pred_h1P):
            ax.scatter(ds_arr[mask_pred_h1P], pred_h1P_arr[mask_pred_h1P],
                    marker='^', color='k', label='H1 Pred Perp')

        # H3 predictions
        mask_pred_h3T = (pred_h3T_arr > 0) & np.isfinite(pred_h3T_arr)
        mask_pred_h3P = (pred_h3P_arr > 0) & np.isfinite(pred_h3P_arr)
        if np.any(mask_pred_h3T):
            ax.scatter(ds_arr[mask_pred_h3T], pred_h3T_arr[mask_pred_h3T],
                    marker='X', color='gray', label='H3 Pred Target')
        if np.any(mask_pred_h3P):
            ax.scatter(ds_arr[mask_pred_h3P], pred_h3P_arr[mask_pred_h3P],
                    marker='^', color='gray', label='H3 Pred Perp')
    except Exception:
        pass

    ax.legend()
    plt.tight_layout()

    outfn = os.path.join(out_dir, 'scaling_fit.png')
    plt.tight_layout()
    outfn = os.path.join(out_dir, 'scaling_fit.png')
    try:
        plt.savefig(outfn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('Saved scaling fit plot to', outfn)
    except Exception as e:
        print('Could not save scaling plot:', e)

    # Save summary npz with leading values, remaining stats and fit params
    try:
        slope1r_val = locals().get('slope1r', np.nan)
        slope3r_val = locals().get('slope3r', np.nan)
        summary_npz = os.path.join(out_dir, 'scaling_summary.npz')
        np.savez_compressed(summary_npz,
                            ds=ds_arr,
                            h1=h1_arr,
                            h3=h3_arr,
                            h1_rem_mean=rem1_arr,
                            h1_rem_std=rem1_std_arr,
                            h3_rem_mean=rem3_arr,
                            h3_rem_std=rem3_std_arr,
                            pred_h1T=pred_h1T_arr,
                            pred_h1P=pred_h1P_arr,
                            pred_h3T=pred_h3T_arr,
                            pred_h3P=pred_h3P_arr,
                            slope1=slope1,
                            slope3=slope3,
                            slope1_rem=slope1r_val,
                            slope3_rem=slope3r_val)
        print('Saved scaling summary to', summary_npz)
    except Exception as e:
        print('Could not save scaling summary:', e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze d-sweep experiments and plot sandwich eigenvalues')
    parser.add_argument('--start', type=int, default=DEFAULT_START)
    parser.add_argument('--stop', type=int, default=DEFAULT_STOP)
    parser.add_argument('--step', type=int, default=DEFAULT_STEP)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--use-cache', action='store_true', help='Load precomputed per-experiment caches instead of recomputing')
    args = parser.parse_args()

    d_vals = list(range(args.start, args.stop + 1, args.step))
    print('d values:', d_vals)
    recs = run_analysis(d_vals, use_cuda=args.cuda, use_cache=args.use_cache)
    # Determine out_dir used by run_analysis (recompute same naming logic)
    try:
        if EXPERIMENT_FILES:
            ds = []
            import re
            for f in EXPERIMENT_FILES:
                m = re.search(r"_D_(\d+)_", f)
                if m:
                    ds.append(int(m.group(1)))
            if not ds:
                ds = [DEFAULT_START, DEFAULT_STOP]
        else:
            ds = list(d_vals)
        d_min = min(ds)
        d_max = max(ds)
    except Exception:
        d_min = DEFAULT_START
        d_max = DEFAULT_STOP

    PROJECT_PLOTS = os.path.join(PROJECT_ROOT, 'plots')
    out_dir = os.path.join(PROJECT_PLOTS, f'd_sweep_d{d_min}_to_d{d_max}')

    # Aggregate and plot scaling fit
    try:
        aggregate_and_plot_scaling(recs, out_dir)
    except Exception as e:
        print('Failed to produce aggregate scaling plot:', e)

    print('\nDone. Saved outputs to project-level plots folder (plots/d_sweep_*)')
