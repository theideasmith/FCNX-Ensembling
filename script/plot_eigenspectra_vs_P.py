#!/usr/bin/env python3
"""
Load several trained FCN3-Erf ensembles and plot their eigenspectra and mean-field
predictions versus P.

Usage (example):
    python script/plot_eigenspectra_vs_P.py \
        --base-dir /home/akiva/exp \
        --outdir script/plots \
        --top-k 200 \
        --exp-names erf_cubic_eps_0.03_P_400_D_40_N_250_epochs_20000000_lrA_2.50e-09_time_20251125_140822 \
                    erf_cubic_eps_0.03_P_200_D_40_N_250_epochs_20000000_lrA_5.00e-09_time_20251125_140822 \
                    erf_cubic_eps_0.03_P_40_D_40_N_250_epochs_20000000_lrA_2.50e-08_time_20251125_140822 \
                    erf_cubic_eps_0.03_P_1000_D_40_N_250_epochs_20000000_lrA_1.00e-09_time_20251125_140822

This script does not run training; it only loads saved `model.pth` files from each
experiment folder and computes/plots spectra.
"""

import argparse
import re
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
# Path to the ensembling script relative to repo root (adjust if different)
ENSEMBLING_SCRIPT = Path(__file__).resolve().parent / "ensembling_fcn3_erf_cubic.py"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_script_dir = Path(__file__).resolve().parent
_lib_dir = _script_dir.parent / 'lib'
sys.path.insert(0, str(_lib_dir))

# Import the Experiment class from the repo's helper module
from Experiment import Experiment
# Import the Experiment helper from the repo
from Experiment import Experiment


def parse_P_from_name(name: str):
    m = re.search(r"[Pp]_(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_P(\d+)_", name)
    if m:
        return int(m.group(1))
    # fallback: look for 'P' followed by digits anywhere
    m = re.search(r"P(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def load_and_compute(exp_dir: Path, top_k=200, compute_predictions=True, device='cuda:1'):
    # parse P/d/N/chi from folder name as best-effort
    name = exp_dir.name
    P = parse_P_from_name(name)

    # Heuristic defaults for arguments expected by Experiment
    # Experiment requires file, N, d, chi, P, ens
    # We'll try to parse N and d too, otherwise use sensible defaults
    def parse_param(key, default=None):
        m = re.search(fr"{key}_(\d+)", name)
        return int(m.group(1)) if m else default

    N = parse_param('N', 250)
    d = parse_param('D', parse_param('d', 40))
    chi = parse_param('chi', 50)
    if P is None:
        raise ValueError(f"Could not parse P from experiment name: {name}")

    exp = Experiment(file=str(exp_dir), N=N, d=d, chi=chi, P=P, ens=10, device = torch.device(device))
    if device is not None:
        try:
            exp.device = torch.device(device)
        except Exception:
            pass

    # Load model and compute predictions
    exp.load(compute_predictions=compute_predictions)

    # Compute eigenspectrum (use k equal to min(5000, d*10) or let function choose)
    X, Y = exp.large_dataset(p_large=3000)
    X = X.to(torch.device(device))
    ls = exp.diagonalize_H(X, k=2000)

    # Convert to numpy and sort descending
    ls_np = ls.detach().cpu().numpy()
    ls_sorted = np.sort(ls_np)[::-1]

    preds = exp.predictions

    return {
        'name': name,
        'P': P,
        'N': N,
        'd': d,
        'lambdas': ls_sorted,
        'predictions': preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot eigenspectra and predictions for a fixed list of experiments")
    parser.add_argument('--base-dir', type=str, default='/home/akiva/exp/fcn3erf', help='Base directory containing experiment folders')
    parser.add_argument('--outdir', type=str, default='script/plots', help='Directory to save plots')
    parser.add_argument('--top-k', type=int, default=200, help='Number of top eigenvalues to plot')
    parser.add_argument('--device', type=str, default=None, help="Optional torch device to use when loading models, e.g. 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    # Hardcoded list of experiment folders (will be looked up under --base-dir)
    EXP_NAMES = [
        'erf_cubic_eps_0.03_P_400_D_40_N_250_epochs_20000000_lrA_2.50e-09_time_20251125_140822',
        'erf_cubic_eps_0.03_P_200_D_40_N_250_epochs_20000000_lrA_5.00e-09_time_20251125_140822',
        'erf_cubic_eps_0.03_P_40_D_40_N_250_epochs_20000000_lrA_2.50e-08_time_20251125_140822',
        'erf_cubic_eps_0.03_P_1000_D_40_N_250_epochs_20000000_lrA_1.00e-09_time_20251125_140822',
    ]

    base = Path(args.base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in EXP_NAMES:
        exp_dir = base / name
        if not exp_dir.exists():
            print(f"Warning: experiment dir not found: {exp_dir}; skipping")
            continue
        print(f"Loading {exp_dir} ...")
        try:
            res = load_and_compute(exp_dir, top_k=args.top_k, compute_predictions=True, device='cuda:1')
            results.append(res)
        except Exception as e:
            print(f"Failed to load/process {exp_dir}: {e}")

    if not results:
        print("No experiments loaded. Exiting.")
        return

    # Sort results by P ascending for plotting vs P
    results = sorted(results, key=lambda r: r['P'])

    # --- Plot 1: overlay eigenspectra (top-k) for each P ---
    plt.figure(figsize=(8, 6))
    for r in results:
        lambdas = r['lambdas'][:args.top_k]
        idx = np.arange(1, len(lambdas) + 1)
        plt.plot(idx, lambdas, marker='o', linestyle='-', label=f"P={r['P']}")

    plt.yscale('log')
    plt.xlabel('Eigenvalue index (largest to smaller)')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Eigenspectra (top {:d}) for experiments'.format(args.top_k))
    plt.legend()
    plt.tight_layout()
    spec_path = outdir / 'eigenspectra_overlay.png'
    plt.savefig(spec_path)
    print(f"Saved eigenspectra overlay to {spec_path}")

    # --- Plot 2: mean-field predicted eigenvalues vs P ---
    # We'll plot a handful of MF predictions: lH1T, lH1P, lH3T, lH3P if available
    Ps = [r['P'] for r in results]
    labels = ['lH1T', 'lH1P', 'lH3T', 'lH3P']
    pred_map = {lab: [] for lab in labels}

    for r in results:
        preds = r['predictions']
        if preds is None:
            for lab in labels:
                pred_map[lab].append(np.nan)
        else:
            pred_map['lH1T'].append(preds.lH1T)
            pred_map['lH1P'].append(preds.lH1P)
            pred_map['lH3T'].append(preds.lH3T)
            pred_map['lH3P'].append(preds.lH3P)

    plt.figure(figsize=(8, 6))
    for lab in labels:
        vals = np.array(pred_map[lab], dtype=np.float64)
        alphas = np.log(Ps) / np.log(np.array([r['d'] for r in results]))
        plt.plot(alphas, vals, marker='o', linestyle='-', label=lab)

    plt.yscale('log')
    plt.xlabel('alpha)')
    plt.ylabel('Predicted eigenvalue (log scale)')
    plt.title('Mean-field predicted eigenvalues vs Alphas')
    plt.legend()
    plt.tight_layout()
    pred_path = outdir / 'mf_predictions_vs_P.png'
    plt.savefig(pred_path)
    print(f"Saved MF predictions vs P to {pred_path}")


if __name__ == '__main__':
    main()
