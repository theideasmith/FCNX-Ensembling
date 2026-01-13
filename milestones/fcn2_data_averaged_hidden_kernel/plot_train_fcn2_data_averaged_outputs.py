#!/usr/bin/env python3
"""
Plot outputs for 2-layer erf data-averaged training runs.
Replicates the key plots from the erf hidden-kernel workflow, including:
- Eigenvalues over time
- Final spectrum with optional Julia predictions overlay
- Predictions vs True (mean over datasets/ensembles)
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt


def _compute_julia_fcn2_erf_predictions(d, N, P, chi):
    """Call Julia to compute theoretical FCN2 erf eigenvalues (lk for δ=1, lkp for δ=0).
    Returns (lk, lkp) or (None, None) on failure.
    """
    julia_script = Path(__file__).parent.parent / 'julia_lib' / 'compute_fcn2_erf_eigs.jl'
    if not julia_script.exists():
        return None, None
    try:
        cmd = ['julia', str(julia_script), '--d', str(d), '--N', str(N), '--P', str(P), '--chi', str(chi)]
        rs = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(rs.stdout)
        lk = float(data.get('lk', 'nan'))
        lkp = float(data.get('lkp', 'nan'))
        if not np.isfinite(lk):
            lk = None
        if not np.isfinite(lkp):
            lkp = None
        return lk, lkp
    except Exception:
        return None, None


def plot_eigenvalues_over_time(run_dir: Path):
    path = run_dir / 'eigenvalues_over_time.json'
    if not path.exists():
        print(f"No eigenvalues file found at {path}")
        return
    with open(path, 'r') as f:
        eig_data = json.load(f)
    epochs = sorted([int(k) for k in eig_data.keys()])
    eigenvalues = np.array([eig_data[str(e)] for e in epochs])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(eigenvalues.shape[1]):
        ax.plot(epochs, eigenvalues[:, i], alpha=0.6, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Data-Avg H Eigenvalues over Training\n{run_dir.name}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / 'eigenvalues_over_time.png', dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir / 'eigenvalues_over_time.png'}")


def plot_final_spectrum(run_dir: Path, d: int, N: int, P: int, chi: float):
    path = run_dir / 'eigenvalues_over_time.json'
    if not path.exists():
        print(f"No eigenvalues file found at {path}")
        return
    with open(path, 'r') as f:
        eig_data = json.load(f)
    last_epoch = max([int(k) for k in eig_data.keys()])
    final_eigs = np.array(eig_data[str(last_epoch)])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len(final_eigs)), final_eigs, marker='o', ms=3, lw=1.2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Final Data-Avg H Spectrum (epoch {last_epoch})\n{run_dir.name}')
    # Optional Julia overlay
    lk, lkp = _compute_julia_fcn2_erf_predictions(d=d, N=N, P=P, chi=chi)
    if lk is not None:
        ax.axhline(lk, color='tab:red', linestyle='--', linewidth=1.5, label='Julia lk (δ=1)')
    if lkp is not None:
        ax.axhline(lkp, color='tab:green', linestyle='-.', linewidth=1.5, label='Julia lkp (δ=0)')
    if lk is not None or lkp is not None:
        ax.legend(loc='best')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = run_dir / 'eigenvalues_final_model.png'
    jpg_path = run_dir / 'eigenvalues_final_model.jpg'
    fig.savefig(png_path, dpi=150)
    fig.savefig(jpg_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path} and {jpg_path}")


def plot_pred_vs_true(run_dir: Path):
    path = run_dir / 'pred_vs_true.json'
    if not path.exists():
        print(f"No pred_vs_true file found at {path}")
        return
    with open(path, 'r') as f:
        data = json.load(f)
    if not data:
        print("pred_vs_true.json is empty")
        return
    last_epoch = max([int(k) for k in data.keys()])
    rec = data[str(last_epoch)]
    y_true = np.array(rec['y_true'])
    y_pred = np.array(rec['y_pred_mean'])
    slope = rec.get('slope', np.nan)
    intercept = rec.get('intercept', np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.7)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted (mean over D,Q)')
    ax.set_title(f'Pred vs True (epoch {last_epoch})\n{run_dir.name}')
    if np.isfinite(slope) and np.isfinite(intercept):
        xs = np.linspace(y_true.min(), y_true.max(), 200)
        ax.plot(xs, slope * xs + intercept, color='orange', lw=2, label=f'fit slope={slope:.3f}')
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / 'pred_vs_true.png', dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir / 'pred_vs_true.png'}")


def main():
    parser = argparse.ArgumentParser(description='Plot outputs for data-averaged FCN2')
    parser.add_argument('--run-dir', type=str, required=True, help='Training run directory')
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--P', type=int, default=200)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--chi', type=float, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    chi = args.chi if args.chi is not None else float(args.N)

    plot_eigenvalues_over_time(run_dir)
    plot_final_spectrum(run_dir, d=args.d, N=args.N, P=args.P, chi=chi)
    plot_pred_vs_true(run_dir)


if __name__ == '__main__':
    main()
