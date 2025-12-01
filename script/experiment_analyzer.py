#!/usr/bin/env python3
"""
ExperimentAnalyzer
------------------
A small, focused helper class for loading the specified FCN3-Erf experiment
folders, computing eigenspectra and mean-field predictions, and producing
plots that compare results as a function of P.

This file purposefully contains detailed, structured comments (an "AI
playbook") describing how to extend the class for other plotting axes
(e.g. n1, chi, lrA) so an automated assistant can add features reliably.

Important: This file is written for analysis only. It does not run any
training, and it will not execute unless you call the provided methods.

Usage example (from the repo root):
    python -c "from script.experiment_analyzer import ExperimentAnalyzer;\n\
               a=ExperimentAnalyzer(); a.run_all()"

Do not run automatically -- edit paths to match your environment if needed.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# Ensure the repository's `lib/` directory is on sys.path so imports like
# `from lib.Experiment import Experiment` work regardless of the current
# working directory. `../` is measured relative to this `script/` folder.
_script_dir = Path(__file__).resolve().parent
_lib_dir = _script_dir.parent / 'lib'
sys.path.insert(0, str(_lib_dir))
sys.path.insert(0, str(_script_dir))
# Import the Experiment class from the repo's helper module
from Experiment import Experiment
import torch


# Load default experiment list from central registry; fall back to a local
# hardcoded list if import fails (keeps behaviour safe if the module is not
# available in some environments).
try:
    from experiment_collections import erf_cubic_P_SWEEP as _DEFAULT_EXPERIMENTS
except Exception:
    _DEFAULT_EXPERIMENTS = [
        'erf_cubic_eps_0.03_P_400_D_40_N_250_epochs_20000000_lrA_2.50e-09_time_20251125_140822',
        'erf_cubic_eps_0.03_P_200_D_40_N_250_epochs_20000000_lrA_5.00e-09_time_20251125_140822',
        'erf_cubic_eps_0.03_P_40_D_40_N_250_epochs_20000000_lrA_2.50e-08_time_20251125_140822',
        'erf_cubic_eps_0.03_P_1000_D_40_N_250_epochs_20000000_lrA_1.00e-09_time_20251125_140822',
    ]


def _parse_P_from_name(name: str) -> Optional[int]:
    m = re.search(r"[Pp]_(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_P(\d+)_", name)
    if m:
        return int(m.group(1))
    m = re.search(r"P(\d+)", name)
    if m:
        return int(m.group(1))
    return None


class ExperimentAnalyzer:
    """Load and analyze a fixed list of FCN3-Erf experiments.

    Responsibilities implemented now:
    - Load saved models from hardcoded experiment folders under `base_dir`.
    - Compute empirical eigenspectra using `Experiment.diagonalize_H`.
    - Compute mean-field predictions (via `Experiment.eig_predictions`).
    - Produce two plots: overlay eigenspectra (top-k) and MF-predictions vs P.

    Extension (AI-friendly) notes for future plotting axes (n1, n2, chi, lrA, etc.):
    1. Parameter discovery: implement `extract_hyperparams_from_name()` which
       parses experiment folder name for keys (e.g. 'N', 'd', 'chi', 'lrA'). If
       model objects contain metadata (e.g. stored in state_dict or log files),
       prefer reading that metadata.
    2. Data schema: store per-experiment data as a dict with keys: 'P', 'N', 'd',
       'chi', 'lrA', 'predictions', 'lambdas', 'summary_metrics'. Keep numeric
       scalars (like top eigenvalue) in 'summary_metrics' for plotting vs scalar axes.
    3. Plotting API: add generic method `plot_vs_axis(axis_key, metric_key, **opts)`
       which groups experiments by axis_key values, sorts, and plots metric_key vs
       axis_key. metric_key may select from summary_metrics or predictions fields.
    4. Sampling n1/n2: if plotting against internal neuron counts (n1/n2), extract
       them from the loaded `Experiment.model` object (e.g. `model.W0.shape[0]`). If
       the model uses unconventional naming, provide a mapping config.
    5. Automation hints for AI: first detect what keys are available in the data
       dict; then pick sensible defaults: top eigenvalue, trace, or MF lH1T.

    Implementation notes for maintainers:
    - Keep plotting code pure-Python with matplotlib so it's easy to run on headless
      servers (use `plt.savefig()` rather than `plt.show()` in scripts).
    - Use small, robust defaults for `k` when calling `diagonalize_H` to trade off
      speed vs accuracy (e.g. k = min(2000, max(100, top_k))).
    """

    def __init__(
        self,
        base_dir: str = '/home/akiva/exp/fcn3erf',
        exp_names: List[str] = None,
        device: Optional[str] = None,
        outdir: str = 'script/plots',
    ):
        self.base_dir = Path(base_dir)
        self.exp_names = exp_names if exp_names is not None else _DEFAULT_EXPERIMENTS
        self.device = device
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # storage for loaded experiment results
        self.results: List[Dict[str, Any]] = []

    def load_all(self, top_k: int = 200, compute_predictions: bool = True):
        """Load every hardcoded experiment and compute lambdas/predictions.

        Stores a list of dicts in `self.results` with keys:
          - name, P, N, d, lambdas (numpy, sorted desc), predictions (Eigenvalues dataclass), summary_metrics
        """
        self.results = []
        for name in self.exp_names:
            exp_dir = self.base_dir / name
            if not exp_dir.exists():
                print(f"Warning: experiment dir not found: {exp_dir}; skipping")
                continue

            print(f"Loading experiment: {exp_dir}")
            # parse params from name (best-effort)
            P = _parse_P_from_name(name)
            # defaults
            N = 250
            d = 40
            chi = 50

            # create Experiment instance
            exp = Experiment(file=str(exp_dir), N=N, d=d, chi=chi, P=P if P is not None else 0, ens=10)
            if self.device is not None:
                try:
                    exp.device = torch.device(self.device)
                except Exception:
                    pass

            # load model and compute predictions
            exp.load(compute_predictions=compute_predictions)

            # compute eigenspectrum
            X, Y1,Y3 = exp.large_dataset(p_large = 3000, flat=True)

            ls = exp.diagonalize_H(X, k=1000)
            ls_np = ls.detach().cpu().numpy()
            ls_sorted = np.sort(ls_np)[::-1]

            # summary metrics
            top_eig = float(ls_sorted[0]) if ls_sorted.size > 0 else np.nan
            trace = float(np.sum(ls_sorted)) if ls_sorted.size > 0 else np.nan
            n_large = int(np.sum(ls_sorted > 1e-3)) if ls_sorted.size > 0 else 0

            rec = {
                'name': name,
                'P': P,
                'N': N,
                'd': d,
                'chi': chi,
                'lambdas': ls_sorted,
                'predictions': exp.predictions,
                'summary_metrics': {
                    'top_eig': top_eig,
                    'trace': trace,
                    'n_large': n_large,
                },
                'experiment_obj': exp,  # keep a reference for advanced inspection
            }

            self.results.append(rec)

        # sort results by P ascending (useful for plotting vs P)
        self.results = sorted(self.results, key=lambda r: (r['P'] if r['P'] is not None else 0))

    def plot_eigenspectra_overlay(self, top_k: int = 300, save_name: str = 'eigenspectra_overlay.png'):
        """Overlay top-k eigenvalues from each experiment on a log-y plot.

        Saves figure to outdir/save_name.
        """
        plt.figure(figsize=(8, 6))
        for r in self.results:
            lambdas = r['lambdas'][:top_k]
            idx = np.arange(1, len(lambdas) + 1)
            plt.plot(idx, lambdas, marker='o', linestyle='-', label=f"P={r['P']}")

        plt.yscale('log')
        plt.xlabel('Eigenvalue index (largest to smaller)')
        plt.ylabel('Eigenvalue (log scale)')
        plt.title(f'Eigenspectra (top {top_k})')
        plt.legend()
        plt.tight_layout()
        out = self.outdir / save_name
        plt.savefig(out)
        print(f"Saved eigenspectra overlay to {out}")

    def plot_top_eig_vs_P(self, save_name: str = 'top_eig_vs_P.png'):
        """Plot the top empirical eigenvalue and MF predicted lH1T/lH1P vs P.

        Saves figure to outdir/save_name.
        """
        Ps = [r['P'] for r in self.results]
        top_vals = [r['summary_metrics']['top_eig'] for r in self.results]

        # MF predictions (if available)
        lH1T = [getattr(r['predictions'], 'lH1T', np.nan) if r['predictions'] is not None else np.nan for r in self.results]
        lH1P = [getattr(r['predictions'], 'lH1P', np.nan) if r['predictions'] is not None else np.nan for r in self.results]

        plt.figure(figsize=(8, 6))
        plt.plot(Ps, top_vals, marker='o', linestyle='-', label='empirical top eig')
        plt.plot(Ps, lH1T, marker='x', linestyle='--', label='MF lH1T')
        # plt.plot(Ps, lH1P, marker='s', linestyle='--', label='MF lH1P')

        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('P (log scale)')
        plt.ylabel('Top eigenvalue (log scale)')
        plt.title('Top eigenvalue vs P')
        plt.legend()
        plt.tight_layout()
        out = self.outdir / save_name
        plt.savefig(out)
        print(f"Saved top eigenvalue vs P to {out}")

    def plot_bar_eigenvalues_with_MF(self, save_prefix: str = 'bar_eigs'):
        """For each loaded experiment, plot a bar chart of all eigenvalues
        (from random SVD) and horizontal lines for the MF predictions.

        Saves one PNG per experiment named `{save_prefix}_P_<P>.png` in `outdir`.
        """
        for r in self.results:
            P = r.get('P')
            name = r.get('name')
            ls = r.get('lambdas')
            preds = r.get('predictions')

            if ls is None:
                print(f"No eigenvalues for {name}; skipping")
                continue

            ls_np = np.asarray(ls)

            plt.figure(figsize=(10, 6))
            plt.bar(np.arange(len(ls_np)), ls_np, color='blue', alpha=0.6, label='Eigenvalues')

            # Plot MF horizontal lines if predictions available
            if preds is not None:
                try:
                    plt.axhline(y=preds.lH1T, color='red', linestyle='--', label=r'$\mathbb{E}[\lambda^{H1}_T]$')
                except Exception:
                    pass
                try:
                    plt.axhline(y=preds.lH1P, color='orange', linestyle='--', label=r'$\mathbb{E}[\lambda^{H1}_P]$')
                except Exception:
                    pass
                try:
                    plt.axhline(y=preds.lH3T, color='green', linestyle='--', label=r'$\mathbb{E}[\lambda^{H3}_T]$')
                except Exception:
                    pass
                try:
                    plt.axhline(y=preds.lH3P, color='purple', linestyle='--', label=r'$\mathbb{E}[\lambda^{H3}_P]$')
                except Exception:
                    pass

            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('Eigenvalue Index')
            plt.ylabel('Eigenvalue Magnitude (log scale)')
            plt.title(f'Eigenvalues of H with MF Predictions (P={P})')
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)

            out = self.outdir / f"{save_prefix}_P_{P}.png"
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print(f"Saved bar eigenvalues with MF predictions to {out}")

    def save_raw_results(self, filename: str = 'results_summary.npy'):
        """Save a compact numpy summary (list of dicts) for later analysis.

        This writes a .npz containing P, top_eig, trace, and the MF predictions
        so other tools can load it quickly.
        """
        Ps = [r['P'] for r in self.results]
        top_eigs = [r['summary_metrics']['top_eig'] for r in self.results]
        traces = [r['summary_metrics']['trace'] for r in self.results]
        lH1T = [getattr(r['predictions'], 'lH1T', np.nan) if r['predictions'] is not None else np.nan for r in self.results]
        lH1P = [getattr(r['predictions'], 'lH1P', np.nan) if r['predictions'] is not None else np.nan for r in self.results]

        outp = self.outdir / filename
        np.savez(outp, P=np.array(Ps), top_eig=np.array(top_eigs), trace=np.array(traces), lH1T=np.array(lH1T), lH1P=np.array(lH1P))
        print(f"Saved summary .npz to {outp}")

    def run_all(self, top_k: int = 1000, compute_predictions: bool = True):
        """Convenience: load data, make plots, and save a summary.
        """
        self.load_all(top_k=top_k, compute_predictions=compute_predictions)
        self.plot_eigenspectra_overlay(top_k=top_k)
        self.plot_bar_eigenvalues_with_MF()
        self.plot_top_eig_vs_P()
        self.save_raw_results()


# If the file is executed, provide a small runnable example using the defaults.
if __name__ == '__main__':
    a = ExperimentAnalyzer()
    a.run_all()
    # Do not run heavy operations automatically — the user should call run_all explicitly.
    print('Created ExperimentAnalyzer with experiments:')
    for n in a.exp_names:
        print(' -', n)
    print('\nCall a.run_all() to load/compute/plot (this may be slow).')
