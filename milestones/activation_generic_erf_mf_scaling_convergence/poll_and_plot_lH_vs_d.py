#!/usr/bin/env python3
"""
Poll d-sweep models every few seconds, compute empirical/top eigenvalue lH, 
compute theoretical lH, and plot both vs d on log scales with slopes shown.

- Scans directories: d{d}_P{P}_N{N}_chi{N}
- Loads current model (prefers model.pt, falls back to model_final.pt)
- Computes empirical lH via H_eig on a large random Xinf
- Computes theoretical lH via ExperimentLinear.eig_predictions()
- Plots log-log lH vs d on dual y-axis (top eigenvalues, rest eigenvalues, and perpendicular)
- Legend includes line slopes for all series
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

# Workspace libs
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric
from Experiment import Experiment

POLL_INTERVAL = 5.0  # seconds
DIMS_DEFAULT = [150]
DEVICE_DEFAULT = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: List[int], suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Search recursively for all model.pt files in base directory.
    For each model.pt found, extract the parent directory (seed dir) and parse the seed number.
    Seed dirs come in form: /...kappa2.0/seed<num>
    Returns list of (path, cfg) with cfg={'d': d, 'P': P, 'N': N, 'chi': chi, 'seed': seed}.
    """
    import re
    selected: List[Tuple[Path, Dict[str, int]]] = []
    
    # Search recursively for all model.pt files
    model_files = list(base.glob("**/model.pt"))
    
    for model_file in model_files:
        # Get the parent directory of model.pt (should be seed<num> dir)
        seed_dir = model_file.parent
        seed_name = seed_dir.name
        
        # Extract seed number from seed_name (e.g., "seed1" -> 1)
        match = re.match(r"seed(\d+)", seed_name)
        if not match:
            continue
        seed = int(match.group(1))
        
        # Try to parse config from parent of seed dir
        # Assuming structure: .../d150_P1200_N1600_chi80_kappa2.0/seed1/model.pt
        config_dir = seed_dir.parent
        config_name = config_dir.name
        
        # Try to extract d, P, N, chi from config_name
        # Pattern: d<d>_P<P>_N<N>_chi<chi>_kappa<kappa>
        pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)"
        m = re.match(pattern, config_name)
        if not m:
            continue
        
        d = int(m.group(1))
        P = int(m.group(2))
        N = int(m.group(3))
        chi = int(float(m.group(4)))  # Handle both int and float
        
        # Filter by requested dims if provided
        if dims and d not in dims:
            continue
        
        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))
    
    # Sort by d, then by seed
    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    print(f"Found {len(selected)} run dirs with model.pt files.")
    return selected


def load_config_from_name(cfg: Dict[str, int]) -> Dict[str, int]:
    """Directly use the parsed config from the folder naming convention."""
    return cfg


def load_model(run_dir: Path, config: Dict, device: torch.device) -> Optional[FCN3NetworkActivationGeneric]:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(config.get("d"))
    P = int(config.get("P"))
    N = int(config.get("N"))
    chi = int(config.get("chi", N))

    # Load state_dict to extract ensemble size from W0
    state_dict = torch.load(model_path, map_location=device)
    ens = int(state_dict['W0'].shape[0])

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model

import traceback
def compute_empirical_lH(
    model: FCN3NetworkActivationGeneric, d: int, device: torch.device, seed: Optional[int] = None
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute top eigenvalue and mean of rest using H_eig(std=True).
    If seed is provided, use it to generate the dataset for reproducibility.
    Returns (mean_top, std_top, mean_rest, std_rest).
    """
    try:
        with torch.no_grad():
            # Use seed if provided for reproducible dataset generation
            if seed is not None:
                torch.manual_seed(seed)
            Xinf = torch.randn(5000, d, device=device)
            eig_mean, eig_std = model.H_eig(Xinf, Xinf, std=True)

            eig_mean_np = eig_mean.detach().cpu().numpy() 
            eig_std_np = None if eig_std is None else eig_std.detach().cpu().numpy()

            # Select the index of the largest mean eigenvalue
            top_idx = int(np.argmax(eig_mean_np))
            mean_top = float(eig_mean_np[top_idx])
            std_top = None if eig_std_np is None else float(eig_std_np[top_idx])
            
            # Compute mean and std of the rest (excluding top)
            rest_mask = np.ones(len(eig_mean_np), dtype=bool)
            rest_mask[top_idx] = False
            rest_eigs = eig_mean_np[rest_mask]
            mean_rest = float(np.mean(rest_eigs)) if len(rest_eigs) > 0 else None
            std_rest = float(np.std(rest_eigs)) if len(rest_eigs) > 0 else None
            
            return mean_top, std_top, mean_rest, std_rest
    except Exception as e:
        print(f"  Warning: failed to compute empirical lH: {e}")
        traceback.print_exc()
        return None, None, None, None


def compute_empirical_spectrum(
    model: FCN3NetworkActivationGeneric, d: int, P: int, device: torch.device, k: Optional[int] = None
) -> Optional[np.ndarray]:
    """Compute the full eigenvalue spectrum using randomized SVD on H.

    - Draw a random dataset X of size P x d
    - Use model.H_eig_random_svd(X, k) to estimate eigenvalues
    - Return eigenvalues as a NumPy array (unsorted, model-defined order)
    """
    try:
        with torch.no_grad():
            # Ensure model is on the same device as X
            model.to(device)
            model.device = device
            X = torch.randn(10000, d, device=device)
            # Choose rank k conservatively (<= P)
            if k is None:
                k = int(9000)
            eigs = model.H_eig_random_svd(X, k)
            return eigs.detach().cpu().numpy()
    except Exception as e:
        print(f"  Warning: failed to compute spectrum via H_eig_random_svd: {e}")
        return None


def compute_theoretical_full_predictions(
    cfg: Dict[str, int], device: torch.device
) -> Optional[object]:
    """Compute full theoretical predictions using ExperimentLinear.

    Returns the predictions object, which may have attributes like
    lH1T/lH1P (linear) or lH1T/lH1P/lH3T/lH3P (erf). For linear, only lH1T/lH1P
    will be present.
    """
    print(cfg)
    try:
        exp = Experiment(
            file=".",
            N=int(cfg.get("N")),
            d=int(cfg.get("d")),
            chi=float(cfg.get("chi")),
            P=int(cfg.get("P")),
            ens=50,
            device=device,
            eps = 0.03
        )
        preds = exp.eig_predictions()
        print(preds)
        print (f"  Theoretical predictions for d={cfg['d']}: {preds}")
        return preds
    except Exception as e:
        print(f"  Warning: failed to compute theoretical predictions: {e}")
        return None


def plot_eigenvalues_bar(
    run_dir: Path,
    cfg: Dict[str, int],
    spectrum_np: np.ndarray,
    preds_obj: Optional[object],
    out_path: Path,
) -> None:
    """Plot a bar chart of the empirical spectrum with theoretical overlays.

    Overlays the available theory curves:
    - Linear networks: lH1T (Target) and lH1P (perpendicular)
    - Erf networks (if provided): lH1T/lH1P and lH3T/lH3P
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        pos = np.arange(len(spectrum_np))
        # breakpoint()
        ax.bar(pos, spectrum_np, color='tab:blue', alpha=0.8, label='Empirical $\\lambda^H$')
        ax.set_yscale('log')

        # Overlay theoretical lines if available
        if preds_obj is not None:
            if hasattr(preds_obj, 'lH1T') and preds_obj.lH1T is not None:
                val = float(preds_obj.lH1T)
                if np.isfinite(val):
                    ax.axhline(y=val, color='tab:red', linestyle='--', linewidth=2.0,
                               label=r'Theory $\lambda_H^{(*)}$ (Target)')
            if hasattr(preds_obj, 'lH1P') and preds_obj.lH1P is not None:
                val = float(preds_obj.lH1P)
                if np.isfinite(val):
                    ax.axhline(y=val, color='tab:orange', linestyle='-.', linewidth=2.0,
                               label=r'Theory $\lambda_H^{(\perp)}$ (Perp)')
            if hasattr(preds_obj, 'lH3T') and preds_obj.lH3T is not None:
                val = float(preds_obj.lH3T)
                if np.isfinite(val):
                    ax.axhline(y=val, color='tab:purple', linestyle='--', linewidth=1.5,
                               label=r'Theory $\lambda_{H3}^{T}$')
            if hasattr(preds_obj, 'lH3P') and preds_obj.lH3P is not None:
                val = float(preds_obj.lH3P)
                if np.isfinite(val):
                    ax.axhline(y=val, color='tab:green', linestyle='-.', linewidth=1.5,
                               label=r'Theory $\lambda_{H3}^{P}$')
        ax.set_title(f"Spectrum for d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Magnitude (log scale)")
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xscale('log')
        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved spectrum bar plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot spectrum bar for {run_dir}: {e}")


def compute_theoretical_lH(run_dir: Path, d: int, device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    try:
        P = 1200
        N = 1600
        chi = 80.0
        kappa = float(2.0)
        exp = Experiment(
            file=str(run_dir), N=N, d=d, chi=chi, P=P, ens=50,
            kappa=kappa,
            device=device,
        )
        preds = exp.eig_predictions()
        print("EIG THEORETICAL PREDS:", preds)
        # Get Target-regime top eigenvalue (lH1T) and perpendicular (lH1P)
        lH1T = None
        lH1P = None
        if hasattr(preds, 'lH1T') and preds.lH1T is not None:
            lH1T = float(preds.lH1T)
        if hasattr(preds, 'lH1P') and preds.lH1P is not None:
            lH1P = float(preds.lH1P)
        return lH1T, lH1P
    except Exception as e:
        print(f"  Warning: failed to compute theoretical lH: {e}")
        return None, None


def line_slope_loglog(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    # slope in log-log: slope = d(log y) / d(log x) using LS fit
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or ys.size < 2:
        return None
    X = np.log(xs)
    Y = np.log(ys)
    A = np.vstack([X, np.ones_like(X)]).T
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m)


def update_plot(
    d_vals: List[int],
    emp_top: List[Optional[float]],
    emp_top_err: List[Optional[float]],
    emp_rest: List[Optional[float]],
    emp_rest_err: List[Optional[float]],
    th_top: List[Optional[float]],
    th_perp: List[Optional[float]],
    out_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Plot single y-axis figure with top/rest eigenvalues and theoretical predictions.
    
    All four series on single log-log plot:
    - Empirical top eigenvalues with error bars
    - Theoretical top eigenvalues
    - Empirical mean of rest eigenvalues with error bars
    - Theoretical perpendicular eigenvalues
    
    Returns (slope_emp_top, slope_th_top, slope_emp_rest, slope_th_perp).
    """
    d_arr = np.array(d_vals, dtype=float)
    emp_top_arr = np.array([v if v is not None else np.nan for v in emp_top], dtype=float)
    emp_top_err_arr = np.array([e if e is not None else np.nan for e in emp_top_err], dtype=float)
    emp_rest_arr = np.array([v if v is not None else np.nan for v in emp_rest], dtype=float)
    emp_rest_err_arr = np.array([e if e is not None else np.nan for e in emp_rest_err], dtype=float)
    th_top_arr = np.array([v if v is not None else np.nan for v in th_top], dtype=float)
    th_perp_arr = np.array([v if v is not None else np.nan for v in th_perp], dtype=float)

    slope_emp_top = line_slope_loglog(d_arr, emp_top_arr)
    slope_th_top = line_slope_loglog(d_arr, th_top_arr)
    slope_emp_rest = line_slope_loglog(d_arr, emp_rest_arr)
    slope_th_perp = line_slope_loglog(d_arr, th_perp_arr)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot empirical top eigenvalues with error bars
    if np.any(~np.isnan(emp_top_err_arr)):
        ax.errorbar(d_arr, emp_top_arr, yerr=emp_top_err_arr, fmt='o', color='tab:blue', ecolor='tab:blue',
                    elinewidth=1.2, capsize=3, label=f"Empirical $\\lambda_H^{{(*)}}$ ± std (slope={slope_emp_top:.3f})" if slope_emp_top is not None else "Empirical $\\lambda_H^{(*)}$ ± std")
        ax.plot(d_arr, emp_top_arr, '-', color='tab:blue', linewidth=2.0)
    else:
        ax.loglog(d_arr, emp_top_arr, 'o-', color='tab:blue', linewidth=2.5, markersize=8,
                  label=f"Empirical $\\lambda_H^{{(*)}}$ (slope={slope_emp_top:.3f})" if slope_emp_top is not None else "Empirical $\\lambda_H^{(*)}$")
    
    # Plot theoretical top eigenvalues
    ax.loglog(d_arr, th_top_arr, 's--', color='tab:red', linewidth=2.0, markersize=7,
              label=f"Theory $\\lambda_H^{{(*)}}$ (slope={slope_th_top:.3f})" if slope_th_top is not None else "Theory $\\lambda_H^{(*)}$")
    
    # Plot empirical rest eigenvalues with error bars
    if np.any(~np.isnan(emp_rest_err_arr)):
        ax.errorbar(d_arr, emp_rest_arr, yerr=emp_rest_err_arr, fmt='^', color='tab:green', ecolor='tab:green',
                    elinewidth=1.2, capsize=3, alpha=0.7, 
                    label=f"Empirical mean(rest) ± std (slope={slope_emp_rest:.3f})" if slope_emp_rest is not None else "Empirical mean(rest)")
        ax.plot(d_arr, emp_rest_arr, '-', color='tab:green', linewidth=2.0, alpha=0.7)
    else:
        ax.loglog(d_arr, emp_rest_arr, '^-', color='tab:green', linewidth=2.5, markersize=8, alpha=0.7,
                  label=f"Empirical mean(rest) (slope={slope_emp_rest:.3f})" if slope_emp_rest is not None else "Empirical mean(rest)")
    
    # Plot theoretical perpendicular eigenvalues
    ax.loglog(d_arr, th_perp_arr, 'D--', color='tab:orange', linewidth=2.0, markersize=7, alpha=0.7,
              label=f"Theory $\\lambda_H^{{(\\perp)}}$ (slope={slope_th_perp:.3f})" if slope_th_perp is not None else "Theory $\\lambda_H^{(\\perp)}$")

    ax.set_xlabel("Dimension $d$ (log scale)")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Empirical ($\\lambda_H^{(*)}$), Theoretical ($\\lambda_H^{(\\perp)}$) Eigenvalues of Preactivation Kernel vs $d$ \n FCN3 Linear Network in Mean-Field Scaling ($\\chi=N$))")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    
    # Set x-axis to start from first d value
    if len(d_arr) > 0:
        ax.set_xlim(left=d_arr[0] * 0.8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

    # Return all slopes for tracking
    return slope_emp_top, slope_th_top, slope_emp_rest, slope_th_perp


def poll_and_plot_once(
    base_dir: Path, device: torch.device, dims: List[int], suffix: str = ""
) -> Tuple[List[int], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Poll models and collect eigenvalue data.
    
    Returns 7 lists:
    - d_list: dimensions
    - emp_top_list: empirical top eigenvalues
    - emp_top_err_list: empirical top eigenvalue stds
    - emp_rest_list: empirical mean of rest eigenvalues
    - emp_rest_err_list: empirical std of rest eigenvalues
    - th_top_list: theoretical top eigenvalues (lH1T)
    - th_perp_list: theoretical perpendicular eigenvalues (lH1P)
    """
    runs = find_run_dirs(base_dir, dims, suffix=suffix)
    d_list: List[int] = []
    emp_top_list: List[Optional[float]] = []
    emp_top_err_list: List[Optional[float]] = []
    emp_rest_list: List[Optional[float]] = []
    emp_rest_err_list: List[Optional[float]] = []
    th_top_list: List[Optional[float]] = []
    th_perp_list: List[Optional[float]] = []

    for run_dir, cfg in runs:
        cfg = load_config_from_name(cfg)
        d = cfg["d"]
        seed = cfg.get("seed", None)  # Extract seed from cfg
        model = load_model(run_dir, cfg, device)
        if not model:
            continue
        
        mean_top, std_top, mean_rest, std_rest = compute_empirical_lH(model, d, device, seed=seed)
        lH1T, lH1P = compute_theoretical_lH(run_dir, d, device)

        # Additionally, compute and save per-run spectrum plots via randomized SVD
        spectrum_np = compute_empirical_spectrum(model, d, cfg["P"], device)
        preds_obj = compute_theoretical_full_predictions(cfg, device)
        if spectrum_np is not None:
            spectrum_out = run_dir / "plots" / f"spectrum_d{d}.png"
            plot_eigenvalues_bar(run_dir, cfg, spectrum_np, preds_obj, spectrum_out)
        
        d_list.append(d)
        emp_top_list.append(mean_top)
        emp_top_err_list.append(std_top)
        emp_rest_list.append(mean_rest)
        emp_rest_err_list.append(std_rest)
        th_top_list.append(lH1T)
        th_perp_list.append(lH1P)
    
    return d_list, emp_top_list, emp_top_err_list, emp_rest_list, emp_rest_err_list, th_top_list, th_perp_list


def poll_and_plot_with_self_consistent_kappa(
    base_dir: Path, device: torch.device, dims: List[int], suffix: str = ""
) -> None:
    """Poll models, group by P, compute dataset-averaged eigenvalues,
    solve for kappa_eff via self-consistent solver, and plot with theory.
    """
    import subprocess
    import tempfile
    import re
    
    runs = find_run_dirs(base_dir, dims, suffix=suffix)
    
    # Group runs by P
    runs_by_P: Dict[int, List[Tuple[Path, Dict]]] = {}
    for run_dir, cfg in runs:
        P = cfg["P"]
        if P not in runs_by_P:
            runs_by_P[P] = []
        runs_by_P[P].append((run_dir, cfg))
    
    # For each P, compute dataset averages and kappa_eff
    P_vals = sorted(runs_by_P.keys())
    empirical_tops = []
    empirical_top_errs = []
    empirical_he1_perp = []  # mean of indices 1:d (He1 perpendicular)
    empirical_he1_perp_errs = []
    empirical_cubic = []  # dth eigenvalue (cubic target)
    empirical_cubic_errs = []
    empirical_cubic_perp = []  # mean of (d+1):end (cubic perpendicular)
    empirical_cubic_perp_errs = []
    theoretical_tops_with_keff = []
    theoretical_perps_with_keff = []
    theoretical_cubic_top = []
    theoretical_cubic_perp = []
    
    # Store all seed-level data for plotting individual seeds
    all_seed_tops_by_P = {}  # P -> list of top eigenvalues (one per seed)
    all_seed_he1_perp_by_P = {}  # P -> list of He1 perp means (one per seed)
    all_seed_cubic_by_P = {}  # P -> list of dth eigenvalues (one per seed)
    all_seed_cubic_perp_by_P = {}  # P -> list of cubic perp means (one per seed)
    
    # Store averaged spectra for full spectrum plots
    avg_spectra_by_P = {}  # P -> averaged spectrum (sorted descending)
    
    for P in P_vals:
        runs_for_P = runs_by_P[P]
        emp_tops_seeds = []
        emp_he1_perp_seeds = []
        emp_cubic_target_seeds = []
        emp_cubic_perp_seeds = []
        
        # Compute empirical eigenvalues for each seed (and store spectra)
        seed_spectra = []
        for run_dir, cfg in runs_for_P:
            cfg = load_config_from_name(cfg)
            d = cfg["d"]
            seed = cfg.get("seed", None)
            model = load_model(run_dir, cfg, device)
            if not model:
                continue

            spectrum_np = compute_empirical_spectrum(model, d, cfg["P"], device)
            if spectrum_np is None:
                continue
            seed_spectra.append(spectrum_np)

            # Sort descending to extract: top (He1), He1_perp (1:d mean), cubic_target (dth), cubic_perp (d+1:end mean)
            spectrum_sorted = np.sort(spectrum_np)[::-1]
            if len(spectrum_sorted) > 0:
                emp_tops_seeds.append(float(spectrum_sorted[0]))
            if len(spectrum_sorted) > d:
                emp_he1_perp_seeds.append(float(np.mean(spectrum_sorted[1:d])))
            else:
                emp_he1_perp_seeds.append(np.nan)
            if len(spectrum_sorted) > d:
                emp_cubic_target_seeds.append(float(spectrum_sorted[d]))
            else:
                emp_cubic_target_seeds.append(np.nan)
            if len(spectrum_sorted) > d + 1:
                emp_cubic_perp_seeds.append(float(np.mean(spectrum_sorted[d+1:])))
            else:
                emp_cubic_perp_seeds.append(np.nan)
        
        if not emp_tops_seeds:
            continue
        
        # Store individual seed data
        all_seed_tops_by_P[P] = emp_tops_seeds
        all_seed_he1_perp_by_P[P] = emp_he1_perp_seeds
        all_seed_cubic_by_P[P] = emp_cubic_target_seeds
        all_seed_cubic_perp_by_P[P] = emp_cubic_perp_seeds
        
        # Compute dataset average and stderr
        emp_top_avg = np.mean(emp_tops_seeds)
        emp_top_err = np.std(emp_tops_seeds) / np.sqrt(len(emp_tops_seeds))
        empirical_tops.append(emp_top_avg)
        empirical_top_errs.append(emp_top_err)

        if emp_he1_perp_seeds:
            emp_he1_perp_vals = [v for v in emp_he1_perp_seeds if np.isfinite(v)]
            if emp_he1_perp_vals:
                emp_he1_perp_avg = np.mean(emp_he1_perp_vals)
                emp_he1_perp_err = np.std(emp_he1_perp_vals) / np.sqrt(len(emp_he1_perp_vals))
            else:
                emp_he1_perp_avg = np.nan
                emp_he1_perp_err = np.nan
        else:
            emp_he1_perp_avg = np.nan
            emp_he1_perp_err = np.nan
        empirical_he1_perp.append(emp_he1_perp_avg)
        empirical_he1_perp_errs.append(emp_he1_perp_err)

        if emp_cubic_target_seeds:
            emp_cubic_avg = np.mean(emp_cubic_target_seeds)
            emp_cubic_err = np.std(emp_cubic_target_seeds) / np.sqrt(len(emp_cubic_target_seeds))
        else:
            emp_cubic_avg = np.nan
            emp_cubic_err = np.nan
        empirical_cubic.append(emp_cubic_avg)
        empirical_cubic_errs.append(emp_cubic_err)
        
        if emp_cubic_perp_seeds:
            emp_cubic_perp_vals = [v for v in emp_cubic_perp_seeds if np.isfinite(v)]
            if emp_cubic_perp_vals:
                emp_cubic_perp_avg = np.mean(emp_cubic_perp_vals)
                emp_cubic_perp_err = np.std(emp_cubic_perp_vals) / np.sqrt(len(emp_cubic_perp_vals))
            else:
                emp_cubic_perp_avg = np.nan
                emp_cubic_perp_err = np.nan
        else:
            emp_cubic_perp_avg = np.nan
            emp_cubic_perp_err = np.nan
        empirical_cubic_perp.append(emp_cubic_perp_avg)
        empirical_cubic_perp_errs.append(emp_cubic_perp_err)
        
        # Call self-consistent kappa solver
        # Compute full spectrum for all seeds and average
        all_spectra = seed_spectra
        
        if all_spectra:
            # Average spectra across seeds
            avg_spectrum = np.mean(all_spectra, axis=0)
            
            # Store sorted spectrum (descending) for full spectrum plot
            avg_spectra_by_P[P] = np.sort(avg_spectrum)[::-1]
            
            # Write to temp JSON and call Julia solver
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf:
                json.dump({"eigenvalues": avg_spectrum.tolist(), "kappa_bare": 2.0}, tf)
                eig_json = tf.name
            
            try:
                julia_path = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
                sc_output = subprocess.check_output(
                    ["julia", str(julia_path), eig_json, str(P)],
                    text=True,
                    stderr=subprocess.STDOUT
                )
                
                # Parse kappa_eff from output
                match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_output)
                if match:
                    kappa_eff = float(match.group(1))
                    print(f"  P={P}: kappa_eff = {kappa_eff:.6f}")
                else:
                    print(f"  P={P}: Warning - could not parse kappa_eff")
                    kappa_eff = 1.0
            except Exception as e:
                print(f"  P={P}: Warning - self-consistent solver failed: {e}")
                kappa_eff = 1.0
            finally:
                import os
                if os.path.exists(eig_json):
                    os.remove(eig_json)
        else:
            kappa_eff = 2.0
        
        # Compute theory with effective kappa
        # Use first run's config as representative
        cfg = load_config_from_name(runs_for_P[0][1])
        d = cfg["d"]
        N = cfg["N"]
        chi = cfg["chi"]
        
        try:
            exp = Experiment(
                file=".",
                N=N,
                d=d,
                chi=chi,
                P=P,
                ens=50,
                device=device,
                kappa=kappa_eff,
                eps=0.03
            )
            preds = exp.eig_predictions()
            lH1T = float(preds.lH1T) if hasattr(preds, 'lH1T') and preds.lH1T is not None else None
            lH1P = float(preds.lH1P) if hasattr(preds, 'lH1P') and preds.lH1P is not None else None
            lH3T = float(preds.lH3T) if hasattr(preds, 'lH3T') and preds.lH3T is not None else None
            lH3P = float(preds.lH3P) if hasattr(preds, 'lH3P') and preds.lH3P is not None else None
            theoretical_tops_with_keff.append(lH1T)
            theoretical_perps_with_keff.append(lH1P)
            theoretical_cubic_top.append(lH3T)
            theoretical_cubic_perp.append(lH3P)
        except Exception as e:
            print(f"  P={P}: Warning - theory computation failed: {e}")
            theoretical_tops_with_keff.append(None)
            theoretical_perps_with_keff.append(None)
            theoretical_cubic_top.append(None)
            theoretical_cubic_perp.append(None)
    
    # Plot dataset averages with self-consistent theory
    if empirical_tops:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        P_arr = np.array(P_vals[:len(empirical_tops)])
        emp_arr = np.array(empirical_tops)
        emp_err_arr = np.array(empirical_top_errs)
        emp_he1_perp_arr = np.array(empirical_he1_perp)
        emp_he1_perp_err_arr = np.array(empirical_he1_perp_errs)
        emp_cubic_arr = np.array(empirical_cubic)
        emp_cubic_err_arr = np.array(empirical_cubic_errs)
        emp_cubic_perp_arr = np.array(empirical_cubic_perp)
        emp_cubic_perp_err_arr = np.array(empirical_cubic_perp_errs)
        th_arr = np.array([v if v is not None else np.nan for v in theoretical_tops_with_keff])
        th_perp_arr = np.array([v if v is not None else np.nan for v in theoretical_perps_with_keff])
        th_cubic_arr = np.array([v if v is not None else np.nan for v in theoretical_cubic_top])
        th_cubic_perp_arr = np.array([v if v is not None else np.nan for v in theoretical_cubic_perp])
        
        # Legend strings include numeric summaries (use last available point)
        latest_emp_mean = float(emp_arr[-1]) if len(emp_arr) else float('nan')
        latest_emp_std = float(emp_err_arr[-1]) if len(emp_err_arr) else float('nan')
        latest_emp_he1_perp = float(emp_he1_perp_arr[-1]) if len(emp_he1_perp_arr) else float('nan')
        latest_emp_he1_perp_std = float(emp_he1_perp_err_arr[-1]) if len(emp_he1_perp_err_arr) else float('nan')
        latest_emp_cubic = float(emp_cubic_arr[-1]) if len(emp_cubic_arr) else float('nan')
        latest_emp_cubic_std = float(emp_cubic_err_arr[-1]) if len(emp_cubic_err_arr) else float('nan')
        latest_emp_cubic_perp = float(emp_cubic_perp_arr[-1]) if len(emp_cubic_perp_arr) else float('nan')
        latest_emp_cubic_perp_std = float(emp_cubic_perp_err_arr[-1]) if len(emp_cubic_perp_err_arr) else float('nan')
        latest_th = float(th_arr[-1]) if len(th_arr) else float('nan')
        latest_th_perp = float(th_perp_arr[-1]) if len(th_perp_arr) else float('nan')
        latest_th_cubic = float(th_cubic_arr[-1]) if len(th_cubic_arr) else float('nan')
        latest_th_cubic_perp = float(th_cubic_perp_arr[-1]) if len(th_cubic_perp_arr) else float('nan')

        # Plot individual seed eigenvalues (all seeds, all P values)
        for i, P in enumerate(P_vals):
            if P in all_seed_tops_by_P:
                seed_tops = np.array(all_seed_tops_by_P[P])
                seed_he1_perp = np.array(all_seed_he1_perp_by_P[P])
                seed_cubic = np.array(all_seed_cubic_by_P[P])
                seed_cubic_perp = np.array(all_seed_cubic_perp_by_P[P])
                
                # Filter to only plot seeds where all four eigenvalue types are valid (finite)
                valid_mask = np.isfinite(seed_tops) & np.isfinite(seed_he1_perp) & np.isfinite(seed_cubic) & np.isfinite(seed_cubic_perp)
                if np.sum(valid_mask) == 0:
                    continue
                
                seed_tops_valid = seed_tops[valid_mask]
                seed_he1_perp_valid = seed_he1_perp[valid_mask]
                seed_cubic_valid = seed_cubic[valid_mask]
                seed_cubic_perp_valid = seed_cubic_perp[valid_mask]
                
                # Plot individual seeds with light transparency
                x_noise = np.random.normal(P, P * 0.02, size=len(seed_tops_valid))  # Add small x-jitter for visibility
                if i == 0:
                    ax.scatter(x_noise, seed_tops_valid, alpha=0.3, s=20, color='tab:blue', label='Individual seeds (He1)')
                    ax.scatter(x_noise, seed_he1_perp_valid, alpha=0.3, s=20, color='tab:orange', label='Individual seeds (He1_perp)')
                    ax.scatter(x_noise, seed_cubic_valid, alpha=0.3, s=20, color='tab:purple', label='Individual seeds (He3_target)')
                    ax.scatter(x_noise, seed_cubic_perp_valid, alpha=0.3, s=20, color='tab:pink', label='Individual seeds (He3_perp)')
                else:
                    ax.scatter(x_noise, seed_tops_valid, alpha=0.3, s=20, color='tab:blue')
                    ax.scatter(x_noise, seed_he1_perp_valid, alpha=0.3, s=20, color='tab:orange')
                    ax.scatter(x_noise, seed_cubic_valid, alpha=0.3, s=20, color='tab:purple')
                    ax.scatter(x_noise, seed_cubic_perp_valid, alpha=0.3, s=20, color='tab:pink')

        # Plot empirical dataset averages (He1)
        ax.errorbar(P_arr, emp_arr, yerr=emp_err_arr, fmt='o', color='tab:blue', ecolor='tab:blue',
                elinewidth=2, capsize=5, markersize=8,
                label=f"Empirical He1 mean (mean±SE): {latest_emp_mean:.4g}±{latest_emp_std:.4g}")
        ax.plot(P_arr, emp_arr, '-', color='tab:blue', linewidth=2.0, alpha=0.7)

        # Plot empirical He1 perpendicular (mean of indices 1:d)
        ax.errorbar(P_arr, emp_he1_perp_arr, yerr=emp_he1_perp_err_arr, fmt='D', color='tab:orange', ecolor='tab:orange',
                elinewidth=2, capsize=5, markersize=7,
                label=f"Empirical He1_perp mean (mean±SE): {latest_emp_he1_perp:.4g}±{latest_emp_he1_perp_std:.4g}")
        ax.plot(P_arr, emp_he1_perp_arr, '-', color='tab:orange', linewidth=1.8, alpha=0.7)

        # Plot empirical dataset averages (He3 cubic target / dth eigenvalue)
        ax.errorbar(P_arr, emp_cubic_arr, yerr=emp_cubic_err_arr, fmt='^', color='tab:purple', ecolor='tab:purple',
                elinewidth=2, capsize=5, markersize=7,
                label=f"Empirical He3_target mean (mean±SE): {latest_emp_cubic:.4g}±{latest_emp_cubic_std:.4g}")
        ax.plot(P_arr, emp_cubic_arr, '-', color='tab:purple', linewidth=1.8, alpha=0.7)

        # Plot empirical cubic perpendicular (mean of d+1:end eigenvalues)
        ax.errorbar(P_arr, emp_cubic_perp_arr, yerr=emp_cubic_perp_err_arr, fmt='s', color='tab:pink', ecolor='tab:pink',
                elinewidth=2, capsize=5, markersize=7,
                label=f"Empirical He3_perp mean (mean±SE): {latest_emp_cubic_perp:.4g}±{latest_emp_cubic_perp_std:.4g}")
        ax.plot(P_arr, emp_cubic_perp_arr, '-', color='tab:pink', linewidth=1.8, alpha=0.7)
        
        # Plot theory with self-consistent kappa as horizontal reference lines (avoid mathtext parse issues)
        if np.isfinite(latest_th):
            ax.axhline(latest_th, color='tab:red', linewidth=2.0, linestyle='--',
                       label=f"Theory kappa_eff He1: {latest_th:.4g}")
        if np.isfinite(latest_th_perp):
            ax.axhline(latest_th_perp, color='tab:orange', linewidth=2.0, linestyle='--', alpha=0.7,
                       label=f"Theory kappa_eff He1_perp: {latest_th_perp:.4g}")

        # Plot theory for cubic (He3) as horizontal reference lines
        if np.isfinite(latest_th_cubic):
            ax.axhline(latest_th_cubic, color='tab:green', linewidth=2.0, linestyle='-.',
                       label=f"Theory kappa_eff He3_target: {latest_th_cubic:.4g}")
        if np.isfinite(latest_th_cubic_perp):
            ax.axhline(latest_th_cubic_perp, color='tab:brown', linewidth=2.0, linestyle='-.', alpha=0.7,
                       label=f"Theory kappa_eff He3_perp: {latest_th_cubic_perp:.4g}")
        
        ax.set_xlabel('P (number of samples)')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Dataset-Averaged Empirical vs Theory Eigenvalues (Self-Consistent Ridge Correction)')
        # Logscale y-axis
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        fig.tight_layout()
        out_path = base_dir / "plots" / f"dataset_avg_with_self_consistent_kappa{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved self-consistent kappa plot to {out_path}")
        
        # Plot full spectra for each P value
        for P in P_vals:
            if P not in avg_spectra_by_P:
                continue
            
            avg_spectrum_sorted = avg_spectra_by_P[P]
            
            # Get corresponding theory predictions
            idx_in_theory = P_vals.index(P)
            th_he1_target = theoretical_tops_with_keff[idx_in_theory]
            th_he1_perp = theoretical_perps_with_keff[idx_in_theory]
            th_he3_target = theoretical_cubic_top[idx_in_theory]
            th_he3_perp = theoretical_cubic_perp[idx_in_theory]
            
            # Create bar plot of full spectrum
            fig, ax = plt.subplots(figsize=(14, 6))
            pos = np.arange(len(avg_spectrum_sorted))
            ax.bar(pos, avg_spectrum_sorted, color='tab:blue', alpha=0.7, label='Empirical eigenvalues')
            
            # Add theoretical reference lines
            if th_he1_target is not None and np.isfinite(th_he1_target):
                ax.axhline(th_he1_target, color='tab:red', linewidth=2.0, linestyle='--', 
                          label=f"Theory He1_target: {th_he1_target:.4g}")
            if th_he1_perp is not None and np.isfinite(th_he1_perp):
                ax.axhline(th_he1_perp, color='tab:orange', linewidth=2.0, linestyle='--', alpha=0.7,
                          label=f"Theory He1_perp: {th_he1_perp:.4g}")
            if th_he3_target is not None and np.isfinite(th_he3_target):
                ax.axhline(th_he3_target, color='tab:green', linewidth=2.0, linestyle='-.',
                          label=f"Theory He3_target: {th_he3_target:.4g}")
            if th_he3_perp is not None and np.isfinite(th_he3_perp):
                ax.axhline(th_he3_perp, color='tab:brown', linewidth=2.0, linestyle='-.', alpha=0.7,
                          label=f"Theory He3_perp: {th_he3_perp:.4g}")
            
            ax.set_yscale('log')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Eigenvalue (log scale)')
            ax.set_title(f'Full Dataset-Averaged Spectrum (P={P}) with Theory Reference Lines')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            fig.tight_layout()
            spectrum_out_path = base_dir / "plots" / f"full_spectrum_P{P}{suffix}.png"
            spectrum_out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(spectrum_out_path, dpi=200)
            plt.close(fig)
            print(f"Saved full spectrum plot to {spectrum_out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Poll and plot lH vs d for d-sweep runs")
    parser.add_argument("--from-checkpoint", action="store_true", help="Poll *_from_checkpoint runs instead of base runs")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    suffix = "_from_checkpoint" if args.from_checkpoint else ""
    out_path = base_dir / "plots" / f"poll_lH_vs_d{suffix}.png"
    slopes_path = base_dir / "plots" / f"slope_history{suffix}.json"
    slopes_plot_path = base_dir / "plots" / f"slopes_over_time{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = DEVICE_DEFAULT

    # Load prior slope history if exists
    slope_hist_emp_top: List[float] = []
    slope_hist_th_top: List[float] = []
    slope_hist_emp_rest: List[float] = []
    slope_hist_th_perp: List[float] = []
    try:
        if slopes_path.exists():
            with open(slopes_path, "r") as f:
                data = json.load(f)
                slope_hist_emp_top = data.get("emp_top", [])
                slope_hist_th_top = data.get("th_top", [])
                slope_hist_emp_rest = data.get("emp_rest", [])
                slope_hist_th_perp = data.get("th_perp", [])
    except Exception:
        pass

    print("Polling d-sweep runs and updating lH vs d plot...")
    print(f"Base: {base_dir}")
    print(f"Output: {out_path}")

    try:
        while True:
            # Generate standard d-sweep plot
            d_vals, emp_top, emp_top_err, emp_rest, emp_rest_err, th_top, th_perp = poll_and_plot_once(
                base_dir, device, DIMS_DEFAULT, suffix=suffix
            )
            if d_vals:
                # Sort by d for clean plotting
                order = np.argsort(d_vals)
                d_vals = list(np.array(d_vals)[order])
                emp_top = list(np.array(emp_top, dtype=object)[order])
                emp_top_err = list(np.array(emp_top_err, dtype=object)[order])
                emp_rest = list(np.array(emp_rest, dtype=object)[order])
                emp_rest_err = list(np.array(emp_rest_err, dtype=object)[order])
                th_top = list(np.array(th_top, dtype=object)[order])
                th_perp = list(np.array(th_perp, dtype=object)[order])
                
                slope_emp_top, slope_th_top, slope_emp_rest, slope_th_perp = update_plot(
                    d_vals, emp_top, emp_top_err, emp_rest, emp_rest_err, th_top, th_perp, out_path
                )

                # Append slopes if changed from last measurement
                def maybe_append(hist: List[float], val: Optional[float], tol: float = 1e-6) -> bool:
                    if val is None or np.isnan(val):
                        return False
                    if not hist:
                        hist.append(float(val))
                        return True
                    if abs(hist[-1] - float(val)) > tol:
                        hist.append(float(val))
                        return True
                    return False

                changed_emp_top = maybe_append(slope_hist_emp_top, slope_emp_top)
                changed_th_top = maybe_append(slope_hist_th_top, slope_th_top)
                changed_emp_rest = maybe_append(slope_hist_emp_rest, slope_emp_rest)
                changed_th_perp = maybe_append(slope_hist_th_perp, slope_th_perp)

                # Persist history if any change
                if changed_emp_top or changed_th_top or changed_emp_rest or changed_th_perp:
                    try:
                        with open(slopes_path, "w") as f:
                            json.dump({
                                "emp_top": slope_hist_emp_top,
                                "th_top": slope_hist_th_top,
                                "emp_rest": slope_hist_emp_rest,
                                "th_perp": slope_hist_th_perp,
                            }, f, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to save slope history: {e}")
                
                # Plot slopes over time (measurement index)
                if slope_hist_emp_top or slope_hist_th_top or slope_hist_emp_rest or slope_hist_th_perp:
                    plt.figure(figsize=(12, 6))
                    if slope_hist_emp_top:
                        plt.plot(range(1, len(slope_hist_emp_top)+1), slope_hist_emp_top, '-o', color='tab:blue', label='Empirical top slope')
                    if slope_hist_th_top:
                        plt.plot(range(1, len(slope_hist_th_top)+1), slope_hist_th_top, '--s', color='tab:red', label='Theory top slope')
                    if slope_hist_emp_rest:
                        plt.plot(range(1, len(slope_hist_emp_rest)+1), slope_hist_emp_rest, '-^', color='tab:green', label='Empirical rest slope', alpha=0.7)
                    if slope_hist_th_perp:
                        plt.plot(range(1, len(slope_hist_th_perp)+1), slope_hist_th_perp, '--D', color='tab:orange', label='Theory perp slope', alpha=0.7)
                    
                    plt.xlabel('Measurement index')
                    plt.ylabel(r'$\alpha$ (slope in log-log)')
                    plt.title(r'Eigenvalue Scaling Exponents vs Time: $\alpha(t)$ where $\lambda \sim d^\alpha$')
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    try:
                        plt.savefig(slopes_plot_path, dpi=200)
                        print(f"Saved slope history plot to {slopes_plot_path}")
                    finally:
                        plt.close()
                
                # Also generate self-consistent kappa plot (dataset averages across P)
                poll_and_plot_with_self_consistent_kappa(base_dir, device, DIMS_DEFAULT, suffix=suffix)
            else:
                print("No matching d{d}_P{3d}_N50_chi50 directories found yet.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping polling.")


if __name__ == "__main__":
    main()
