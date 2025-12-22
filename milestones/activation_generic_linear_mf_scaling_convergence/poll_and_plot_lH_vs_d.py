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
from Experiment import ExperimentLinear

POLL_INTERVAL = 5.0  # seconds
DIMS_DEFAULT = [2, 6, 8, 10]
DEVICE_DEFAULT = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: List[int]) -> List[Tuple[Path, Dict[str, int]]]:
    """Select only the folders following the pattern:
    d{d}_P{3d}_N50_chi50 for d in dims.
    Returns list of (path, cfg) with cfg={'d': d, 'P': 3*d, 'N': 50, 'chi': 50}.
    """
    selected: List[Tuple[Path, Dict[str, int]]] = []
    for d in dims:
        name = f"d{d}_P{3*d}_N50_chi50"
        p = base / name
        if p.is_dir():
            selected.append((p, {"d": d, "P": 3*d, "N": 50, "chi": 50}))
    # Sort by d
    selected.sort(key=lambda x: x[1]["d"]) 
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

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=50,
        activation="linear",
        weight_initialization_variance=weight_var,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

import traceback
def compute_empirical_lH(
    model: FCN3NetworkActivationGeneric, d: int, device: torch.device
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute top eigenvalue and mean of rest using H_eig(std=True).
    Returns (mean_top, std_top, mean_rest, std_rest).
    """
    try:
        with torch.no_grad():
            Xinf = torch.randn(3000, d, device=device)
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
            Xinf = torch.randn(P, d, device=device)
            # Choose rank k conservatively (<= P)
            if k is None:
                k = int(min(P, 1000))
            eigs = model.H_eig(Xinf, Xinf)
            return eigs.detach().cpu().numpy()
    except Exception as e:
        print(f"  Warning: failed to compute spectrum via H_eig_random_svd: {e}")
        return None


def compute_theoretical_full_predictions(
    cfg: Dict[str, int], device: torch.device
) -> Optional[object]:
    """Compute full theoretical predictions using ExperimentLinear.

    Returns the predictions object, which may have attributes like
    lHT/lHP (linear) or lH1T/lH1P/lH3T/lH3P (erf). For linear, only lHT/lHP
    will be present.
    """
    try:
        exp = ExperimentLinear(
            file=".",
            N=int(cfg.get("N")),
            d=int(cfg.get("d")),
            chi=float(cfg.get("chi")),
            P=int(cfg.get("P")),
            ens=50,
            device=device,
        )
        preds = exp.eig_predictions()
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
    - Linear networks: lHT (training) and lHP (perpendicular)
    - Erf networks (if provided): lH1T/lH1P and lH3T/lH3P
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        pos = np.arange(len(spectrum_np))
        ax.bar(pos, spectrum_np, color='tab:blue', alpha=0.8, label='Empirical $\\lambda^H$')
        ax.set_yscale('log')

        # Overlay theoretical lines if available
        if preds_obj is not None:
            # Linear predictions
            if hasattr(preds_obj, 'lHT') and preds_obj.lHT is not None:
                ax.axhline(y=float(preds_obj.lHT), color='tab:red', linestyle='--', linewidth=2.0,
                           label=r'Theory $\lambda_H^{(*)}$ (training)')
            if hasattr(preds_obj, 'lHP') and preds_obj.lHP is not None:
                ax.axhline(y=float(preds_obj.lHP), color='tab:orange', linestyle='-.', linewidth=2.0,
                           label=r'Theory $\lambda_H^{(\perp)}$ (population)')

            # Erf-style predictions (if object carries these attributes)
            if hasattr(preds_obj, 'lH1T') and preds_obj.lH1T is not None:
                ax.axhline(y=float(preds_obj.lH1T), color='tab:red', linestyle='--', linewidth=1.5,
                           label=r'Theory $\lambda_{H1}^{T}$')
            if hasattr(preds_obj, 'lH1P') and preds_obj.lH1P is not None:
                ax.axhline(y=float(preds_obj.lH1P), color='tab:orange', linestyle='-.', linewidth=1.5,
                           label=r'Theory $\lambda_{H1}^{P}$')
            if hasattr(preds_obj, 'lH3T') and preds_obj.lH3T is not None:
                ax.axhline(y=float(preds_obj.lH3T), color='tab:purple', linestyle='--', linewidth=1.5,
                           label=r'Theory $\lambda_{H3}^{T}$')
            if hasattr(preds_obj, 'lH3P') and preds_obj.lH3P is not None:
                ax.axhline(y=float(preds_obj.lH3P), color='tab:green', linestyle='-.', linewidth=1.5,
                           label=r'Theory $\lambda_{H3}^{P}$')

        ax.set_title(f"Spectrum for d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Magnitude (log scale)")
        ax.grid(True, which='both', alpha=0.3)
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
        P = 3 * d
        N = 50 
        chi = 50
        kappa = float(1.0)
        exp = ExperimentLinear(
            file=str(run_dir), N=N, d=d, chi=chi, P=P, ens=50,
            kappa=kappa,
            device=device,
        )
        preds = exp.eig_predictions()
        # Get training-regime top eigenvalue (lHT) and perpendicular (lHP)
        lHT = None
        lHP = None
        if hasattr(preds, 'lHT') and preds.lHT is not None:
            lHT = float(preds.lHT)
        if hasattr(preds, 'lHP') and preds.lHP is not None:
            lHP = float(preds.lHP)
        return lHT, lHP
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
    base_dir: Path, device: torch.device, dims: List[int]
) -> Tuple[List[int], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Poll models and collect eigenvalue data.
    
    Returns 7 lists:
    - d_list: dimensions
    - emp_top_list: empirical top eigenvalues
    - emp_top_err_list: empirical top eigenvalue stds
    - emp_rest_list: empirical mean of rest eigenvalues
    - emp_rest_err_list: empirical std of rest eigenvalues
    - th_top_list: theoretical top eigenvalues (lHT)
    - th_perp_list: theoretical perpendicular eigenvalues (lHP)
    """
    runs = find_run_dirs(base_dir, dims)
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
        model = load_model(run_dir, cfg, device)
        if not model:
            continue
        
        mean_top, std_top, mean_rest, std_rest = compute_empirical_lH(model, d, device)
        lHT, lHP = compute_theoretical_lH(run_dir, d, device)

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
        th_top_list.append(lHT)
        th_perp_list.append(lHP)
    
    return d_list, emp_top_list, emp_top_err_list, emp_rest_list, emp_rest_err_list, th_top_list, th_perp_list


def main():
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "poll_lH_vs_d.png"
    slopes_path = base_dir / "plots" / "slope_history.json"
    slopes_plot_path = base_dir / "plots" / "slopes_over_time.png"
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
            d_vals, emp_top, emp_top_err, emp_rest, emp_rest_err, th_top, th_perp = poll_and_plot_once(
                base_dir, device, DIMS_DEFAULT
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

            else:
                print("No matching d{d}_P{3d}_N50_chi50 directories found yet.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping polling.")


if __name__ == "__main__":
    main()
