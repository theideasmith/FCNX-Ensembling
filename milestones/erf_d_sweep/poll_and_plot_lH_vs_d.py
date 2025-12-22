#!/usr/bin/env python3
"""
Poll erf d-sweep models periodically, compute empirical eigenvalues for linear (Y1) and
cubic (Y3) eigenfunctions, compare with theoretical predictions, and plot vs d.

- Scans directories: d{d}_P{3d}_N50_chi50
- Loads current model (prefers model.pt, falls back to model_final.pt)
- Empirical: Y1 via H_eig; Y3 via QB+SVD projection of Hessian surrogate
- Theory: Experiment.eig_predictions() for lH1T, lH3T
- Plots log-log lH1/lH3 vs d with slopes; tracks slope history over time
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

# Workspace libs
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleErf
from Experiment import Experiment  # noqa: F401 imported for side effects/config


POLL_INTERVAL = 10.0  # seconds
DIMS_DEFAULT = [4, 6, 8, 10]
DEVICE_DEFAULT = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: List[int]) -> List[Tuple[Path, Dict[str, int]]]:
    """Select erf run folders named d{d}_P{3d}_N50_chi50 for d in dims."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    for d in dims:
        name = f"d{d}_P{10*d}_N50_chi50"
        p = base / name
        if p.is_dir():
            selected.append((p, {"d": d, "P": 3 * d, "N": 50, "chi": 50}))
    selected.sort(key=lambda x: x[1]["d"])
    return selected


def load_config_from_name(cfg: Dict[str, int]) -> Dict[str, int]:
    return cfg


def load_model(run_dir: Path, config: Dict, device: torch.device) -> Optional[FCN3NetworkEnsembleErf]:
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
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=N,
        n2=N,
        P=P,
        ens=10,
        weight_initialization_variance=weight_var,
    ).to(device)
    model.device = device
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_empirical_lH(
    model: FCN3NetworkEnsembleErf, d: int, device: torch.device
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute eigenvalues for linear (Y1) and cubic (Y3) eigenfunctions on current device."""
    try:
        with torch.no_grad():
            X = torch.randn(3000, d, device=device)
            
            # Randomized QB decomposition for Hessian surrogate
            Q, Z = model.H_random_QB(X, k=200, p=10)
            Ut, _S, V = torch.linalg.svd(Z.T)
            U = torch.matmul(Q, Ut)

            # Linear eigenfunctions: Y1 = X
            Y1 = X
            Y1_norm = Y1 / torch.norm(Y1, dim=0)
            left_eigenvaluesY1 = (
                torch.matmul(Y1_norm.t(), U)
                @ _S.diag()
                @ torch.matmul(U.T, Y1_norm)
            ).diagonal() / torch.norm(Y1_norm, dim=0) / X.shape[0]

            # Cubic eigenfunctions: Y3 = (X^3 - 3X)/6
            Y3 = (X ** 3 - 3 * X) / 6.0
            Y3_norm = Y3 / torch.norm(Y3, dim=0)
            left_eigenvaluesY3 = (
                torch.matmul(Y3_norm.t(), U)
                @ _S.diag()
                @ torch.matmul(U.T, Y3_norm)
            ).diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]

            lH1_np = left_eigenvaluesY1.detach().cpu().numpy()
            print(f"  lH1_np: {lH1_np}")
            lH3_np = left_eigenvaluesY3.detach().cpu().numpy()

            lH1_mean = float(np.max(lH1_np)) if len(lH1_np) > 0 else None
            lH3_mean = float(np.max(lH3_np)) if len(lH3_np) > 0 else None
            lH1_std = float(np.std(lH1_np)) if len(lH1_np) > 1 else None
            lH3_std = float(np.std(lH3_np)) if len(lH3_np) > 1 else None

            return lH1_mean, lH1_std, lH3_mean, lH3_std
    except Exception as e:  # pragma: no cover - observational script
        print(f"  Warning: failed to compute empirical lH: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None


def compute_theoretical_lH(run_dir: Path, d: int, device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    """Compute theoretical lH1T and lH3T using Experiment.eig_predictions()."""
    try:
        P = 3 * d
        N = 50
        chi = 50
        kappa = 1.0 / chi
        
        # Create Experiment instance
        exp = Experiment(
            file=str(run_dir),
            N=N,
            d=d,
            chi=chi,
            P=P,
            ens=10,
            eps=0.03,
            device=device,
        )
        
        # Get predictions from Experiment
        preds = exp.eig_predictions()
        
        # Extract linear and cubic eigenvalues
        lH1T = None
        lH3T = None
        if hasattr(preds, 'lH1T') and preds.lH1T is not None:
            lH1T = float(preds.lH1T)
        if hasattr(preds, 'lH3T') and preds.lH3T is not None:
            lH3T = float(preds.lH3T)
        
        print(f"  Theoretical lH for d={d}: lH1T={lH1T}, lH3T={lH3T}")
        return lH1T, lH3T
    except Exception as e:
        print(f"  Warning: failed to compute theoretical lH for d={d}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def line_slope_loglog(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or ys.size < 2:
        return None
    X = np.log(xs)
    Y = np.log(ys)
    A = np.vstack([X, np.ones_like(X)]).T
    m, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m)


def update_plot(
    d_vals: List[int],
    emp_lH1: List[Optional[float]],
    emp_lH1_err: List[Optional[float]],
    emp_lH3: List[Optional[float]],
    emp_lH3_err: List[Optional[float]],
    th_lH1: List[Optional[float]],
    th_lH3: List[Optional[float]],
    out_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    d_arr = np.array(d_vals, dtype=float)
    emp_lH1_arr = np.array([v if v is not None else np.nan for v in emp_lH1], dtype=float)
    emp_lH1_err_arr = np.array([e if e is not None else np.nan for e in emp_lH1_err], dtype=float)
    emp_lH3_arr = np.array([v if v is not None else np.nan for v in emp_lH3], dtype=float)
    emp_lH3_err_arr = np.array([e if e is not None else np.nan for e in emp_lH3_err], dtype=float)
    th_lH1_arr = np.array([v if v is not None else np.nan for v in th_lH1], dtype=float)
    th_lH3_arr = np.array([v if v is not None else np.nan for v in th_lH3], dtype=float)

    slope_emp_lH1 = line_slope_loglog(d_arr, emp_lH1_arr)
    slope_th_lH1 = line_slope_loglog(d_arr, th_lH1_arr)
    slope_emp_lH3 = line_slope_loglog(d_arr, emp_lH3_arr)
    slope_th_lH3 = line_slope_loglog(d_arr, th_lH3_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear
    ax = axes[0]
    if np.any(~np.isnan(emp_lH1_err_arr)):
        ax.errorbar(
            d_arr,
            emp_lH1_arr,
            yerr=emp_lH1_err_arr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.2,
            capsize=3,
            label=(
                f"Empirical $\\lambda_H^{(1)}$ ± std (slope={slope_emp_lH1:.3f})"
                if slope_emp_lH1 is not None
                else "Empirical $\\lambda_H^{(1)}$ ± std"
            ),
        )
        ax.plot(d_arr, emp_lH1_arr, "-", color="tab:blue", linewidth=2.0)
    else:
        ax.loglog(
            d_arr,
            emp_lH1_arr,
            "o-",
            color="tab:blue",
            linewidth=2.5,
            markersize=8,
            label=(
                f"Empirical $\\lambda_H^{(1)}$ (slope={slope_emp_lH1:.3f})"
                if slope_emp_lH1 is not None
                else "Empirical $\\lambda_H^{(1)}$"
            ),
        )
    ax.loglog(
        d_arr,
        th_lH1_arr,
        "s--",
        color="tab:red",
        linewidth=2.0,
        markersize=7,
        label=(
            f"Theory $\\lambda_H^{(1)}$ (slope={slope_th_lH1:.3f})"
            if slope_th_lH1 is not None
            else "Theory $\\lambda_H^{(1)}$"
        ),
    )
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Eigenvalue (log scale)")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Linear eigenvalues vs $d$\n(Updated: {timestamp})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")

    # Cubic
    ax = axes[1]
    if np.any(~np.isnan(emp_lH3_err_arr)):
        ax.errorbar(
            d_arr,
            emp_lH3_arr,
            yerr=emp_lH3_err_arr,
            fmt="^",
            color="tab:green",
            ecolor="tab:green",
            elinewidth=1.2,
            capsize=3,
            label=(
                f"Empirical $\\lambda_H^{(3)}$ ± std (slope={slope_emp_lH3:.3f})"
                if slope_emp_lH3 is not None
                else "Empirical $\\lambda_H^{(3)}$ ± std"
            ),
        )
        ax.plot(d_arr, emp_lH3_arr, "-", color="tab:green", linewidth=2.0)
    else:
        ax.loglog(
            d_arr,
            emp_lH3_arr,
            "^-",
            color="tab:green",
            linewidth=2.5,
            markersize=8,
            label=(
                f"Empirical $\\lambda_H^{(3)}$ (slope={slope_emp_lH3:.3f})"
                if slope_emp_lH3 is not None
                else "Empirical $\\lambda_H^{(3)}$"
            ),
        )
    ax.loglog(
        d_arr,
        th_lH3_arr,
        "D--",
        color="tab:orange",
        linewidth=2.0,
        markersize=7,
        label=(
            f"Theory $\\lambda_H^{(3)}$ (slope={slope_th_lH3:.3f})"
            if slope_th_lH3 is not None
            else "Theory $\\lambda_H^{(3)}$"
        ),
    )
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Cubic eigenvalues vs $d$\n(Updated: {timestamp})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[{timestamp}] Saved plot to {out_path}")
    print(f"  Empirical slopes: lH1={slope_emp_lH1:.4f}, lH3={slope_emp_lH3:.4f}")
    print(f"  Empirical eigenvalues: lH1={emp_lH1_arr}, lH3={emp_lH3_arr}")

    return slope_emp_lH1, slope_th_lH1, slope_emp_lH3, slope_th_lH3


def poll_and_plot_once(
    base_dir: Path, device: torch.device, dims: List[int]
) -> Tuple[
    List[int],
    List[Optional[float]],
    List[Optional[float]],
    List[Optional[float]],
    List[Optional[float]],
    List[Optional[float]],
    List[Optional[float]],
]:
    runs = find_run_dirs(base_dir, dims)
    d_list: List[int] = []
    emp_lH1_list: List[Optional[float]] = []
    emp_lH1_err_list: List[Optional[float]] = []
    emp_lH3_list: List[Optional[float]] = []
    emp_lH3_err_list: List[Optional[float]] = []
    th_lH1_list: List[Optional[float]] = []
    th_lH3_list: List[Optional[float]] = []

    for run_dir, cfg in runs:
        cfg = load_config_from_name(cfg)
        d = cfg["d"]
        model = load_model(run_dir, cfg, device)
        if not model:
            continue

        lH1_mean, lH1_std, lH3_mean, lH3_std = compute_empirical_lH(model, d, device)
        lH1T, lH3T = compute_theoretical_lH(run_dir, d, device)

        d_list.append(d)
        emp_lH1_list.append(lH1_mean)
        emp_lH1_err_list.append(lH1_std)
        emp_lH3_list.append(lH3_mean)
        emp_lH3_err_list.append(lH3_std)
        th_lH1_list.append(lH1T)
        th_lH3_list.append(lH3T)

    return d_list, emp_lH1_list, emp_lH1_err_list, emp_lH3_list, emp_lH3_err_list, th_lH1_list, th_lH3_list


def compute_theoretical_slope_from_range(d_range: List[int], device: torch.device, base_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """Compute theoretical slopes using predictions from a range of d values."""
    th_lH1_vals = []
    th_lH3_vals = []
    d_vals = []
    
    for d in d_range:
        P = 3 * d
        N = 50
        chi = 50
        # Use base_dir as dummy run_dir since we don't need actual model
        lH1T, lH3T = compute_theoretical_lH(base_dir, d, device)
        if lH1T is not None and lH3T is not None:
            d_vals.append(d)
            th_lH1_vals.append(lH1T)
            th_lH3_vals.append(lH3T)
    
    if len(d_vals) < 2:
        return None, None
    
    d_arr = np.array(d_vals, dtype=float)
    th_lH1_arr = np.array(th_lH1_vals, dtype=float)
    th_lH3_arr = np.array(th_lH3_vals, dtype=float)
    
    slope_th_lH1 = line_slope_loglog(d_arr, th_lH1_arr)
    slope_th_lH3 = line_slope_loglog(d_arr, th_lH3_arr)
    
    return slope_th_lH1, slope_th_lH3


def main():
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "poll_lH_vs_d.png"
    slopes_path = base_dir / "plots" / "slope_history.json"
    slopes_plot_path = base_dir / "plots" / "slopes_over_time.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = DEVICE_DEFAULT

    # Compute theoretical slopes once using 10 d values from 2 to 50
    print("Computing theoretical slopes from d=2 to d=50...")
    d_theory_range = [4,6,8,10]#np.linspace(10, 100, 10, dtype=int).tolist()
    slope_th_lH1_fixed, slope_th_lH3_fixed = compute_theoretical_slope_from_range(d_theory_range, device, base_dir)
    print(f"Theoretical slopes: lH1={slope_th_lH1_fixed:.4f}, lH3={slope_th_lH3_fixed:.4f}")

    slope_hist_emp_lH1: List[float] = []
    slope_hist_emp_lH3: List[float] = []
    try:
        if slopes_path.exists():
            with open(slopes_path, "r") as f:
                data = json.load(f)
                slope_hist_emp_lH1 = data.get("empirical_lH1", [])
                slope_hist_emp_lH3 = data.get("empirical_lH3", [])
    except Exception:
        pass

    print("Polling d-sweep runs (erf) and updating lH vs d plot...")
    print(f"Base: {base_dir}")
    print(f"Output: {out_path}")

    try:
        while True:
            d_vals, emp_lH1, emp_lH1_err, emp_lH3, emp_lH3_err, th_lH1, th_lH3 = poll_and_plot_once(
                base_dir, device, DIMS_DEFAULT
            )

            if d_vals:
                order = np.argsort(d_vals)
                d_vals = list(np.array(d_vals)[order])
                emp_lH1 = list(np.array(emp_lH1, dtype=object)[order])
                emp_lH1_err = list(np.array(emp_lH1_err, dtype=object)[order])
                emp_lH3 = list(np.array(emp_lH3, dtype=object)[order])
                emp_lH3_err = list(np.array(emp_lH3_err, dtype=object)[order])
                th_lH1 = list(np.array(th_lH1, dtype=object)[order])
                th_lH3 = list(np.array(th_lH3, dtype=object)[order])

                slope_emp_lH1, slope_th_lH1, slope_emp_lH3, slope_th_lH3 = update_plot(
                    d_vals, emp_lH1, emp_lH1_err, emp_lH3, emp_lH3_err, th_lH1, th_lH3, out_path
                )

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

                changed_emp_lH1 = maybe_append(slope_hist_emp_lH1, slope_emp_lH1)
                changed_emp_lH3 = maybe_append(slope_hist_emp_lH3, slope_emp_lH3)

                if changed_emp_lH1 or changed_emp_lH3:
                    try:
                        with open(slopes_path, "w") as f:
                            json.dump(
                                {
                                    "empirical_lH1": slope_hist_emp_lH1,
                                    "empirical_lH3": slope_hist_emp_lH3,
                                },
                                f,
                                indent=2,
                            )
                    except Exception as e:  # pragma: no cover - observational script
                        print(f"Warning: failed to save slope history: {e}")

                if any([slope_hist_emp_lH1, slope_hist_emp_lH3]):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                    if slope_hist_emp_lH1:
                        ax1.plot(
                            range(1, len(slope_hist_emp_lH1) + 1),
                            slope_hist_emp_lH1,
                            "-o",
                            color="tab:blue",
                            label="Empirical lH1 slope",
                        )
                    # Use the same theoretical slope computed from actual d values in update_plot
                    if slope_th_lH1 is not None:
                        ax1.axhline(
                            y=slope_th_lH1,
                            color="tab:red",
                            linestyle="--",
                            linewidth=2,
                            label=f"Theory lH1 slope = {slope_th_lH1:.3f}",
                        )
                    ax1.set_xlabel("Measurement index")
                    ax1.set_ylabel("Slope (log-log)")
                    ax1.set_title("Linear (Y1) slope history")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc="best")

                    if slope_hist_emp_lH3:
                        ax2.plot(
                            range(1, len(slope_hist_emp_lH3) + 1),
                            slope_hist_emp_lH3,
                            "-o",
                            color="tab:green",
                            label="Empirical lH3 slope",
                        )
                    # Use the same theoretical slope computed from actual d values in update_plot
                    if slope_th_lH3 is not None:
                        ax2.axhline(
                            y=slope_th_lH3,
                            color="tab:orange",
                            linestyle="--",
                            linewidth=2,
                            label=f"Theory lH3 slope = {slope_th_lH3:.3f}",
                        )
                    ax2.set_xlabel("Measurement index")
                    ax2.set_ylabel("Slope (log-log)")
                    ax2.set_title("Cubic (Y3) slope history")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc="best")

                    fig.tight_layout()
                    try:
                        fig.savefig(slopes_plot_path, dpi=200)
                        print(f"Saved slope history plot to {slopes_plot_path}")
                    finally:
                        plt.close(fig)

            else:
                print("No matching d{d}_P{3d}_N50_chi50 directories found yet.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping polling.")


if __name__ == "__main__":
    main()
