#!/usr/bin/env python3
"""
Poll erf standard-scaling d-sweep runs, compute empirical eigenvalues, compare to theory, and plot log-log λ_H vs d with slopes over time.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleErf
from Experiment import Experiment

POLL_INTERVAL = 5.0
DIMS_DEFAULT = [2, 6, 8, 10]
DEVICE_DEFAULT = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def arcsin_kernel(X):
    """Compute arcsin kernel matrix as provided in the snippet."""
    d = X.shape[1]
    XXT = torch.einsum('ui,vi->uv', X, X) 
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    return (2 / torch.pi) * torch.arcsin(arg)

def arcsin_gpr_predict(X, Y, sigma0_sq):
    """Simple GP prediction with arcsin kernel on training data."""
    J = arcsin_kernel(X)
    n = J.shape[0]
    K = J + sigma0_sq * torch.eye(n, device=X.device)
    alpha = torch.linalg.solve(K, Y)
    return J @ alpha


def find_run_dirs(base: Path, dims: List[int]) -> List[Tuple[Path, Dict[str, int]]]:
    runs: List[Tuple[Path, Dict[str, int]]] = []
    for d in dims:
        name = f"d{d}_P{3*d}_N256_chi1"
        p = base / name
        if p.is_dir():
            runs.append((p, {"d": d, "P": 3 * d, "N": 256, "chi": 1}))
    runs.sort(key=lambda x: x[1]["d"])
    return runs


def load_model(run_dir: Path, cfg: Dict, device: torch.device) -> Optional[FCN3NetworkEnsembleErf]:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(cfg["d"])
    P = int(cfg["P"])
    N = int(cfg["N"])
    chi = int(cfg.get("chi", 1))

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkEnsembleErf(d=d, n1=N, n2=N, P=P, ens=50,
                                   weight_initialization_variance=weight_var).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_empirical_lH(model: FCN3NetworkEnsembleErf, d: int, device: torch.device) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        with torch.no_grad():
            Xinf = torch.randn(3000, d, device=device)
            eig_mean, eig_std = model.H_eig(Xinf, Xinf, std=True)
            eig_mean_np = eig_mean.detach().cpu().numpy()
            eig_std_np = None if eig_std is None else eig_std.detach().cpu().numpy()

            top_idx = int(np.argmax(eig_mean_np))
            mean_top = float(eig_mean_np[top_idx])
            std_top = None if eig_std_np is None else float(eig_std_np[top_idx])

            rest_mask = np.ones(len(eig_mean_np), dtype=bool)
            rest_mask[top_idx] = False
            rest_eigs = eig_mean_np[rest_mask]
            mean_rest = float(np.mean(rest_eigs)) if len(rest_eigs) > 0 else None
            std_rest = float(np.std(rest_eigs)) if len(rest_eigs) > 0 else None
            return mean_top, std_top, mean_rest, std_rest
    except Exception as e:
        print(f"  Warning: empirical lH failed: {e}")
        return None, None, None, None


def compute_theoretical_lH(run_dir: Path, d: int, device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    try:
        P = 3 * d
        N = 256
        chi = 1.0
        kappa = 1.0
        exp = Experiment(file=str(run_dir), N=N, d=d, chi=chi, P=P, ens=50, kappa=kappa, device=device)
        preds = exp.eig_predictions()
        lH1T = float(preds.lH1T) if hasattr(preds, 'lH1T') and preds.lH1T is not None else None
        lH1P = float(preds.lH1P) if hasattr(preds, 'lH1P') and preds.lH1P is not None else None
        return lH1T, lH1P
    except Exception as e:
        print(f"  Warning: theoretical lH failed: {e}")
        return None, None


def plot_gpr_vs_network(model: FCN3NetworkEnsembleErf, X: torch.Tensor, Y: torch.Tensor, 
                        d: int, kappa: float, out_path: Path, device: torch.device):
    """Plot arcsin GPR prediction vs network outputs with error bars."""
    try:
        with torch.no_grad():
            sigma0_sq = 2 * kappa
            gpr_pred = arcsin_gpr_predict(X, Y, sigma0_sq).cpu().numpy().ravel()
            model_out_full = model.forward(X).detach().cpu().numpy()
            model_out = model_out_full.mean(axis=-1).ravel()
            model_std = model_out_full.std(axis=-1).ravel()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.errorbar(gpr_pred, model_out, yerr=model_std, fmt='o', markersize=6, alpha=0.6,
                       elinewidth=1.2, capsize=2.5, label="model ± std")
            mn = min(gpr_pred.min(), (model_out - model_std).min())
            mx = max(gpr_pred.max(), (model_out + model_std).max())
            ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.4, label="y = x")
            
            slope, intercept = np.polyfit(gpr_pred, model_out, 1)
            x_line = np.linspace(mn, mx, 200)
            ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.4,
                   label=f"best fit (slope={slope:.4f})")
            
            ax.set_xlabel("Arcsin GPR prediction")
            ax.set_ylabel("Network output (mean)")
            ax.set_title(f"GPR vs Network (d={d}, P={3*d}, N=256, κ={kappa})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(out_path), dpi=150)
            plt.close(fig)
            print(f"  Saved GPR vs network plot to {out_path}")
            return slope
    except Exception as e:
        print(f"  Warning: GPR vs network plot failed: {e}")
        return None


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


def update_plot(d_vals, emp_top, emp_top_err, emp_rest, emp_rest_err, th_top, th_perp, out_path):
    d_arr = np.array(d_vals, dtype=float)
    emp_top_arr = np.array([v if v is not None else np.nan for v in emp_top], dtype=float)
    emp_top_err_arr = np.array([v if v is not None else np.nan for v in emp_top_err], dtype=float)
    emp_rest_arr = np.array([v if v is not None else np.nan for v in emp_rest], dtype=float)
    emp_rest_err_arr = np.array([v if v is not None else np.nan for v in emp_rest_err], dtype=float)
    th_top_arr = np.array([v if v is not None else np.nan for v in th_top], dtype=float)
    th_perp_arr = np.array([v if v is not None else np.nan for v in th_perp], dtype=float)

    slope_emp_top = line_slope_loglog(d_arr, emp_top_arr)
    slope_th_top = line_slope_loglog(d_arr, th_top_arr)
    slope_emp_rest = line_slope_loglog(d_arr, emp_rest_arr)
    slope_th_perp = line_slope_loglog(d_arr, th_perp_arr)

    fig, ax = plt.subplots(figsize=(11, 7))

    if np.any(~np.isnan(emp_top_err_arr)):
        ax.errorbar(d_arr, emp_top_arr, yerr=emp_top_err_arr, fmt='o', color='tab:blue', ecolor='tab:blue',
                    elinewidth=1.2, capsize=3,
                    label=f"Empirical $\\lambda_H^{{(*)}}$ ± std (slope={slope_emp_top:.3f})" if slope_emp_top is not None else "Empirical $\\lambda_H^{(*)}$ ± std")
        ax.plot(d_arr, emp_top_arr, '-', color='tab:blue', linewidth=2.0)
    else:
        ax.loglog(d_arr, emp_top_arr, 'o-', color='tab:blue', linewidth=2.5, markersize=8,
                  label=f"Empirical $\\lambda_H^{{(*)}}$ (slope={slope_emp_top:.3f})" if slope_emp_top is not None else "Empirical $\\lambda_H^{(*)}$")

    ax.loglog(d_arr, th_top_arr, 's--', color='tab:red', linewidth=2.0, markersize=7,
              label=f"Theory $\\lambda_H^{{(*)}}$ (slope={slope_th_top:.3f})" if slope_th_top is not None else "Theory $\\lambda_H^{(*)}$")

    if np.any(~np.isnan(emp_rest_err_arr)):
        ax.errorbar(d_arr, emp_rest_arr, yerr=emp_rest_err_arr, fmt='^', color='tab:green', ecolor='tab:green',
                    elinewidth=1.2, capsize=3, alpha=0.7,
                    label=f"Empirical mean(rest) ± std (slope={slope_emp_rest:.3f})" if slope_emp_rest is not None else "Empirical mean(rest)")
        ax.plot(d_arr, emp_rest_arr, '-', color='tab:green', linewidth=2.0, alpha=0.7)
    else:
        ax.loglog(d_arr, emp_rest_arr, '^-', color='tab:green', linewidth=2.5, markersize=8, alpha=0.7,
                  label=f"Empirical mean(rest) (slope={slope_emp_rest:.3f})" if slope_emp_rest is not None else "Empirical mean(rest)")

    ax.loglog(d_arr, th_perp_arr, 'D--', color='tab:orange', linewidth=2.0, markersize=7, alpha=0.7,
              label=f"Theory $\\lambda_H^{{(\\perp)}}$ (slope={slope_th_perp:.3f})" if slope_th_perp is not None else "Theory $\\lambda_H^{(\\perp)}$")

    ax.set_xlabel("Dimension $d$ (log scale)")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Empirical ($\\lambda_H^{(*)}$) vs Theory ($\\lambda_H^{(\\perp)}$) — Erf Standard Scaling (chi=1)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    if len(d_arr) > 0:
        ax.set_xlim(left=d_arr[0] * 0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

    return slope_emp_top, slope_th_top, slope_emp_rest, slope_th_perp


def poll_and_plot_once(base_dir: Path, device: torch.device, dims: List[int]):
    runs = find_run_dirs(base_dir, dims)
    d_list: List[int] = []
    emp_top_list: List[Optional[float]] = []
    emp_top_err_list: List[Optional[float]] = []
    emp_rest_list: List[Optional[float]] = []
    emp_rest_err_list: List[Optional[float]] = []
    th_top_list: List[Optional[float]] = []
    th_perp_list: List[Optional[float]] = []

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for run_dir, cfg in runs:
        model = load_model(run_dir, cfg, device)
        if not model:
            continue
        mean_top, std_top, mean_rest, std_rest = compute_empirical_lH(model, cfg["d"], device)
        lH1T, lH1P = compute_theoretical_lH(run_dir, cfg["d"], device)

        # Generate training data (standard Gaussian) and target (x0 + eps*He3(x0))
        d = cfg["d"]
        P = cfg["P"]
        eps = 0.03
        torch.manual_seed(42)  # Use same seed for consistency
        X = torch.randn(P, d, device=device)
        x0 = X[:, 0]
        def hermite3(x):
            return (x**3 - 3.0 * x) / np.sqrt(6.0)
        Y = x0 + eps * hermite3(x0)
        
        kappa = 1.0
        gpr_plot_path = plots_dir / f"gpr_vs_network_d{d}_n_{cfg['N']}.png"
        plot_gpr_vs_network(model, X, Y, d, kappa, gpr_plot_path, device)

        d_list.append(cfg["d"])
        emp_top_list.append(mean_top)
        emp_top_err_list.append(std_top)
        emp_rest_list.append(mean_rest)
        emp_rest_err_list.append(std_rest)
        th_top_list.append(lH1T)
        th_perp_list.append(lH1P)

    return d_list, emp_top_list, emp_top_err_list, emp_rest_list, emp_rest_err_list, th_top_list, th_perp_list


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "poll_lH_vs_d.png"
    slopes_path = plots_dir / "slope_history.json"
    slopes_plot_path = plots_dir / "slopes_over_time.png"
    device = DEVICE_DEFAULT

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

    print("Polling d-sweep runs (standard scaling, erf) and updating plots...")
    print(f"Base: {base_dir}")
    print(f"Output: {out_path}")

    try:
        while True:
            d_vals, emp_top, emp_top_err, emp_rest, emp_rest_err, th_top, th_perp = poll_and_plot_once(base_dir, device, DIMS_DEFAULT)
            if d_vals:
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
                    plt.title(r'Eigenvalue Scaling Exponents vs Time: $\alpha(t)$ where $\lambda \sim d^\alpha$ (chi=1)')
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    try:
                        plt.savefig(slopes_plot_path, dpi=200)
                        print(f"Saved slope history plot to {slopes_plot_path}")
                    finally:
                        plt.close()
            else:
                print("No matching d{d}_P{3d}_N256_chi1 directories found yet.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping polling.")


if __name__ == "__main__":
    main()
