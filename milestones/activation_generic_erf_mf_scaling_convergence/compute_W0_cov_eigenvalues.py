#!/usr/bin/env python3
"""Compute eigenvalues of the W0 weight covariance for each ensemble member.

For each run (d<d>_P<P>_N<N>_chi<chi>/seed<seed>/model.pt), load the model,
compute per-ensemble covariance eigenvalues of W0 (shape: ens x N x d), and
save the eigenvalues to <run>/plots/W0_cov_eigenvalues.npy. Also prints basic
stats (mean/min/max) for quick inspection.

Usage:
    python compute_W0_cov_eigenvalues.py --base-dir <runs_root> [--dims 150] [--suffix "_tag"]
"""

import argparse
import re
import subprocess
import tempfile
import json as json_lib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import sys
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric
from Experiment import Experiment

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def compute_kappa_eff(d: int, P: int, kappa: float, n_samples: int = 5000):
    """Compute effective ridge by running self-consistent kappa solver using arcsin kernel eigenvalues."""
    try:
        # Compute n_samples x n_samples arcsin kernel eigenvalues
        np.random.seed(0)
        X = np.random.randn(n_samples, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / n_samples
        
        # Run self-consistent solver
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        eigenvalues = eigvals.tolist()
        with open(eig_json, "w") as f:
            json_lib.dump({"eigenvalues": eigenvalues, "kappa_bare": kappa}, f)
        
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
        sc_cmd = [
            "julia", str(julia_script),
            eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True)
        
        # Extract kappa_eff from output
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
            return kappa_eff
        else:
            return kappa
    except Exception as e:
        print(f"Warning: kappa_eff computation failed: {e}. Using bare kappa.")
        return kappa
    finally:
        try:
            if 'eig_json' in locals() and Path(eig_json).exists():
                Path(eig_json).unlink()
        except:
            pass


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories that contain model.pt and match the naming convention."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?).*"
    model_files = glob.glob(str(base / f"**/*seed*/model.pt"), recursive=True) 
    print("Searching for model.pt files in", base)
    for model_file in model_files:

        seed_dir = Path(model_file).parent
        print(seed_dir)
        seed_name = seed_dir.name
        m_seed = re.match(r".*seed(\d+)", seed_name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        cfg_dir = seed_dir.parent
        cfg_name = cfg_dir.name
        m_cfg = re.match(pattern, cfg_name)
        if not m_cfg:
            continue

        d = int(m_cfg.group(1))
        if dims and d not in dims:
            continue
        P = int(m_cfg.group(2))
        N = int(m_cfg.group(3))
        chi = int(float(m_cfg.group(4)))

        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))

    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    print(f"Found {len(selected)} runs with model.pt")
    return selected


def load_model(run_dir: Path, config: Dict[str, int], device: torch.device) -> Optional[FCN3NetworkActivationGeneric]:
    """Load model from run directory."""
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(config.get("d"))
    P = int(config.get("P"))
    N = int(config.get("N"))
    chi = int(config.get("chi", N))

    state_dict = torch.load(model_path, map_location=device)


    if len(state_dict['W0'].shape) == 4:
        # 
        state_dict['W0'] = state_dict['W0'].squeeze(0)
        state_dict['W1'] = state_dict['W1'].squeeze(0)
        state_dict['A'] = state_dict['A'].squeeze(0)
    ens = int(state_dict['W0'].shape[0])
    print(state_dict['W0'].shape)
    print("Loading model from", model_path, f"with W0 shape {state_dict['W0'].shape} and ensemble size {ens}")
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return model


def langevin_avg_W0(model: FCN3NetworkActivationGeneric, d: int, N: int, chi: float, 
                    kappa: float, num_epochs: int = 200000, lr0: float = 3e-5, 
                    device: torch.device = DEVICE_DEFAULT) -> Dict[str, np.ndarray]:
    """Run Langevin dynamics on model to average W0, using implementation from d_sweep.py.
    
    Applies the same Langevin update as d_sweep.py:
    - Gradient descent with weight decay (temperature-dependent)
    - Langevin noise proportional to sqrt(2 * lr * temperature)
    
    Tracks eigenvalues of W0 covariance every 1000 steps to monitor convergence.
    
    Args:
        model: FCN3 model to evolve
        d: Input dimension
        N: Hidden layer dimension
        chi: Chi parameter for computing temperature
        kappa: Kappa parameter for computing temperature
        num_epochs: Number of Langevin steps
        lr0: Base learning rate
        device: Device to use
    
    Returns:
        Dictionary with keys:
        - 'W0_avg_pooled': Eigenvalues of covariance of averaged W0 (d,) using pooled method
        - 'W0_avg_per_ens': Eigenvalues of covariance of averaged W0 (ens, d) using per-ensemble method
        - 'W0_cov_avg_pooled': Mean of pooled covariance eigenvalues sampled every 1000 steps (d,)
        - 'W0_cov_stderr_pooled': Std error of pooled covariance eigenvalues (d,)
        - 'W0_cov_avg_per_ens': Mean of per-ensemble covariance eigenvalues sampled every 1000 steps (d,)
        - 'W0_cov_stderr_per_ens': Std error of per-ensemble covariance eigenvalues (d,)
        - 'num_samples': Number of Langevin samples used for stats (int)
    """
    from pathlib import Path
    
    model.train()
    ens = model.ens
    temperature = 2 * kappa / chi
    
    # Weight decay coefficients (from d_sweep.py)
    wd_fc1 = d * temperature  # for W0
    
    # Streaming accumulator for W0
    W0_sum = torch.zeros_like(model.W0)  # (ens, N, d)
    
    # Track covariance eigenvalues every 1000 steps (both methods)
    W0_cov_eigs_pooled_list = []
    W0_cov_eigs_per_ens_list = []
    
    print(f"Running Langevin dynamics for {num_epochs} epochs (temperature={temperature:.4f}, lr0={lr0})...")
    print(f"  Weight decay (W0): {wd_fc1:.6e}")
    print(f"  Computing covariance eigenvalues every 1000 steps (pooled and per-ens methods)...")
    
    for epoch in tqdm(range(num_epochs), desc="Langevin dynamics"):
        # Use simple Langevin dynamics without gradient (just weight decay + noise)
        # This corresponds to sampling from p(W0) ~ exp(-wd_fc1/2 * ||W0||^2)
        
        lr = 1e-2/800
        noise_scale = np.sqrt(2.0 * lr * temperature)
        
        with torch.no_grad():
            # Langevin update: W0 <- W0 - lr * weight_decay * W0 + noise
            # (no gradient, just weight decay regularization)
            noise = torch.randn_like(model.W0) * noise_scale
            model.W0.data.add_(-lr * wd_fc1 * model.W0.data)
            model.W0.data.add_(noise)
        
        # Stream sum for averaging
        W0_sum += model.W0.detach().clone()
        
        # Every 1000 steps, compute and store covariance eigenvalues (both methods)
        if (epoch + 1) % 1000 == 0:
            W0_cov_eigs_pooled = compute_W0_eigenvalues_from_tensor(model.W0.detach().clone(), d, N)
            W0_cov_eigs_per_ens = compute_W0_eigenvalues_per_ensemble(model.W0.detach().clone(), d, N)
            # Reduce per-ensemble to a mean spectrum per sample
            per_ens_sorted = np.sort(W0_cov_eigs_per_ens, axis=1)[:, ::-1]
            per_ens_mean = per_ens_sorted.mean(axis=0)
            W0_cov_eigs_pooled_list.append(W0_cov_eigs_pooled)
            W0_cov_eigs_per_ens_list.append(per_ens_mean)
        
        if (epoch + 1) % 10000 == 0:
            w0_norm = (model.W0 ** 2).sum().sqrt().item()
            print(f"  Epoch {epoch+1:6d}: W0 norm = {w0_norm:.4f}")
    
    # Compute eigenvalues from averaged W0 (both methods)
    W0_avg = W0_sum / num_epochs
    W0_avg_eigs_pooled = compute_W0_eigenvalues_from_tensor(W0_avg, d, N)
    W0_avg_eigs_per_ens = compute_W0_eigenvalues_per_ensemble(W0_avg, d, N)
    
    def mean_and_stderr(samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
        samples_arr = np.array(samples)
        n_samples = samples_arr.shape[0]
        mean = samples_arr.mean(axis=0) if n_samples > 0 else np.zeros((d,))
        if n_samples > 1:
            stderr = samples_arr.std(axis=0, ddof=1) / np.sqrt(n_samples)
        else:
            stderr = np.zeros_like(mean)
        return mean, stderr, n_samples

    # Mean and stderr across Langevin samples (treat samples as an ensemble)
    W0_cov_eigs_avg_pooled, W0_cov_eigs_stderr_pooled, n_samples = mean_and_stderr(W0_cov_eigs_pooled_list)
    W0_cov_eigs_avg_per_ens, W0_cov_eigs_stderr_per_ens, _ = mean_and_stderr(W0_cov_eigs_per_ens_list)
    
    return {
        "W0_avg_pooled": W0_avg_eigs_pooled,
        "W0_avg_per_ens": W0_avg_eigs_per_ens,
        "W0_cov_avg_pooled": W0_cov_eigs_avg_pooled,
        "W0_cov_stderr_pooled": W0_cov_eigs_stderr_pooled,
        "W0_cov_avg_per_ens": W0_cov_eigs_avg_per_ens,
        "W0_cov_stderr_per_ens": W0_cov_eigs_stderr_per_ens,
        "num_samples": n_samples,
    }


def compute_W0_eigenvalues_from_tensor(W0_tensor: torch.Tensor, d: int, N: int) -> np.ndarray:
    """Compute eigenvalues of W0 covariance from a tensor using pooled ens*N samples.
    
    Follows the d_sweep.py method: reshape (ens, N, d) to (ens*N, d), compute global
    covariance matrix, and get eigenvalues.
    
    Args:
        W0_tensor: Weight matrix (ens, N, d)
        d: Input dimension
        N: Hidden layer dimension
    
    Returns:
        Eigenvalues array (d,) sorted in descending order
    """
    # W0 shape: (ens, N, d)
    W0_reshaped = W0_tensor.view(W0_tensor.shape[0] * N, d)  # shape: (ens*N, d)
    cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (W0_tensor.shape[0] * N)  # shape: (d, d)
    eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.detach().cpu().numpy()  # shape: (d,)
    return eigvals_W0


def compute_W0_eigenvalues_per_ensemble(W0_tensor: torch.Tensor, d: int, N: int) -> np.ndarray:
    """Compute eigenvalues of W0 covariance separately for each ensemble member.
    
    For each ensemble member e, compute covariance of W0[e] (shape N, d) and get eigenvalues.
    
    Args:
        W0_tensor: Weight matrix (ens, N, d)
        d: Input dimension
        N: Hidden layer dimension
    
    Returns:
        Eigenvalues array (ens, d) - one eigenvalue vector per ensemble member
    """
    W0 = W0_tensor.detach().cpu().numpy()  # (ens, N, d)
    ens = W0.shape[0]
    eigvals_list = []
    for e in range(ens):
        W0_e = W0[e]  # (N, d)
        cov = np.cov(W0_e, rowvar=False)  # (d, d)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_list.append(eigvals)
    return np.array(eigvals_list)  # (ens, d)


def compute_W0_cov_eigenvalues(model: FCN3NetworkActivationGeneric, d: int, N: int,
                               use_langevin_avg: bool = False, chi: float = None, 
                               kappa: float = None, device: torch.device = DEVICE_DEFAULT):
    """Return eigenvalues of W0 covariance using both pooled and per-ensemble methods.

    W0 shape: (ens, N, d)
    
    Args:
        model: FCN3 model
        d: Input dimension
        N: Hidden layer dimension
        use_langevin_avg: If True, run Langevin dynamics to average W0 before computing eigenvalues
        chi: Chi parameter (required if use_langevin_avg=True)
        kappa: Kappa parameter (required if use_langevin_avg=True)
        device: Device to use
    
    Returns:
        Dictionary with keys:
        - 'pooled': (d,) eigenvalues from pooled ens*N samples (d_sweep.py method)
        - 'per_ens': (ens, d) eigenvalues computed separately for each ensemble member
        - If use_langevin_avg=True, adds 'langevin' key with dict containing pooled and per-ensemble averages
    """
    eigvals_dict = {}
    
    # Compute both methods on current W0
    eigvals_dict['pooled'] = compute_W0_eigenvalues_from_tensor(model.W0, d, N)
    eigvals_dict['per_ens'] = compute_W0_eigenvalues_per_ensemble(model.W0, d, N)
    
    # If Langevin averaging requested, run it
    if use_langevin_avg:
        if chi is None or kappa is None:
            raise ValueError("chi and kappa required when use_langevin_avg=True")
        langevin_result = langevin_avg_W0(model, d, N, chi, kappa, device=device)
        eigvals_dict['langevin'] = langevin_result
    
    return eigvals_dict


def plot_W0_cov_eigenvalues(
    eigvals: np.ndarray,
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Bar plot of eigenvalues with theory lines (lWT, lWP).
    
    eigvals shape: (d,) - eigenvalue vector for pooled W0 samples
    Always displays:
    - Theoretical largest and bulk average
    - Empirical largest and bulk average with % error
    """
    try:
        eigvals_sorted = np.sort(eigvals)[::-1]  # Sort descending
        idx = np.arange(eigvals_sorted.shape[0])
        
        # Compute largest and bulk average empirically
        empirical_largest = eigvals_sorted[0] if eigvals_sorted.shape[0] > 0 else 0.0
        empirical_bulk_avg = eigvals_sorted[1:].mean() if eigvals_sorted.shape[0] > 1 else 0.0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(idx, eigvals_sorted, color='tab:blue', alpha=0.7, label='eigenvalue')
        
        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / empirical_largest)) * 100.0 if empirical_largest > 0 else 0.0
            ax.axhline(theory_lwt, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWT={theory_lwt:.3e} (emp: {empirical_largest:.3e}, err: {err_pct_wt:+.1f}%)')
        
        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / empirical_bulk_avg)) * 100.0 if empirical_bulk_avg > 0 else 0.0
            ax.axhline(theory_lwp, color='tab:orange', linestyle='-.', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWP={theory_lwp:.3e} (emp: {empirical_bulk_avg:.3e}, err: {err_pct_wp:+.1f}%)')
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add text box with summary
        textstr = f"Largest: Theory={theory_lwt:.3e}, Emp={empirical_largest:.3e}\nBulk Avg: Theory={theory_lwp:.3e}, Emp={empirical_bulk_avg:.3e}"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W0 covariance eigenvalue plot to {out_path}")
        print(f"  Empirical: Largest={empirical_largest:.3e}, Bulk_avg={empirical_bulk_avg:.3e}")
        if theory_lwt is not None:
            print(f"  Theory:    Largest={theory_lwt:.3e}, Bulk_avg={theory_lwp:.3e}")
    except Exception as e:
        print(f"  Warning: failed to plot W0 covariance eigenvalues for cfg={cfg}: {e}")


def plot_W0_cov_eigenvalues_per_ensemble(
    eigvals: np.ndarray,
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Bar plot of mean eigenvalues with std shading per-ensemble method.
    
    eigvals shape: (ens, d) - eigenvalues computed separately for each ensemble member
    Always displays:
    - Theoretical largest and bulk average
    - Empirical largest and bulk average with % error (ensemble-averaged)
    """
    try:
        eigvals_sorted = np.sort(eigvals, axis=1)[:, ::-1]  # Sort each ensemble member
        mean_spec = eigvals_sorted.mean(axis=0)  # Mean across ensemble members
        std_spec = eigvals_sorted.std(axis=0)   # Std across ensemble members
        idx = np.arange(mean_spec.shape[0])
        
        # Compute largest and bulk average empirically (from ensemble average)
        empirical_largest = mean_spec[0] if mean_spec.shape[0] > 0 else 0.0
        empirical_bulk_avg = mean_spec[1:].mean() if mean_spec.shape[0] > 1 else 0.0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(idx, mean_spec, color='tab:green', alpha=0.7, label='mean eigenvalue (per-ens avg)')
        ax.errorbar(idx, mean_spec, yerr=std_spec, fmt='none', ecolor='gray', elinewidth=1.0, capsize=2, label='±1 std')
        
        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / empirical_largest)) * 100.0 if empirical_largest > 0 else 0.0
            ax.axhline(theory_lwt, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWT={theory_lwt:.3e} (emp: {empirical_largest:.3e}, err: {err_pct_wt:+.1f}%)')
        
        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / empirical_bulk_avg)) * 100.0 if empirical_bulk_avg > 0 else 0.0
            ax.axhline(theory_lwp, color='tab:orange', linestyle='-.', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWP={theory_lwp:.3e} (emp: {empirical_bulk_avg:.3e}, err: {err_pct_wp:+.1f}%)')
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues per-ensemble (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add text box with summary
        textstr = f"Largest: Theory={theory_lwt:.3e}, Emp={empirical_largest:.3e}\nBulk Avg: Theory={theory_lwp:.3e}, Emp={empirical_bulk_avg:.3e}"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W0 covariance eigenvalue (per-ensemble) plot to {out_path}")
        print(f"  Empirical: Largest={empirical_largest:.3e}, Bulk_avg={empirical_bulk_avg:.3e}")
        if theory_lwt is not None:
            print(f"  Theory:    Largest={theory_lwt:.3e}, Bulk_avg={theory_lwp:.3e}")
    except Exception as e:
        print(f"  Warning: failed to plot W0 covariance eigenvalues (per-ensemble) for cfg={cfg}: {e}")


def plot_W0_cov_eigenvalues_both(
    eigvals_pooled: np.ndarray,
    eigvals_per_ens: np.ndarray,
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Plot pooled and per-ensemble eigenvalues in the same figure.

    Pooled: (d,) eigenvalues from pooled ens*N samples.
    Per-ensemble: (ens, d) eigenvalues per ensemble member (mean with std bars).
    """
    try:
        pooled_sorted = np.sort(eigvals_pooled)[::-1]
        per_ens_sorted = np.sort(eigvals_per_ens, axis=1)[:, ::-1]
        per_ens_mean = per_ens_sorted.mean(axis=0)
        per_ens_std = per_ens_sorted.std(axis=0)

        idx = np.arange(pooled_sorted.shape[0])
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(idx, pooled_sorted, color='tab:blue', linewidth=2.0, label='pooled (ens*N)')
        ax.errorbar(
            idx,
            per_ens_mean,
            yerr=per_ens_std,
            fmt='o',
            color='tab:green',
            ecolor='gray',
            elinewidth=1.0,
            capsize=2,
            markersize=4,
            label='per-ensemble mean ± std',
        )

        pooled_largest = pooled_sorted[0] if pooled_sorted.shape[0] > 0 else 0.0
        pooled_bulk = pooled_sorted[1:].mean() if pooled_sorted.shape[0] > 1 else 0.0

        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / pooled_largest)) * 100.0 if pooled_largest > 0 else 0.0
            ax.axhline(
                theory_lwt,
                color='tab:red',
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWT={theory_lwt:.3e} (pooled err: {err_pct_wt:+.1f}%)',
            )

        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / pooled_bulk)) * 100.0 if pooled_bulk > 0 else 0.0
            ax.axhline(
                theory_lwp,
                color='tab:orange',
                linestyle='-.',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWP={theory_lwp:.3e} (pooled err: {err_pct_wp:+.1f}%)',
            )

        ax.set_ylim(bottom=0)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues (pooled vs per-ensemble) (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)

        textstr = (
            f"Pooled: Largest={pooled_largest:.3e}, Bulk={pooled_bulk:.3e}\n"
            f"Per-ens mean: Largest={per_ens_mean[0]:.3e}, Bulk={per_ens_mean[1:].mean():.3e}"
        )
        ax.text(
            0.98,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined W0 covariance eigenvalue plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot combined W0 covariance eigenvalues for cfg={cfg}: {e}")


def plot_W0_cov_eigenvalues_both_with_stderr(
    mean_pooled: np.ndarray,
    stderr_pooled: np.ndarray,
    mean_per_ens: np.ndarray,
    stderr_per_ens: np.ndarray,
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Plot pooled vs per-ensemble means with stderr error bars (Langevin samples)."""
    try:
        pooled_sorted = np.sort(mean_pooled)[::-1]
        pooled_stderr = stderr_pooled[np.argsort(mean_pooled)[::-1]]
        per_ens_sorted = np.sort(mean_per_ens)[::-1]
        per_ens_stderr = stderr_per_ens[np.argsort(mean_per_ens)[::-1]]

        idx = np.arange(pooled_sorted.shape[0])
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.errorbar(
            idx,
            pooled_sorted,
            yerr=pooled_stderr,
            fmt='-o',
            color='tab:blue',
            ecolor='gray',
            elinewidth=1.0,
            capsize=2,
            markersize=3,
            label='pooled mean ± stderr',
        )
        ax.errorbar(
            idx,
            per_ens_sorted,
            yerr=per_ens_stderr,
            fmt='-o',
            color='tab:green',
            ecolor='gray',
            elinewidth=1.0,
            capsize=2,
            markersize=3,
            label='per-ens mean ± stderr',
        )

        pooled_largest = pooled_sorted[0] if pooled_sorted.shape[0] > 0 else 0.0
        pooled_bulk = pooled_sorted[1:].mean() if pooled_sorted.shape[0] > 1 else 0.0

        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / pooled_largest)) * 100.0 if pooled_largest > 0 else 0.0
            ax.axhline(
                theory_lwt,
                color='tab:red',
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWT={theory_lwt:.3e} (pooled err: {err_pct_wt:+.1f}%)',
            )

        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / pooled_bulk)) * 100.0 if pooled_bulk > 0 else 0.0
            ax.axhline(
                theory_lwp,
                color='tab:orange',
                linestyle='-.',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWP={theory_lwp:.3e} (pooled err: {err_pct_wp:+.1f}%)',
            )

        ax.set_ylim(bottom=0)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues (Langevin mean ± stderr) (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved Langevin stderr plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot Langevin stderr for cfg={cfg}: {e}")


def plot_W0_cov_eigenvalues_aggregate(
    model_spectra: List[np.ndarray],
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Aggregate scatter + mean/error plot across models (seeds).

    Each entry in model_spectra is a (d,) eigenvalue spectrum for a model.
    Error bars are std / sqrt(num_models).
    
    Always displays:
    - Theoretical largest and bulk average
    - Empirical largest and bulk average with % error
    """
    try:
        if not model_spectra:
            print(f"  Warning: no spectra to plot for cfg={cfg}")
            return

        spectra = np.stack(model_spectra, axis=0)  # (num_models, d)
        spectra_sorted = np.sort(spectra, axis=1)[:, ::-1]
        num_models = spectra_sorted.shape[0]

        lambda_w_star = spectra_sorted[:, 0]
        lambda_w_perp = spectra_sorted[:, 1:].mean(axis=1) if spectra_sorted.shape[1] > 1 else np.zeros(num_models)

        values = np.stack([lambda_w_star, lambda_w_perp], axis=1)  # (num_models, 2)
        mean_vals = values.mean(axis=0)
        std_vals = values.std(axis=0)
        err_vals = std_vals / np.sqrt(num_models)

        idx = np.arange(2)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(num_models):
            ax.scatter(idx, values[i], color='tab:blue', alpha=0.12, s=18)

        ax.errorbar(idx, mean_vals, yerr=err_vals, fmt='none', ecolor='gray',
                    elinewidth=1.5, capsize=3, label='mean ± std/sqrt(N)', linewidth=2)
        ax.scatter(idx, mean_vals, color='black', alpha=0.95, s=80, zorder=3, edgecolors='white', linewidth=1.5)
        
        # Add theory lines with error percentages
        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / mean_vals[0])) * 100.0 if mean_vals[0] > 0 else 0.0
            ax.axhline(theory_lwt, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWT (err: {err_pct_wt:+.1f}%)')
        
        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / mean_vals[1])) * 100.0 if mean_vals[1] > 0 else 0.0
            ax.axhline(theory_lwp, color='tab:orange', linestyle='-.', linewidth=2.5, alpha=0.8,
                       label=f'Theory lWP (err: {err_pct_wp:+.1f}%)')

        ax.set_ylim(bottom=0)
        ax.set_xticks(idx, [r'$\lambda^W_*$ (Largest)', r'$\langle \lambda^W_\perp \rangle$ (Bulk Avg)'], fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues (aggregate) (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=11, loc='best')

        # Add text box with comprehensive summary
        textstr = (f"Empirical: Largest={mean_vals[0]:.3e}±{err_vals[0]:.3e}, "
                   f"Bulk={mean_vals[1]:.3e}±{err_vals[1]:.3e}\n"
                   f"Theory:    Largest={theory_lwt:.3e}, Bulk={theory_lwp:.3e}")
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='gray'))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved aggregated W0 covariance eigenvalue plot to {out_path}")
        print(f"  Empirical: Largest={mean_vals[0]:.3e}±{err_vals[0]:.3e}, Bulk_avg={mean_vals[1]:.3e}±{err_vals[1]:.3e}")
        if theory_lwt is not None:
            print(f"  Theory:    Largest={theory_lwt:.3e}, Bulk_avg={theory_lwp:.3e}")
    except Exception as e:
        print(f"  Warning: failed to plot aggregated W0 covariance eigenvalues for cfg={cfg}: {e}")


def plot_W0_cov_eigenvalues_aggregate_both(
    pooled_spectra: List[np.ndarray],
    per_ens_spectra: List[np.ndarray],
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Aggregate plot comparing pooled vs per-ensemble spectra across seeds.

    pooled_spectra: list of (d,) arrays (pooled method) per model
    per_ens_spectra: list of (d,) arrays (per-ensemble mean) per model
    """
    try:
        if not pooled_spectra or not per_ens_spectra:
            print(f"  Warning: no spectra to plot for cfg={cfg}")
            return

        pooled_arr = np.array(pooled_spectra)
        per_ens_arr = np.array(per_ens_spectra)

        pooled_mean = pooled_arr.mean(axis=0)
        pooled_std = pooled_arr.std(axis=0)
        per_ens_mean = per_ens_arr.mean(axis=0)
        per_ens_std = per_ens_arr.std(axis=0)

        idx = np.arange(pooled_mean.shape[0])
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(idx, pooled_mean, color='tab:blue', linewidth=2.0, label='pooled mean')
        ax.fill_between(idx, pooled_mean - pooled_std, pooled_mean + pooled_std, color='tab:blue', alpha=0.2)

        ax.plot(idx, per_ens_mean, color='tab:green', linewidth=2.0, label='per-ens mean')
        ax.fill_between(idx, per_ens_mean - per_ens_std, per_ens_mean + per_ens_std, color='tab:green', alpha=0.2)

        pooled_largest = pooled_mean[0] if pooled_mean.shape[0] > 0 else 0.0
        pooled_bulk = pooled_mean[1:].mean() if pooled_mean.shape[0] > 1 else 0.0

        if theory_lwt is not None:
            err_pct_wt = (1.0 - (theory_lwt / pooled_largest)) * 100.0 if pooled_largest > 0 else 0.0
            ax.axhline(
                theory_lwt,
                color='tab:red',
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWT={theory_lwt:.3e} (pooled err: {err_pct_wt:+.1f}%)',
            )

        if theory_lwp is not None:
            err_pct_wp = (1.0 - (theory_lwp / pooled_bulk)) * 100.0 if pooled_bulk > 0 else 0.0
            ax.axhline(
                theory_lwp,
                color='tab:orange',
                linestyle='-.',
                linewidth=2.5,
                alpha=0.8,
                label=f'Theory lWP={theory_lwp:.3e} (pooled err: {err_pct_wp:+.1f}%)',
            )

        ax.set_ylim(bottom=0)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(
            f"W0 covariance eigenvalues (aggregate pooled vs per-ens) (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)

        textstr = (
            f"Pooled mean: Largest={pooled_mean[0]:.3e}, Bulk={pooled_mean[1:].mean():.3e}\n"
            f"Per-ens mean: Largest={per_ens_mean[0]:.3e}, Bulk={per_ens_mean[1:].mean():.3e}"
        )
        ax.text(
            0.98,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved aggregated combined W0 covariance eigenvalue plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot aggregated combined W0 covariance eigenvalues for cfg={cfg}: {e}")


def compute_theory(
    run_dir: Path,
    cfg: Dict[str, int],
    device: torch.device,
    kappa_eff_cache: Optional[Dict[Tuple[int, int, float], float]] = None,
) -> Dict[str, Optional[float]]:
    """Compute lWT/lWP theory using Experiment.eig_predictions.
    
    Loads kappa from config.json, computes kappa_eff using arcsin kernel,
    and uses kappa_eff for theory predictions.
    """
    try:
        # Load config.json to get actual kappa used in training
        config_path = run_dir / "config.json"
        kappa_bare = 1.0  # default
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json_lib.load(f)
            kappa_bare = float(config_data.get("kappa", 1.0))
        
        # Compute kappa_eff once per (d, P, kappa_bare)
        cache_key = (int(cfg["d"]), int(cfg["P"]), float(kappa_bare))
        if kappa_eff_cache is not None and cache_key in kappa_eff_cache:
            kappa_eff = kappa_eff_cache[cache_key]
        else:
            print(f"Computing kappa_eff for d={cfg['d']}, P={cfg['P']}, kappa_bare={kappa_bare}")
            kappa_eff = compute_kappa_eff(cfg["d"], cfg["P"], kappa_bare)
            print(f"  -> kappa_eff={kappa_eff:.6f}")
            if kappa_eff_cache is not None:
                kappa_eff_cache[cache_key] = kappa_eff
        
        exp = Experiment(
            file=str(run_dir),
            kappa=kappa_eff,
            N=int(cfg["N"]),
            d=int(cfg["d"]),
            chi=float(cfg["chi"]),
            P=int(cfg["P"]),
            ens=int(cfg.get("ens", 50)),
            device=device,
            eps=0.03,
        )

        preds = exp.eig_predictions()
        lwt = float(preds.lWT) if hasattr(preds, "lWT") and preds.lWT is not None else None
        lwp = float(preds.lWP) if hasattr(preds, "lWP") and preds.lWP is not None else None
        return {"lWT": lwt, "lWP": lwp, "kappa_eff": kappa_eff}
    except Exception as e:
        print(f"  Warning: failed to compute theory for d={cfg.get('d')}, P={cfg.get('P')}: {e}")
        import traceback
        traceback.print_exc()
        return {"lWT": None, "lWP": None}


def main():
    parser = argparse.ArgumentParser(description="Compute W0 covariance eigenvalues for runs.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    parser.add_argument("--langevin-avg", action="store_true",
                        help="Run Langevin dynamics for 50k epochs to average W0 before computing eigenvalues")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims
    use_langevin_avg = args.langevin_avg
    
    if use_langevin_avg:
        print("*** Langevin averaging enabled: W0 will be averaged via 50k epochs of Langevin dynamics ***\n")

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to do.")
        return

    aggregate: Dict[Tuple[int, int, int, int], Dict[str, object]] = {}
    kappa_eff_cache: Dict[Tuple[int, int, float], float] = {}

    for run_dir, cfg in runs:
        try:
            model = load_model(run_dir, cfg, device)
        except:
            print(f"Error loading model for {run_dir}, skipping. Exception:")
            import traceback
            traceback.print_exc()

        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue

        cfg_local = dict(cfg)
        cfg_local["ens"] = int(model.W0.shape[0])
        theory = compute_theory(run_dir, cfg_local, device, kappa_eff_cache=kappa_eff_cache)
        
        # Load kappa from config.json
        config_path = run_dir / "config.json"
        kappa_bare = 1.0  # default
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json_lib.load(f)
            kappa_bare = float(config_data.get("kappa", 1.0))

        result = compute_W0_cov_eigenvalues(model, cfg["d"], cfg["N"], 
                                            use_langevin_avg=use_langevin_avg, 
                                            chi=float(cfg["chi"]), kappa=kappa_bare,
                                            device=device)
        out_dir = run_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # result is always a dict with 'pooled', 'per_ens', and optionally 'langevin' keys
        eigvals_pooled = result['pooled']  # (d,)
        eigvals_per_ens = result['per_ens']  # (ens, d)
        
        # Save both methods
        suffix = "_langevin_avg" if use_langevin_avg else ""
        
        # Pooled method (d_sweep.py style)
        out_path_pooled = out_dir / f"W0_cov_eigenvalues{suffix}_pooled.npy"
        np.save(out_path_pooled, eigvals_pooled)
        print(
            f"Saved W0 covariance eigenvalues (pooled method) to {out_path_pooled} | shape {eigvals_pooled.shape}, "
            f"mean={eigvals_pooled.mean():.4e}, min={eigvals_pooled.min():.4e}, max={eigvals_pooled.max():.4e}"
        )
        
        # Per-ensemble method
        out_path_per_ens = out_dir / f"W0_cov_eigenvalues{suffix}_per_ens.npy"
        np.save(out_path_per_ens, eigvals_per_ens)
        print(
            f"Saved W0 covariance eigenvalues (per-ensemble method) to {out_path_per_ens} | shape {eigvals_per_ens.shape}, "
            f"mean={eigvals_per_ens.mean():.4e}, min={eigvals_per_ens.min():.4e}, max={eigvals_per_ens.max():.4e}"
        )
        
        # If Langevin averaging was done, save those too
        if use_langevin_avg and 'langevin' in result:
            langevin_result = result['langevin']
            for key, eigvals in langevin_result.items():
                out_path = out_dir / f"W0_cov_eigenvalues{suffix}_{key}.npy"
                np.save(out_path, eigvals)
                if isinstance(eigvals, np.ndarray):
                    print(f"  Saved {key} to {out_path} | mean={eigvals.mean():.4e}, min={eigvals.min():.4e}, max={eigvals.max():.4e}")
                else:
                    print(f"  Saved {key} to {out_path} | value={eigvals}")
            # Plot Langevin sample mean with stderr
            if (
                "W0_cov_avg_pooled" in langevin_result
                and "W0_cov_stderr_pooled" in langevin_result
                and "W0_cov_avg_per_ens" in langevin_result
                and "W0_cov_stderr_per_ens" in langevin_result
            ):
                plot_path_langevin = out_dir / f"W0_cov_eigenvalues{suffix}_langevin_stderr.png"
                plot_W0_cov_eigenvalues_both_with_stderr(
                    langevin_result["W0_cov_avg_pooled"],
                    langevin_result["W0_cov_stderr_pooled"],
                    langevin_result["W0_cov_avg_per_ens"],
                    langevin_result["W0_cov_stderr_per_ens"],
                    cfg,
                    plot_path_langevin,
                    theory_lwt=theory.get("lWT"),
                    theory_lwp=theory.get("lWP"),
                )
        
        # Plot both methods in one figure
        plot_path_both = out_dir / f"W0_cov_eigenvalues{suffix}_both.png"
        plot_W0_cov_eigenvalues_both(
            eigvals_pooled,
            eigvals_per_ens,
            cfg,
            plot_path_both,
            theory_lwt=theory.get("lWT"),
            theory_lwp=theory.get("lWP"),
        )
        
        # Aggregate across seeds: use pooled method for aggregation
        key = (cfg["d"], cfg["P"], cfg["N"], cfg["chi"])
        if key not in aggregate:
            aggregate[key] = {
                "cfg": dict(cfg),
                "spectra_pooled": [],
                "spectra_per_ens": [],
                "theory": theory,
            }
        aggregate[key]["spectra_pooled"].append(eigvals_pooled)  # eigvals_pooled is (d,)
        aggregate[key]["spectra_per_ens"].append(eigvals_per_ens)  # eigvals_per_ens is (ens, d)

    # Plot aggregated spectra across seeds
    for key, entry in aggregate.items():
        cfg_entry = entry["cfg"]
        out_dir = base_dir / "plots"
        suffix = "_langevin_avg" if use_langevin_avg else ""
        
        # Plot aggregated pooled vs per-ensemble in one figure
        per_ens_model_spectra = []
        for per_ens_spectrum in entry["spectra_per_ens"]:
            per_ens_sorted = np.sort(per_ens_spectrum, axis=1)[:, ::-1]
            per_ens_model_spectra.append(per_ens_sorted.mean(axis=0))
        out_path_both = out_dir / f"W0_cov_eigenvalues_aggregate_both_d{cfg_entry['d']}_P{cfg_entry['P']}_N{cfg_entry['N']}_chi{cfg_entry['chi']}{suffix}.png"
        plot_W0_cov_eigenvalues_aggregate_both(
            entry["spectra_pooled"],
            per_ens_model_spectra,
            cfg_entry,
            out_path_both,
            theory_lwt=entry["theory"].get("lWT") if entry.get("theory") else None,
            theory_lwp=entry["theory"].get("lWP") if entry.get("theory") else None,
        )


if __name__ == "__main__":
    main()
