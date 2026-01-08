#!/usr/bin/env python3
"""Generate bar plots of the full empirical spectrum (sorted) with theory overlays.

- Scans for model.pt files under a base directory with the pattern d<d>_P<P>_N<N>_chi<chi>[/...]/seed<seed>/model.pt
- Computes empirical eigenvalues via H_eig_random_svd on a P x d random dataset
- Sorts the spectrum in descending order and plots it on a log y-scale
- Adds horizontal lines for available theory predictions (He1/He1_perp/He3/He3_perp)
- Saves one plot per run under <run_dir>/plots/full_spectrum_sorted.png

Usage:
    python plot_full_spectrum.py --base-dir . --dims 150
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric
from Experiment import Experiment

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gaussian_action(x: np.ndarray, var: Optional[float]) -> np.ndarray:
    """Compute -log p(x) for N(0, var). Returns nan where var is invalid."""
    if var is None or not np.isfinite(var) or var <= 0:
        return np.full_like(x, np.nan, dtype=float)
    return x**2 / (2.0 * var) + 0.5 * np.log(2.0 * np.pi * var)


def h_random_QB_activation_generic(model, X, k=2000, p=10):
    """Low-rank QB approximation for ActivationGeneric models (no built-in H_random_QB).

    Mirrors the logic used in the linear ensemble variant: builds random projections
    against h1 preactivations, computes QR, and returns Q, Z.
    """
    with torch.no_grad():
        l = k + p
        h1 = model.h1_preactivation(X)  # (N, ens, n1)
        Omega = torch.randn((X.shape[0], l), device=model.device, dtype=h1.dtype)

        res = torch.zeros((X.shape[0], l), device=model.device, dtype=h1.dtype)
        chunk_size = 4096
        N = X.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h1, h1, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)

        Z = torch.zeros((X.shape[0], l), device=model.device, dtype=h1.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h1, h1) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        return Q, Z


def j_random_QB_activation_generic(model, X, k=2000, p=10):
    """Low-rank QB approximation for J kernel using h0 activations.

    Similar to H kernel but uses h0_activation (first layer activations) instead of
    h1_preactivation (second layer preactivations).
    """
    with torch.no_grad():
        l = k + p
        h0 = model.h0_activation(X)  # (N, ens, n1)
        Omega = torch.randn((X.shape[0], l), device=model.device, dtype=h0.dtype)

        res = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        chunk_size = 4096
        N = X.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)

        Z = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        return Q, Z


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories that contain model.pt and match the naming convention."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)"

    model_files = list(base.glob(f"**/*{suffix}*/model.pt")) if suffix else list(base.glob("**/model.pt"))
    for model_file in model_files:
        seed_dir = model_file.parent
        seed_name = seed_dir.name
        m_seed = re.match(r"seed(\d+)", seed_name)
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
    ens = int(state_dict['W0'].shape[0])

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


def compute_empirical_spectrum(
    model: FCN3NetworkActivationGeneric, d: int, P: int, device: torch.device, k: Optional[int] = None
) -> Optional[np.ndarray]:
    """Compute eigenvalues using H_eig_random_svd on a large random dataset."""
    try:
        with torch.no_grad():
            model.to(device)
            model.device = device
            X = torch.randn(10_000, d, device=device)
            if k is None:
                k = 9000
            eigs = model.H_eig_random_svd(X, k)
            return eigs.detach().cpu().numpy()
    except Exception as e:
        print(f"  Warning: failed to compute spectrum via projection method for {P=}, {d=}: {e}")
        return None


def compute_empirical_spectrum_activation(
    model: FCN3NetworkActivationGeneric, d: int, P: int, device: torch.device, k: Optional[int] = None
) -> Optional[np.ndarray]:
    """Compute activation layer eigenvalues via projection method using J_random_QB and J_eig.

    Recipe:
    1) Generate large dataset X (p_large x d) and cubic features Y1=X, Y3=(X^3-3X)/6.
    2) Compute low-rank QB approximation of J with j_random_QB_activation_generic.
    3) SVD of B (=Z^T), reconstruct U, Sigma.
    4) Left eigenvalues from J acting on normalized Y1 and Y3 projections.
    5) Return concatenated eigenvalues (Y1 then Y3) as NumPy.
    """
    try:
        with torch.no_grad():
            model.to(device)
            model.device = device

            p_large = 10_000
            X = torch.randn(p_large, d, device=device)
            Y1 = X
            Y3 = (X ** 3 - 3.0 * X) / 6.0**0.5

            # Low-rank approximation QB for J kernel
            Q, Z = j_random_QB_activation_generic(model, X, k=9000, p=10)
            Ut, _S, V = torch.linalg.svd(Z.T)
            m, n = Z.shape[1], Z.shape[0]
            k_eff = min(m, n)
            Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
            Sigma[:k_eff, :k_eff] = torch.diag(_S[:k_eff])
            U = torch.matmul(Q, Ut)

            # Left eigenvalues for Y1 via J_eig
            Y1_norm = Y1 / torch.norm(Y1, dim=0)
            left_eigenvaluesY1 = model.J_eig(X, Y1_norm)

            # Left eigenvalues for Y3 via projection through U, Sigma
            Y3_norm = Y3 / torch.norm(Y3, dim=0)
            proj = (Y3_norm.t() @ U)
            left_Y3_mat = proj @ torch.diag(_S[:k_eff]) @ (U.t() @ Y3_norm)
            left_eigenvaluesY3 = left_Y3_mat.diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]

            left_eigenvalues = torch.cat([left_eigenvaluesY1, left_eigenvaluesY3], dim=0)
            return left_eigenvalues.detach().cpu().numpy()
    except Exception as e:
        print(f"  Warning: failed to compute activation spectrum via projection method for {P=}, {d=}: {e}")
        return None


def compute_theory(cfg: Dict[str, int], device: torch.device) -> Dict[str, Dict[str, Optional[float]]]:
    """Get theoretical predictions by calling Julia eos_fcn3erf.jl and reading JSON output."""
    d = int(cfg.get("d"))
    P = int(cfg.get("P"))
    N = int(cfg.get("N"))
    chi = float(cfg.get("chi"))
    kappa = float(cfg.get("kappa", 2.0))
    eps = float(cfg.get("epsilon", 0.03))

    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--n2={N}",
        f"--chi={chi}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={to_path}",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(to_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Julia theory solver failed: {e}")
        data = {}
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass

    tgt = data.get("target", {}) if isinstance(data, dict) else {}
    perp = data.get("perpendicular", {}) if isinstance(data, dict) else {}

    return {
        "H": {
            "lH1T": tgt.get("lH1T"),
            "lH1P": perp.get("lH1P"),
            "lH3T": tgt.get("lH3T"),
            "lH3P": perp.get("lH3P"),
        },
        "J": {
            "lJ1T": tgt.get("lJ1T"),
            "lJ1P": perp.get("lJ1P"),
            "lJ3T": tgt.get("lJ3T"),
            "lJ3P": perp.get("lJ3P"),
        }
    }


def plot_full_spectrum(
    run_dir: Path,
    cfg: Dict[str, int],
    spectrum_np: np.ndarray,
    theory: Dict[str, Optional[float]],
    out_path: Path,
    title_prefix: str = "Preactivation (H)",
    theory_label_prefix: str = "H",
) -> None:
    """Plot sorted spectrum with theory reference lines on log y-axis."""
    try:
        spectrum_sorted = np.sort(spectrum_np)[::-1]
        fig, ax = plt.subplots(figsize=(14, 6))
        pos = np.arange(len(spectrum_sorted))
        
        d = cfg['d']
        
        # Color-code eigenvalues by their theoretical correspondence:
        # 0: target He1 (red), 1:d-1: perp He1 (orange), d: target He3 (green), d+1+: perp He3 (brown)
        colors = np.full(len(spectrum_sorted), 'tab:brown', dtype=object)  # default: perp He3
        if len(colors) > 0:
            colors[0] = 'tab:red'  # Target He1
        if len(colors) > 1 and d > 1:
            colors[1:min(d, len(colors))] = 'tab:orange'  # Perp He1
        if len(colors) > d:
            colors[d] = 'tab:green'  # Target He3
        # Rest remain brown (perp He3)
        
        # Plot bars with appropriate colors
        for i in range(len(spectrum_sorted)):
            label = None
            if i == 0:
                label = 'Target He1'
            elif i == 1 and d > 1:
                label = 'Perp He1'
            elif i == d:
                label = 'Target He3'
            elif i == d + 1:
                label = 'Perp He3'
            ax.bar(pos[i], spectrum_sorted[i], color=colors[i], alpha=0.75, label=label)

        # Theory reference lines
        l1T_key = f"l{theory_label_prefix}1T"
        l1P_key = f"l{theory_label_prefix}1P"
        l3T_key = f"l{theory_label_prefix}3T"
        l3P_key = f"l{theory_label_prefix}3P"
        
        if theory.get(l1T_key) is not None and np.isfinite(theory[l1T_key]):
            ax.axhline(theory[l1T_key], color='tab:red', linestyle='--', linewidth=2.0, label=f"Theory {theory_label_prefix}e1: {theory[l1T_key]:.4g}")
        if theory.get(l1P_key) is not None and np.isfinite(theory[l1P_key]):
            ax.axhline(theory[l1P_key], color='tab:orange', linestyle='--', linewidth=2.0, alpha=0.7, label=f"Theory {theory_label_prefix}e1_perp: {theory[l1P_key]:.4g}")
        if theory.get(l3T_key) is not None and np.isfinite(theory[l3T_key]):
            ax.axhline(theory[l3T_key], color='tab:green', linestyle='-.', linewidth=2.0, label=f"Theory {theory_label_prefix}e3: {theory[l3T_key]:.4g}")
        if theory.get(l3P_key) is not None and np.isfinite(theory[l3P_key]):
            ax.axhline(theory[l3P_key], color='tab:brown', linestyle='-.', linewidth=2.0, alpha=0.7, label=f"Theory {theory_label_prefix}e3_perp: {theory[l3P_key]:.4g}")

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.set_title(f"{title_prefix} Spectrum (sorted) for d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved full spectrum plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot spectrum for {run_dir}: {e}")


def plot_weight_distribution(
    run_dir: Path,
    cfg: Dict[str, int],
    model: FCN3NetworkActivationGeneric,
    out_path: Path,
) -> None:
    """Plot distribution of W0 weights: 0th feature column vs. mean across all features."""
    try:
        # W0 shape: (ens, N, d)
        W0 = model.W0.detach().cpu().numpy()

        # Distribution of 0th feature: W0[:, :, 0] flattened
        w0_feature0 = W0[:, :, 0].flatten()
        
        # Mean across d dimension, then flatten: W0.mean(axis=2).flatten()
        w0_mean_across_d = W0[:,:,1:].mean(axis=2).flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: histogram of 0th feature weights
        axes[0].hist(w0_feature0, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', density=True)
        axes[0].set_xlabel('Weight value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'W0[:, :, 0] distribution (0th feature)\nVar={w0_feature0.var():.4e}, Mean={w0_feature0.mean():.4e}')
        axes[0].grid(True, alpha=0.3)
        
        # Right: histogram of mean across d
        axes[1].hist(w0_mean_across_d, bins=50, color='tab:orange', alpha=0.7, edgecolor='black', density=True)
        axes[1].set_xlabel('Weight value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'W0.mean(axis=2) distribution (avg over d)\nVar={w0_mean_across_d.var():.4e}, Mean={w0_mean_across_d.mean():.4e}')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f"Read-in Layer W0 Weight Distributions (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})", fontsize=12)
        fig.tight_layout()
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved weight distribution plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot weight distribution for {run_dir}: {e}")


def plot_output_projection_histograms(
    run_dir: Path,
    cfg: Dict[str, int],
    model: FCN3NetworkActivationGeneric,
    device: torch.device,
    out_path: Path,
    theory_H: Dict[str, Optional[float]],
) -> Optional[Dict[str, np.ndarray]]:
    """Plot action histograms (-log P) of output projections onto target x[:,0] and perp x[:,3].

    - Generates x with seed from cfg for reproducibility.
    - Computes model outputs averaged across ensembles.
    - Builds histograms of per-sample projections output*x0 and output*x3.
    - Plots action curves (-log probability) for both directions.
    """
    try:
        d = int(cfg.get("d"))
        P = int(cfg.get("P"))
        seed = int(cfg.get("seed")) if "seed" in cfg and cfg["seed"] is not None else None

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x = torch.randn(10000, d, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model.h1_preactivation(x)  # (P, ens) or (P,)
            # Ensure shape (P, ens) to get per-ensemble projection distribution
            if output.ndim == 1:
                output = output.unsqueeze(1)
        P = output.shape[0]
        x0_target = x[:, 0]
        x0_target_normed = x0_target / x0_target.norm() * np.sqrt(P)
        x3_perp = x[:, 3] if d > 3 else torch.randn_like(x[:, 0])
        x3_perp_normed = x3_perp / x3_perp.norm() * np.sqrt(P)
        # Hermite cubic polynomials for target and perp: (x^3 - 3x)/sqrt(6)
        h3_target = (x0_target**3 - 3.0 * x0_target) / (6.0**0.5)
        h3_target_normed = h3_target / h3_target.norm() * np.sqrt(P)
        h3_perp = (x3_perp**3 - 3.0 * x3_perp) / (6.0**0.5)
        h3_perp_normed = h3_perp / h3_perp.norm() * np.sqrt(P)

        # Project outputs onto target/perp directions per ensemble
        # normalize by norm of vectors
        proj_lin_target = torch.einsum('pqn,p->qn', output, x0_target_normed) / P 
        proj_lin_perp = torch.einsum('pqn,p->qn', output, x3_perp_normed) / P 
        proj_cubic_target = torch.einsum('pqn,p->qn', output, h3_target_normed) / P 
        proj_cubic_perp = torch.einsum('pqn,p->qn', output, h3_perp_normed) / P

        proj_lin_target_samples = proj_lin_target.detach().cpu().numpy()
        proj_lin_perp_samples = proj_lin_perp.detach().cpu().numpy()
        proj_cubic_target_samples = proj_cubic_target.detach().cpu().numpy()
        proj_cubic_perp_samples = proj_cubic_perp.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        collections = {
            "lin_target": proj_lin_target_samples,
            "lin_perp": proj_lin_perp_samples,
            "cubic_target": proj_cubic_target_samples,
            "cubic_perp": proj_cubic_perp_samples,
        }

        lin_pairs = [
            ("lin_target", "Target linear x[:,0]", "royalblue", theory_H.get("lH1T")),
            ("lin_perp", "Perp linear x[:,3]", "orange", theory_H.get("lH1P")),
        ]
        cubic_pairs = [
            ("cubic_target", "Target cubic He3(x[:,0])", "tab:green", theory_H.get("lH3T")),
            ("cubic_perp", "Perp cubic He3(x[:,3])", "tab:brown", theory_H.get("lH3P")),
        ]

        for ax, pairs, title in zip(
            axes,
            [lin_pairs, cubic_pairs],
            ["Linear projections", "Cubic projections"],
        ):
            for key, label, color, var_th in pairs:
                samples = collections[key]
                hist, bin_edges = np.histogram(samples, bins=200, density=True)
                bin_widths = np.diff(bin_edges)
                dens = hist * bin_widths
                # dens = np.clip(dens, 1e-12, None)
                action = -np.log(dens)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mask = np.isfinite(action) & (action > 0)
                sample_var = float(np.var(samples)) if samples.size > 0 else float("nan")
                ax.plot(bin_centers[mask], action[mask], label=f"{label} (var={sample_var:.3g})", color=color)

                # Gaussian theory overlay
                gauss_act = gaussian_action(bin_centers, var_th)
                if np.any(np.isfinite(gauss_act)):
                    ax.plot(
                        bin_centers,
                        gauss_act,
                        linestyle="--",
                        color=color,
                        alpha=0.6,
                        label=f"Gaussian var={var_th:.3g}" if var_th is not None else "Gaussian (no var)",
                    )

            ax.set_title(title)
            ax.set_xlabel("Output projection value")
            ax.set_ylabel("Action: -log P")
            ax.grid(True, alpha=0.3)
            ax.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved output projection histograms to {out_path}")
        return collections
    except Exception as e:
        print(f"  Warning: failed to plot output projections for {run_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot full empirical spectra with theory reference lines.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to plot.")
        return

    aggregated_samples = {
        "lin_target": [],
        "lin_perp": [],
        "cubic_target": [],
        "cubic_perp": [],
    }
    theory_first = None

    for run_dir, cfg in runs:
        model = load_model(run_dir, cfg, device)
        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue

        theory = compute_theory(cfg, device)
        if theory_first is None:
            theory_first = theory

        out_proj_path = run_dir / "plots" / "output_projection_histograms.png"
        collected = plot_output_projection_histograms(run_dir, cfg, model, device, out_proj_path, theory["H"])
        if collected:
            for k in aggregated_samples:
                if k in collected:
                    aggregated_samples[k].append(collected[k])

    # Aggregate across seeds and plot dataset-averaged action
    combined = {k: np.concatenate(v) if v else np.array([]) for k, v in aggregated_samples.items()}
    if any(arr.size > 0 for arr in combined.values()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        lin_pairs = [
            ("lin_target", "Target linear", "royalblue", theory_first["H"].get("lH1T") if theory_first else None),
            ("lin_perp", "Perp linear", "orange", theory_first["H"].get("lH1P") if theory_first else None),
        ]
        cubic_pairs = [
            ("cubic_target", "Target cubic", "tab:green", theory_first["H"].get("lH3T") if theory_first else None),
            ("cubic_perp", "Perp cubic", "tab:brown", theory_first["H"].get("lH3P") if theory_first else None),
        ]
        for ax, pairs, title in zip(axes, [lin_pairs, cubic_pairs], ["Linear projections (aggregated)", "Cubic projections (aggregated)"]):
            for key, label, color, var_th in pairs:
                arr = combined[key]
                if arr.size == 0:
                    continue
                hist, bin_edges = np.histogram(arr, bins=200, density=True)
                bin_widths = np.diff(bin_edges)
                dens = hist * bin_widths
                action = -np.log(dens)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mask = np.isfinite(action) & (action > 0)
                sample_var = float(np.var(arr))
                ax.plot(bin_centers[mask], action[mask], label=f"{label} (var={sample_var:.3g})", color=color)
                gauss_act = gaussian_action(bin_centers, var_th)
                if np.any(np.isfinite(gauss_act)):
                    ax.plot(
                        bin_centers,
                        gauss_act,
                        linestyle="--",
                        color=color,
                        alpha=0.6,
                        label=f"Gaussian var={var_th:.3g}" if var_th is not None else "Gaussian (no var)",
                    )
            ax.set_title(title)
            ax.set_xlabel("Output projection value")
            ax.set_ylabel("Action: -log P")
            ax.grid(True, alpha=0.3)
            ax.legend()

        agg_out = base_dir / "plots" / "output_projection_histograms_aggregated.png"
        agg_out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(agg_out, dpi=200)
        plt.close(fig)
        print(f"Saved aggregated output projection histograms to {agg_out}")


if __name__ == "__main__":
    main()
