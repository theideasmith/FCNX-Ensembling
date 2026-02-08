#!/usr/bin/env python3
"""
Compute batched cubic (He3) projections of hidden-layer preactivations for saved FCN3 models.

Loads models from given run directories (containing model state and config),
streams random inputs X in batches, computes a1 = preactivation at second layer,
and projects onto Hermite cubic features in target and perpendicular directions.

Outputs JSON summary per run with mean/std/variance of projections and comparison plots.
"""

import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
import re
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from typing import Optional
import subprocess
import tempfile


def load_model_from_run(run_dir: Path, device: torch.device):
    """Recreate FCN3 model from run directory by loading config and weights."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    d = int(cfg.get("d"))
    N = int(cfg.get("N"))
    P = int(cfg.get("P"))
    ens = int(cfg.get("ens", 1))
    activation = cfg.get("activation", "erf")
    
    # Import FCN3Network
    from FCN3Network import FCN3NetworkActivationGeneric
    
    # Find model file
    state_paths = [
        run_dir / "model_final.pt",
        run_dir / "model.pt",
        run_dir / "checkpoint.pt",
    ]
    
    model_path = None
    for sp in state_paths:
        if sp.exists():
            model_path = sp
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No loadable model state found in {run_dir}")
    
    # Instantiate model
    model = FCN3NetworkActivationGeneric(
        d,
        N,
        N,
        P,
        ens=ens,
        activation=activation,
        device=device
    ).to(device)
    
    # Load weights
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    if len(state["W0"].shape) == 4:
        state["W0"] = state["W0"].squeeze(0)  # Remove num_seeds dim if present
        state["W1"] = state["W1"].squeeze(0)
        state["A"] = state["A"].squeeze(0)
    model.load_state_dict(state)
    
    # Use float64 for projection accuracy
    model = model.double()
    model.eval()
    return model


import re
import json
from pathlib import Path

def parse_run_params(run_dir: Path):
    """Parse d, P, N, chi, kappa, epsilon from directory name or config."""
    # Initialize with default epsilon
    d = P = N = chi = kappa = None
    epsilon = 0.03 
    
    # 1. Try loading from config.json
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            d = int(cfg.get("d")) if cfg.get("d") is not None else None
            P = int(cfg.get("P")) if cfg.get("P") is not None else None
            N = int(cfg.get("N")) if cfg.get("N") is not None else None
            chi = float(cfg.get("chi")) if cfg.get("chi") is not None else None
            kappa = float(cfg.get("kappa")) if cfg.get("kappa") is not None else None
            if cfg.get("eps") is not None:
                epsilon = float(cfg.get("eps"))
        except Exception:
            pass

    # 2. Regex fallback for: d100_P1200_N1000_chi10_kappa2.0
    # Matches integers for d, P, N and floats for chi, kappa
    pattern = (
        r"d(?P<d>\d+)_"
        r"P(?P<P>\d+)_"
        r"N(?P<N>\d+)_"
        r"chi(?P<chi>[-+]?\d*\.?\d+)_"
        r"kappa(?P<kappa>[-+]?\d*\.?\d+)"
        r"(?:_eps_(?P<eps>[-+]?\d*\.?\d+))?" # Still checks for eps just in case
    )

    m = re.search(pattern, run_dir.name)
    if m:
        if d is None: d = int(m.group("d"))
        if P is None: P = int(m.group("P"))
        if N is None: N = int(m.group("N"))
        if chi is None: chi = float(m.group("chi"))
        if kappa is None: kappa = float(m.group("kappa"))
        # Only override default eps if it exists in the filename
        if m.group("eps"): epsilon = float(m.group("eps"))

    # 3. Final safety for kappa
    if kappa is None and chi is not None:
        kappa = 1.0 / chi
    
    return d, P, N, chi, kappa, epsilon


def compute_theory_with_julia(d: int, n1: int, P: int, chi: float, kappa: float, epsilon: Optional[float] = None):
    """Invoke Julia FCN3 eigensolver and return parsed JSON results."""
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    if epsilon is None:
        epsilon = 0.0
    n2 = n1  # For FCN3, n2 = n1 (unless otherwise specified)
    b = 4.0 / (3.0 * np.pi)
    lr = 1e-6
    max_iter = 6_000_000
    anneal = True
    anneal_steps = 30000
    tol = 1e-12
    precision = 8
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "julia",
            str(julia_script),
            "--d", str(d),
            "--kappa", str(kappa),
            "--epsilon", str(epsilon),
            "--P", str(P),
            "--n1", str(n1),
            "--n2", str(n2),
            "--chi", str(chi),
            "--b", str(b),
            "--lr", str(lr),
            "--max-iter", str(max_iter),
            "--anneal-steps", str(anneal_steps),
            "--tol", str(tol),
            "--precision", str(precision),
            "--to", str(tmp_path),
            "--quiet"
        ]
        if not anneal:
            cmd.append("--no-anneal")
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, "r") as f:
            result = json.load(f)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def second_moment(projections):
    k = projections.flatten().shape[0]
    return torch.sum(projections**2) / k


def compute_h3_projections_streaming(
    model,
    d: int,
    P_total: int = 2_000_000,
    batch_size: int = 100_000,
    device: torch.device = torch.device("cpu"),
    perp_dim: int = 1,
):
    """
    Stream random inputs and compute cubic projections of hidden-layer preactivations.
    Uses model.h1_preactivation for FCN3.
    Returns a dict with mean/std/var for target and perpendicular projections.
    """
    assert perp_dim >= 0 and perp_dim < d, f"perp_dim {perp_dim} out of range for d={d}"

    dtype = torch.float32
    ens = model.ens
    n1 = model.n1
    model.to(dtype)
    
    # Accumulators for summed projections over P (He3 and He1)
    proj3_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj3_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)

    num_batches = P_total // batch_size
    remainder = P_total % batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches + (1 if remainder > 0 else 0)), desc="Batches"):
            bs = batch_size if i < num_batches else remainder
            if bs == 0:
                break
            X_batch = torch.randn(bs, d, dtype=dtype, device=device)

            # Cubic Hermite features (normalized to project; scaling by P later)
            x0 = X_batch[:, 0]
            x_perp = X_batch[:, perp_dim]
            phi3_target = (x0**3 - 3.0 * x0) 
            phi3_perp = (x_perp**3 - 3.0 * x_perp) 
            # Linear features (He1)
            phi1_target = x0
            phi1_perp = x_perp
            
            # Hidden preactivations a1: (bs, ens, n1) - FCN3 uses h1_preactivation
            a1 = model.h1_preactivation(X_batch)

            # Accumulate projections (sum over samples)
            proj3_target_sum += torch.einsum('pqn,p->qn', a1, phi3_target) / P_total
            proj3_perp_sum += torch.einsum('pqn,p->qn', a1, phi3_perp) / P_total
            proj1_target_sum += torch.einsum('pqn,p->qn', a1, phi1_target) / P_total
            proj1_perp_sum += torch.einsum('pqn,p->qn', a1, phi1_perp) / P_total
            torch.cuda.empty_cache()
            del X_batch, x0, x_perp, phi3_target, phi3_perp, phi1_target, phi1_perp, a1
    
    # Normalize by total P to get expectation
    proj3_target = proj3_target_sum
    proj3_perp = proj3_perp_sum
    proj1_target = proj1_target_sum
    proj1_perp = proj1_perp_sum

    # Save all h3 projections as eigenvalues: target and d-1 degenerate perp
    eig_target1 = proj1_target.var().cpu().numpy().flatten()
    eig_perp1 = proj1_perp.var().cpu().numpy().flatten()
    eig_target3 = proj3_target.var().cpu().numpy().flatten()
    eig_perp3 = proj3_perp.var().cpu().numpy().flatten()
    d_minus_1 = d - 1

    # Eigenvalues: [target, perp, perp, ..., perp] (d-1 times)
    eigenvalues = np.concatenate([
        eig_target1, 
        np.tile(eig_perp1, d_minus_1), 
        eig_target3, 
        np.tile(eig_perp3, d**3 - 1)
    ]).flatten()

    stats = {
        "ens": int(ens),
        "n1": int(n1),
        "d": int(d),
        "P_total": int(P_total),
        "batch_size": int(batch_size),
        "perp_dim": int(perp_dim),
        "h3": {
            "target": {
                "mean": float(torch.mean(proj3_target).item()),
                "std": float(torch.std(proj3_target).item()),
                "var": float(proj3_target.var().item()),
                "second_moment": float(second_moment(proj3_target).item()),
            },
            "perp": {
                "mean": float(torch.mean(proj3_perp).item()),
                "std": float(torch.std(proj3_perp).item()),
                "var": float(proj3_perp.var().item()),
                "second_moment": float(second_moment(proj3_perp).item()),
            },
        },
        "h1": {
            "target": {
                "mean": float(torch.mean(proj1_target).item()),
                "std": float(torch.std(proj1_target).item()),
                "var": float(proj1_target.var().item()),
                "second_moment": float(second_moment(proj1_target).item()),
            },
            "perp": {
                "mean": float(torch.mean(proj1_perp).item()),
                "std": float(torch.std(proj1_perp).item()),
                "var": float(proj1_perp.var().item()),
                "second_moment": float(second_moment(proj1_perp).item()),
            },
        },
        "h3_eigenvalues": eigenvalues.tolist(),
        "h3_target_eigenvalues": eig_target3.tolist(),
        "h3_perp_eigenvalues": eig_perp3.tolist(),
    }
    return stats


def save_stats(run_dir: Path, stats: dict):
    out_path = run_dir / "h3_projections_fcn3.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {out_path}")


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def main():
    parser = argparse.ArgumentParser(description="Compute He3 projections for FCN3 runs")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Run directories to process",
    )
    parser.add_argument("--P_total", type=int, default=2_000_000, help="Total samples (streamed)")
    parser.add_argument("--batch_size", type=int, default=100_000, help="Batch size for streaming")
    parser.add_argument("--perp_dim", type=int, default=1, help="Perpendicular dimension index")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--use-cache", action="store_true", help="Skip computation and plot from cached JSON files")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    if not args.runs:
        print("No runs specified. Use --runs to provide run directories.")
        return
    
    # Skip computation if using cache
    if not args.use_cache:
        for run in args.runs:
            run_dir = Path(run)
            if not run_dir.exists():
                print(f"Skipping missing run dir: {run_dir}")
                continue
            print(f"Processing: {run_dir}")
            try:
                model = load_model_from_run(run_dir, device)
                stats = compute_h3_projections_streaming(
                    model, 
                    model.d, 
                    args.P_total, 
                    args.batch_size, 
                    device, 
                    args.perp_dim
                )
                save_stats(run_dir, stats)
            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
                traceback.print_exc()
                continue

    # Process cached results and add theory
    for run in args.runs:
        run_dir = Path(run)
        rp = run_dir / "h3_projections_fcn3.json"
        if not rp.exists():
            print(f"No cached projections for {run_dir}")
            continue
        with open(rp, "r") as f:
            stats = json.load(f)
        
        # Parse run params
        d, P, N, chi, kappa, eps = parse_run_params(run_dir)
        if d is None or P is None or N is None or chi is None or kappa is None:
            print(f"Missing params for {run_dir}")
            continue
        
        # --- Compute 5K x 5K arcsin kernel eigenvalues for kappa bare ---
        np.random.seed(0)
        X = np.random.randn(5000, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / 5000.0
        stats["arcsin_kernel_eigenvalues"] = eigvals.tolist()
        
        # --- Bare theory eigenvalues ---
        try:
            theory = compute_theory_with_julia(d, N, P, chi, kappa, eps)
            stats["theory_bare"] = theory
        except Exception as e:
            print(f"Theory computation failed for {run_dir}: {e}")
            continue
        
        # --- Kappa correction: run self-consistent solver ---
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        # Use arcsin kernel eigenvalues for kappa bare
        eigenvalues = eigvals.tolist()
        with open(eig_json, "w") as f2:
            json.dump({"eigenvalues": eigenvalues, "kappa_bare": kappa}, f2)
        sc_cmd = [
            "julia", str(Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"),
            eig_json, str(P)
        ]
        try:
            sc_out = subprocess.check_output(sc_cmd, text=True)
            match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
            if match:
                kappa_eff = float(match.group(1))
                print(f"  Found kappa_eff: {kappa_eff:.6f} for {run_dir}")
            else:
                print(f"Warning: could not parse kappa_eff from self-consistent solver output for {run_dir}.")
                kappa_eff = kappa
        except Exception as e:
            print(f"Self-consistent solver failed for {run_dir}: {e}")
            kappa_eff = kappa
        
        # --- Rerun theory with kappa_eff ---
        try:
            theory_eff = compute_theory_with_julia(d, N, P, chi, kappa_eff, eps)
            stats["theory_kappa_eff"] = theory_eff
            stats["kappa_eff"] = kappa_eff
        except Exception as e:
            print(f"Theory (kappa_eff) computation failed for {run_dir}: {e}")
        
        # Save updated stats
        save_stats(run_dir, stats)

    # Aggregate and plot across runs
    labels = []
    h3_target_vars = []
    h3_perp_vars = []
    h1_target_vars = []
    h1_perp_vars = []
    theory_eff_h3_target = None
    theory_eff_h3_perp = None
    theory_eff_h1_target = None
    theory_eff_h1_perp = None
    theory_bare_h3_target = None
    theory_bare_h3_perp = None
    theory_bare_h1_target = None
    theory_bare_h1_perp = None
    
    for run in args.runs:
        rp = Path(run) / "h3_projections_fcn3.json"
        if not rp.exists():
            continue
        with open(rp, "r") as f:
            s = json.load(f)
        lbl = re.search(r"seed_(\d+)", Path(run).name)
        labels.append(lbl.group(1) if lbl else Path(run).name)
        h3_target_vars.append(s["h3"]["target"]["var"])
        h3_perp_vars.append(s["h3"]["perp"]["var"])
        h1_target_vars.append(s["h1"]["target"]["var"])
        h1_perp_vars.append(s["h1"]["perp"]["var"])
        
        # Get theory_kappa_eff predictions
        theory_eff_block = s.get("theory_kappa_eff", {})
        target_eff_block = theory_eff_block.get("target", {}) if isinstance(theory_eff_block, dict) else {}
        perp_eff_block = theory_eff_block.get("perpendicular", {}) if isinstance(theory_eff_block, dict) else {}
        theory_eff_h3_target = target_eff_block.get("lH3T", theory_eff_h3_target)
        theory_eff_h1_target = target_eff_block.get("lH1T", theory_eff_h1_target)
        theory_eff_h3_perp = perp_eff_block.get("lH3P", theory_eff_h3_perp)
        theory_eff_h1_perp = perp_eff_block.get("lH1P", theory_eff_h1_perp)
        
        # Get theory_bare predictions
        theory_bare_block = s.get("theory_bare", {})
        target_bare_block = theory_bare_block.get("target", {}) if isinstance(theory_bare_block, dict) else {}
        perp_bare_block = theory_bare_block.get("perpendicular", {}) if isinstance(theory_bare_block, dict) else {}
        theory_bare_h3_target = target_bare_block.get("lH3T", theory_bare_h3_target)
        theory_bare_h1_target = target_bare_block.get("lH1T", theory_bare_h1_target)
        theory_bare_h3_perp = perp_bare_block.get("lH3P", theory_bare_h3_perp)
        theory_bare_h1_perp = perp_bare_block.get("lH1P", theory_bare_h1_perp)

    if labels:
        try:
            x = np.arange(len(labels))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            
            # He3 Target subplot
            axes[0, 0].bar(x, h3_target_vars, color="#4C78A8")
            if theory_eff_h3_target is not None:
                axes[0, 0].axhline(theory_eff_h3_target, color="#E45756", linestyle="--", label="Theory (κ_eff)")
            if theory_bare_h3_target is not None:
                axes[0, 0].axhline(theory_bare_h3_target, color="#F58518", linestyle=":", label="Theory (κ_bare)")
            axes[0, 0].set_title("He3 Projections (target)")
            axes[0, 0].set_ylabel("Variance")
            if theory_eff_h3_target is not None or theory_bare_h3_target is not None:
                axes[0, 0].legend()

            # He3 Perp subplot
            axes[0, 1].bar(x, h3_perp_vars, color="#4C78A8")
            if theory_eff_h3_perp is not None:
                axes[0, 1].axhline(theory_eff_h3_perp, color="#E45756", linestyle="--", label="Theory (κ_eff)")
            if theory_bare_h3_perp is not None:
                axes[0, 1].axhline(theory_bare_h3_perp, color="#F58518", linestyle=":", label="Theory (κ_bare)")
            axes[0, 1].set_title("He3 Projections (perp)")
            axes[0, 1].set_ylabel("Variance")
            if theory_eff_h3_perp is not None or theory_bare_h3_perp is not None:
                axes[0, 1].legend()

            # He1 Target subplot
            axes[1, 0].bar(x, h1_target_vars, color="#72B7B2")
            if theory_eff_h1_target is not None:
                axes[1, 0].axhline(theory_eff_h1_target, color="#E45756", linestyle="--", label="Theory (κ_eff)")
            if theory_bare_h1_target is not None:
                axes[1, 0].axhline(theory_bare_h1_target, color="#F58518", linestyle=":", label="Theory (κ_bare)")
            axes[1, 0].set_title("He1 Projections (target)")
            axes[1, 0].set_ylabel("Variance")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(labels)
            if theory_eff_h1_target is not None or theory_bare_h1_target is not None:
                axes[1, 0].legend()

            # He1 Perp subplot
            axes[1, 1].bar(x, h1_perp_vars, color="#72B7B2")
            if theory_eff_h1_perp is not None:
                axes[1, 1].axhline(theory_eff_h1_perp, color="#E45756", linestyle="--", label="Theory (κ_eff)")
            if theory_bare_h1_perp is not None:
                axes[1, 1].axhline(theory_bare_h1_perp, color="#F58518", linestyle=":", label="Theory (κ_bare)")
            axes[1, 1].set_title("He1 Projections (perp)")
            axes[1, 1].set_ylabel("Variance")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(labels)
            if theory_eff_h1_perp is not None or theory_bare_h1_perp is not None:
                axes[1, 1].legend()

            fig.tight_layout()
            # Save plot to parent directory of first run
            if args.runs:
                parent_dir = Path(args.runs[0]).parent
            else:
                parent_dir = Path(__file__).parent
            out_plot = parent_dir / "h_projections_summary_fcn3.png"
            fig.savefig(out_plot, dpi=150)
            plt.close(fig)
            print(f"Saved summary plot: {out_plot}")

            # Save a JSON config with the image path and models used
            summary_json = parent_dir / "h_projections_summary_fcn3.json"
            summary = {
                "image": str(out_plot.name),
                "models": args.runs
            }
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary config: {summary_json}")
        except Exception as e:
            print(f"Plotting failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()