#!/usr/bin/env python3
"""
Post-process a GPR convergence run: reload model/checkpoint, regenerate data
(using the same seed/procedure as training), compute eigenvalues and
model-vs-GP scatter with best-fit and y=x reference, and save a plot.

Usage:
    python analyze_gpr_results.py /path/to/run_dir

The run_dir must contain:
    - config.txt (written by train_gpr_convergence.py)
    - model.pt (trained weights)

Output:
    - analysis_plot.png saved into run_dir
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Library imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear
from GPKit import gpr_dot_product_explicit

torch.set_default_dtype(torch.float32)


def load_config(cfg_path: Path):
    cfg = {}
    with cfg_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            cfg[key.strip()] = val.strip()
    # Cast expected fields
    cfg_typed = {
        "P": int(cfg.get("P", 0)),
        "N": int(cfg.get("N", 0)),
        "d": int(cfg.get("d", 0)),
        "k": float(cfg.get("k", 0.0)),
        "chi": float(cfg.get("chi", 1.0)),
        "lr": float(cfg.get("lr", 0.0)),
        "epochs": int(cfg.get("epochs", 0)),
    }
    return cfg_typed


def analyze_one(run_dir: Path):
    cfg_path = run_dir / "config.txt"
    model_path = run_dir / "model.pt"

    if not cfg_path.exists() or not model_path.exists():
        print(f"Skipping {run_dir}: missing config.txt or model.pt")
        return None

    cfg = load_config(cfg_path)
    P, N, d, k, chi = cfg["P"], cfg["N"], cfg["d"], cfg["k"], cfg["chi"]

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Recreate data exactly as training did
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0]
    X_large = torch.randn(400, d, device=device)

    # Model
    model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=1,
                                      weight_initialization_variance=(1/d, 1/N, 1/N)).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Eigenvalues on larger eval set
    with torch.no_grad():
        h_eigs = model.H_eig(X_large, X_large).cpu().numpy()

    # GP prediction on train set (dot-product kernel)
    sigma_0_sq = 2 * k
    with torch.no_grad():
        gpr_pred = gpr_dot_product_explicit(X, Y, X, sigma_0_sq)
    model_out = model(X).detach().cpu().numpy().flatten()
    gpr_pred_np = gpr_pred.cpu().numpy().flatten()

    # Best-fit line
    slope, intercept = np.polyfit(gpr_pred_np, model_out, 1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Eigenvalues plot
    axes[0].plot(np.sort(h_eigs), marker='o', linestyle='-', linewidth=1.5)
    axes[0].set_title(f"Eigenvalues (P={P}, N={N}, d={d}, k={k:.3e}, chi={chi})")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("λ")
    axes[0].set_yscale("log")

    # Scatter with y=x and best-fit
    axes[1].scatter(gpr_pred_np, model_out, s=10, alpha=0.5, label="model vs GPR")
    # y=x
    mn = min(gpr_pred_np.min(), model_out.min())
    mx = max(gpr_pred_np.max(), model_out.max())
    axes[1].plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label="y = x")
    # best-fit
    x_line = np.linspace(mn, mx, 100)
    axes[1].plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5,
                 label=f"best fit (slope={slope:.3f})")
    axes[1].set_title(f"Model vs GP (P={P}, N={N}, d={d}, k={k:.3e}, chi={chi})")
    axes[1].set_xlabel("GP prediction")
    axes[1].set_ylabel("Model output")
    axes[1].legend()

    fig.tight_layout()
    out_path = run_dir / "analysis_plot.png"
    fig.savefig(str(out_path), dpi=200)
    print(f"Saved plot to {out_path}")
    return out_path


def main():
    # If a run_dir is provided, analyze only that. Otherwise, auto-discover runs.
    if len(sys.argv) == 2:
        run_dir = Path(sys.argv[1]).expanduser().resolve()
        analyze_one(run_dir)
    else:
        root = Path(__file__).parent.resolve()
        pattern = "P*_N*_d*_k*_chi*"
        run_dirs = sorted(root.glob(pattern))
        if not run_dirs:
            print(f"No run directories found matching {pattern} under {root}")
            sys.exit(0)
        for rd in run_dirs:
            analyze_one(rd)


if __name__ == "__main__":
    main()
