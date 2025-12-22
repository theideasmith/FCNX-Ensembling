#!/usr/bin/env python3
"""
Compare linear FCN2 network outputs with GPR predictions using linear kernel K(x, x') = (x·x')/d.

Usage:
    python compare_linear_gpr_dot_kernel.py --run-dir <path> [--seed 42] [--num-samples 500]
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def linear_dot_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute normalized dot product kernel matrix: K(x, x') = (x·x') / d."""
    d = X.shape[1]
    K = torch.einsum('ui,vi->uv', X, X) / d
    return K


def linear_gpr_predict(X: torch.Tensor, Y: torch.Tensor, sigma0_sq: float) -> torch.Tensor:
    """GPR prediction with normalized linear kernel on training data."""
    K = linear_dot_kernel(X)
    n = K.shape[0]
    K_reg = K + sigma0_sq * torch.eye(n, device=X.device)
    alpha = torch.linalg.solve(K_reg, Y)
    return K @ alpha


def load_model(run_dir: Path, device: torch.device):
    """Load linear model and config from run directory."""
    checkpoint_path = run_dir / "checkpoint.pt"
    model_path = run_dir / "model.pt"
    config_path = run_dir / "config.json"

    config = None
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    state_dict = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if config is None:
            config = checkpoint.get("config", None)
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location=device)

    if config is None:
        raise FileNotFoundError("config.json or checkpoint config not found")

    d = int(config["d"])
    P = int(config["P"])
    N = int(config["N"])
    ens = int(config.get("ens", 5))

    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="linear",
        weight_initialization_variance=(1/d, 1/N),
        device=device,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Compare linear FCN2 with GPR linear kernel")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:X)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input generation")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--sigma0", type=float, default=0.01, help="Observation noise std for GPR")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, config = load_model(run_dir, torch.device("cpu"))
    d = int(config["d"])
    P_cfg = int(config["P"])

    print(f"\nComparing Linear FCN2 with GPR (linear kernel K(x,x')=(x·x')/d)")
    print(f"  Model: {run_dir.name}")
    print(f"  d={d}, P_cfg={P_cfg}, N={config['N']}")
    print()

    # Generate test data
    torch.manual_seed(args.seed)
    X = torch.randn(args.num_samples, d, device=torch.device("cpu"))
    Y = X[:, 0]  # Target is first dimension

    # Network predictions
    with torch.no_grad():
        network_out = model(X)  # (num_samples, ens)
        network_mean = network_out.mean(dim=1)
        network_std = network_out.std(dim=1)

    # GPR predictions
    with torch.no_grad():
        gpr_pred = linear_gpr_predict(X, Y, sigma0_sq=args.sigma0**2)

    # Move to numpy
    Y_np = Y.numpy()
    net_mean_np = network_mean.numpy()
    net_std_np = network_std.numpy()
    gpr_np = gpr_pred.numpy()

    # Compute residuals and statistics
    residuals = net_mean_np - gpr_np
    correlation = np.corrcoef(net_mean_np, gpr_np)[0, 1]
    mse = np.mean((net_mean_np - gpr_np) ** 2)
    mae = np.mean(np.abs(net_mean_np - gpr_np))

    print(f"Results:")
    print(f"  Network output:  mean={net_mean_np.mean():.6f}, std={net_mean_np.std():.6f}")
    print(f"  GPR output:      mean={gpr_np.mean():.6f}, std={gpr_np.std():.6f}")
    print(f"  Target Y:        mean={Y_np.mean():.6f}, std={Y_np.std():.6f}")
    print()
    print(f"  Residuals (Net - GPR):")
    print(f"    Mean: {residuals.mean():.6e}")
    print(f"    Std:  {residuals.std():.6e}")
    print(f"    Max:  {np.abs(residuals).max():.6e}")
    print(f"  Correlation (Net vs GPR): {correlation:.6f}")
    print(f"  MSE (Net vs GPR): {mse:.6e}")
    print(f"  MAE (Net vs GPR): {mae:.6e}")
    print()

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Network vs Target
    axes[0, 0].scatter(Y_np, net_mean_np, alpha=0.5, s=20, label="Network")
    axes[0, 0].errorbar(Y_np[::10], net_mean_np[::10], yerr=net_std_np[::10], 
                       fmt='none', elinewidth=0.5, capsize=1, alpha=0.3)
    mn, mx = min(Y_np.min(), net_mean_np.min()), max(Y_np.max(), net_mean_np.max())
    axes[0, 0].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
    axes[0, 0].set_xlabel("Target Y")
    axes[0, 0].set_ylabel("Network Output")
    axes[0, 0].set_title("Linear FCN2 vs Target")
    axes[0, 0].grid(alpha=0.3)

    # GPR vs Target
    axes[0, 1].scatter(Y_np, gpr_np, alpha=0.5, s=20, color="orange", label="GPR")
    mn, mx = min(Y_np.min(), gpr_np.min()), max(Y_np.max(), gpr_np.max())
    axes[0, 1].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
    axes[0, 1].set_xlabel("Target Y")
    axes[0, 1].set_ylabel("GPR Output (linear kernel)")
    axes[0, 1].set_title("GPR (linear K) vs Target")
    axes[0, 1].grid(alpha=0.3)

    # Network vs GPR
    axes[1, 0].scatter(gpr_np, net_mean_np, alpha=0.5, s=20, color="green")
    axes[1, 0].errorbar(gpr_np[::10], net_mean_np[::10], yerr=net_std_np[::10],
                       fmt='none', elinewidth=0.5, capsize=1, alpha=0.3)
    mn, mx = min(gpr_np.min(), net_mean_np.min()), max(gpr_np.max(), net_mean_np.max())
    axes[1, 0].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
    axes[1, 0].set_xlabel("GPR Output")
    axes[1, 0].set_ylabel("Network Output")
    axes[1, 0].set_title(f"Network vs GPR (corr={correlation:.4f})")
    axes[1, 0].grid(alpha=0.3)

    # Residual histogram
    axes[1, 1].hist(residuals, bins=40, alpha=0.8, edgecolor='black')
    axes[1, 1].axvline(residuals.mean(), color='r', linestyle='--', lw=2, label=f"mean={residuals.mean():.2e}")
    axes[1, 1].set_xlabel("Residual (Network - GPR)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Linear FCN2 vs GPR (linear kernel)\n{run_dir.name} | d={d} | σ₀²={args.sigma0**2:.2e}",
                fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    output_path = run_dir / "linear_vs_gpr_linear_kernel.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    # Save results JSON
    results = {
        "d": d,
        "P_config": P_cfg,
        "num_test_samples": args.num_samples,
        "sigma0_sq": float(args.sigma0**2),
        "network_output": {
            "mean": float(net_mean_np.mean()),
            "std": float(net_mean_np.std()),
            "min": float(net_mean_np.min()),
            "max": float(net_mean_np.max()),
        },
        "gpr_output": {
            "mean": float(gpr_np.mean()),
            "std": float(gpr_np.std()),
            "min": float(gpr_np.min()),
            "max": float(gpr_np.max()),
        },
        "target_y": {
            "mean": float(Y_np.mean()),
            "std": float(Y_np.std()),
        },
        "residuals": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "max_abs": float(np.abs(residuals).max()),
        },
        "correlation": float(correlation),
        "mse": float(mse),
        "mae": float(mae),
    }

    results_path = run_dir / "linear_vs_gpr_linear_kernel_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
