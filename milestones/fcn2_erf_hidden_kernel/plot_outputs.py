#!/usr/bin/env python3
"""
Plot network outputs against target to visualize alignment.

Usage:
    python plot_outputs.py --run-dir <path_to_run_dir> [--device cpu] [--seed 42] [--num-samples P]

The script loads the trained checkpoint (checkpoint.pt or model.pt + config.json),
regenerates random Gaussian inputs, and plots outputs vs targets (Y = X[:,0]).
"""

import argparse
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def load_model(run_dir: Path, device: torch.device):
    """Load model and config from a run directory."""
    # Prefer full checkpoint if available
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
        # Allow config override from checkpoint
        if config is None:
            config = checkpoint.get("config", None)
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location=device)

    if config is None:
        raise FileNotFoundError("config.json or checkpoint config not found in run directory")

    d = int(config["d"])
    P = int(config["P"])
    N = int(config["N"])
    ens = int(config.get("ens", 50))

    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Plot network outputs vs target")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda:X)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input generation")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples (defaults to config P)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")

    model, config = load_model(run_dir, device)

    d = int(config["d"])
    P_cfg = int(config["P"])
    P = args.num_samples if args.num_samples is not None else P_cfg

    torch.manual_seed(args.seed)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0]

    with torch.no_grad():
        outputs = model(X)  # (P, ens)
        outputs_mean = outputs.mean(dim=1)

    # Move to cpu for plotting
    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    outputs_cpu = outputs.cpu()
    outputs_mean_cpu = outputs_mean.cpu()

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter of mean output vs target
    ax.scatter(Y_cpu.numpy(), outputs_mean_cpu.numpy(), s=15, alpha=0.7, label="Mean over ensembles")

    # Optional: overlay a few ensemble members if P is small
    if outputs_cpu.shape[1] <= 5:
        for i in range(outputs_cpu.shape[1]):
            ax.scatter(Y_cpu.numpy(), outputs_cpu[:, i].numpy(), s=8, alpha=0.4, label=f"Ensemble {i}")

    # y=x line
    y_min = min(Y_cpu.min().item(), outputs_mean_cpu.min().item())
    y_max = max(Y_cpu.max().item(), outputs_mean_cpu.max().item())
    ax.plot([y_min, y_max], [y_min, y_max], "k--", lw=1, label="y = x")

    ax.set_xlabel("Target (Y = X[:,0])")
    ax.set_ylabel("Network output")
    ax.set_title(f"Output vs Target\n{run_dir.name} | P={P} | d={d} | N={config['N']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = run_dir / "output_vs_target.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Residual histogram
    residuals = (outputs_mean_cpu - Y_cpu).numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.set_xlabel("Residual (output_mean - target)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    out_path_hist = run_dir / "output_residual_hist.png"
    fig.savefig(out_path_hist, dpi=150)
    plt.close(fig)

    print(f"Saved plots to:\n  {run_dir / 'output_vs_target.png'}\n  {run_dir / 'output_residual_hist.png'}")


if __name__ == "__main__":
    main()
