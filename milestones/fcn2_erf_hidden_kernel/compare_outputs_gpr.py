#!/usr/bin/env python3
"""
Compare FCN2 network outputs to GPR predictions using the arcsin kernel.

Usage:
    python compare_outputs_gpr.py --run-dir <path_to_run_dir> [--sigma0 0.0] [--seed 42] [--num-samples P] [--num-datasets 5]

- Loads a trained FCN2 model from the run directory (checkpoint.pt or model.pt + config.json).
- Regenerates Gaussian inputs X ~ N(0, I) and targets Y = X[:, 0] for each dataset.
- Computes:
    * Network outputs (mean over ensembles)
    * GPR predictions using the arcsin kernel with noise variance sigma0^2
- Supports averaging over multiple independent datasets for aggregated plots and summary stats.
- Produces scatter plot (GPR vs model) and residual histogram.
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    return (2 / torch.pi) * torch.arcsin(arg)


def arcsin_gpr_predict(X: torch.Tensor, Y: torch.Tensor, sigma0_sq: float) -> torch.Tensor:
    """Simple GPR prediction with arcsin kernel on training data."""
    K = arcsin_kernel(X)
    n = K.shape[0]
    K_reg = K + sigma0_sq * torch.eye(n, device=X.device)
    alpha = torch.linalg.solve(K_reg, Y)
    return K @ alpha


def eval_dataset(model: FCN2NetworkActivationGeneric, P: int, d: int, sigma0_sq: float, seed: int):
    """Generate one dataset, run model and GPR, return numpy arrays."""
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=torch.device("cuda:1")).to(torch.device('cpu'))
    Y = X[:, 0]

    with torch.no_grad():
        outputs = model(X)  # (P, ens) on CPU
        outputs_mean = outputs.mean(dim=1)
        outputs_std = outputs.std(dim=1) / np.sqrt(model.ens)
        gpr_pred = arcsin_gpr_predict(X, Y, sigma0_sq)

    return {
        "X0": X[:, 0].numpy(),
        "Y": Y.numpy(),
        "outputs_mean": outputs_mean.numpy(),
        "outputs_std": outputs_std.numpy(),
        "gpr": gpr_pred.numpy(),
    }


def load_model(run_dir: Path, device: torch.device):
    """Load model and config from a run directory."""
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
    parser = argparse.ArgumentParser(description="Compare FCN2 outputs to arcsin GPR")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:X)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input generation")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples (defaults to config P)")
    parser.add_argument("--sigma0", type=float, default=1.0, help="Observation noise std (sigma0^2 added to kernel diag)")
    parser.add_argument("--num-datasets", type=int, default=1, help="Number of independent test datasets to average over")
    parser.add_argument("--num-bins", type=int, default=30, help="Number of bins for X[:,0] aggregation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Load model on CPU for stability in kernel inversion
    model, config = load_model(run_dir, torch.device("cpu"))

    d = int(config["d"])
    P_cfg = int(config["P"])
    P = args.num_samples if args.num_samples is not None else P_cfg
    sigma0_sq = args.sigma0 

    datasets = []
    seeds = []
    for i in range(args.num_datasets):
        ds_seed = args.seed + i
        print("Generating dataset", i, "with seed", ds_seed)
        seeds.append(ds_seed)
        datasets.append(eval_dataset(model, P, d, sigma0_sq, seed=ds_seed))

    Y_np = np.concatenate([ds["Y"] for ds in datasets])
    outputs_mean_np = np.concatenate([ds["outputs_mean"] for ds in datasets])
    outputs_std_np = np.concatenate([ds["outputs_std"] for ds in datasets])
    gpr_np = np.concatenate([ds["gpr"] for ds in datasets])
    X0_np = np.concatenate([ds["X0"] for ds in datasets])

    residuals = outputs_mean_np - gpr_np
    per_dataset_stats = []
    for ds_idx, ds in enumerate(datasets):
        ds_resid = ds["outputs_mean"] - ds["gpr"]
        per_dataset_stats.append({
            "dataset_index": ds_idx,
            "seed": seeds[ds_idx],
            "residual_mean": float(ds_resid.mean()),
            "residual_std": float(ds_resid.std()),
            "rmse": float(np.sqrt(np.mean(ds_resid ** 2))),
        })

    stats = {
        "num_datasets": args.num_datasets,
        "seeds": seeds,
        "P": P,
        "d": d,
        "N": int(config["N"]),
        "sigma0_sq": sigma0_sq,
        "overall": {
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        },
        "per_dataset": per_dataset_stats,
    }

    bins = np.linspace(X0_np.min(), X0_np.max(), args.num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(X0_np, bins) - 1
    num_bins = len(bin_centers)
    bin_counts = np.zeros(num_bins, dtype=int)
    model_mean_bins = np.full(num_bins, np.nan)
    model_sem_bins = np.full(num_bins, np.nan)
    gpr_mean_bins = np.full(num_bins, np.nan)
    resid_mean_bins = np.full(num_bins, np.nan)
    resid_std_bins = np.full(num_bins, np.nan)

    for b in range(num_bins):
        mask = bin_idx == b
        count = mask.sum()
        bin_counts[b] = int(count)
        if count == 0:
            continue
        model_vals = outputs_mean_np[mask]
        gpr_vals = gpr_np[mask]
        resid_vals = residuals[mask]
        model_mean_bins[b] = model_vals.mean()
        model_sem_bins[b] = model_vals.std(ddof=0) / np.sqrt(count)
        gpr_mean_bins[b] = gpr_vals.mean()
        resid_mean_bins[b] = resid_vals.mean()
        resid_std_bins[b] = resid_vals.std(ddof=0)

    stats["binned"] = {
        "bin_centers": bin_centers.tolist(),
        "bin_counts": bin_counts.tolist(),
        "model_mean": np.nan_to_num(model_mean_bins, nan=float("nan")).tolist(),
        "model_sem": np.nan_to_num(model_sem_bins, nan=float("nan")).tolist(),
        "gpr_mean": np.nan_to_num(gpr_mean_bins, nan=float("nan")).tolist(),
        "residual_mean": np.nan_to_num(resid_mean_bins, nan=float("nan")).tolist(),
        "residual_std": np.nan_to_num(resid_std_bins, nan=float("nan")).tolist(),
    }

    # Scatter plot: GPR vs model
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(gpr_np, outputs_mean_np, yerr=outputs_std_np, fmt='o', markersize=4,
                alpha=0.7, elinewidth=0.8, capsize=2.0,
                label=f"Model mean ± ensemble std (all samples across {args.num_datasets} datasets)")
    mn = min(gpr_np.min(), outputs_mean_np.min())
    mx = max(gpr_np.max(), outputs_mean_np.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="y = x")
    ax.set_xlabel("GPR arcsin prediction")
    ax.set_ylabel("Model output (mean over ensembles)")
    ax.set_title(f"GPR vs Model\n{run_dir.name} | P={P} | d={d} | N={config['N']} | datasets={args.num_datasets}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = run_dir / "gpr_vs_model.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Residual histogram (model - gpr)
    residuals = outputs_mean_np - gpr_np
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.set_xlabel("Residual (model - gpr)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (all samples)")
    fig.tight_layout()
    out_path_hist = run_dir / "gpr_model_residual_hist.png"
    fig.savefig(out_path_hist, dpi=150)
    plt.close(fig)

    # Model and GPR against X

    fig, ax = plt.subplots(figsize=(8, 4))
    Idx = np.argsort(X0_np)
    X0_sorted = X0_np[Idx]
    outputs_mean_np_sorted = outputs_mean_np[Idx]
    gpr_np_sorted = gpr_np[Idx]

    ax.errorbar(X0_sorted, outputs_mean_np_sorted, yerr=outputs_std_np[Idx], fmt='o', markersize=4, color='blue', ls=':',
        alpha=0.7, label="Model mean ± ensemble std (all samples)")
    ax.plot(X0_sorted, gpr_np_sorted, 'o', markersize=4, ls='--', color='orange',
        alpha=0.7, label="GPR prediction (all samples)")
    ax.set_title(f"Model and GPR outputs vs X[:, 0]\n{run_dir.name} | P={P} | d={d} | N={config['N']} | datasets={args.num_datasets}")
    ax.legend()
    ax.set_xlabel("X[:, 0]")
    ax.set_ylabel("Value")
    fig.tight_layout()
    out_path_vs_x0 = run_dir / "gpr_vs_model_vs_x0.png"
    fig.savefig(out_path_vs_x0, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    valid = ~np.isnan(model_mean_bins) & ~np.isnan(gpr_mean_bins)
    ax.errorbar(bin_centers[valid], model_mean_bins[valid], yerr=model_sem_bins[valid], fmt='o', markersize=5, color='blue',
                alpha=0.8, label="Model binned mean ± SEM")
    ax.plot(bin_centers[valid], gpr_mean_bins[valid], 's--', markersize=5, color='orange', alpha=0.8, label="GPR binned mean")
    ax.set_title(f"Binned averages vs X[:, 0]\n{run_dir.name} | bins={args.num_bins} | datasets={args.num_datasets}")
    ax.set_xlabel("X[:, 0] bin center")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path_binned = run_dir / "gpr_vs_model_binned.png"
    fig.savefig(out_path_binned, dpi=150)
    plt.close(fig)

    stats_path = run_dir / "gpr_compare_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        "Saved plots to:\n"
        f"  {out_path}\n"
        f"  {out_path_hist}\n"
        f"  {out_path_vs_x0}\n"
        f"  {out_path_binned}\n"
        "Saved stats to:\n"
        f"  {stats_path}"
    )

if __name__ == "__main__":
    main()
