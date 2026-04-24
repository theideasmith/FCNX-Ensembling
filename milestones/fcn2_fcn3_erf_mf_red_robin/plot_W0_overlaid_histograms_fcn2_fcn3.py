#!/usr/bin/env python3
"""Plot FCN2-vs-FCN3 W0 histograms and cumulants across P.

Figure 1: 8 panels (2 x 4)
- Columns: matched P values (up to 4)
- Row 1: target slice W0[:, :, 0], FCN2 overlaid with FCN3
- Row 2: perpendicular slice W0[:, :, 1:], FCN2 overlaid with FCN3

Figure 2: first three cumulants vs P for target/perpendicular slices.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


def load_run_config(seed_dir: Path) -> dict:
    """Load config.json from a seed directory."""
    config_path = seed_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {seed_dir}")
    with open(config_path) as f:
        return json.load(f)


def find_seed_dirs(results_dir: Path, family: str) -> List[Path]:
    """Find all seed directories for a given family (recursively)."""
    family_dir = results_dir / f"results_{family}_erf"
    if not family_dir.exists():
        return []
    
    # Recursively find seed directories by looking for checkpoint files
    candidates = list(family_dir.glob("**/seed*/model_final.pt")) + list(family_dir.glob("**/seed*/model.pt"))
    seed_dirs = sorted({p.parent.resolve() for p in candidates})
    return list(seed_dirs)


def find_checkpoint_file(seed_dir: Path) -> Path:
    """Find checkpoint file, trying multiple common names."""
    candidates = ["model_final.pt", "model.pt", "checkpoint.pt"]
    for cand in candidates:
        path = seed_dir / cand
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def normalize_state_dict(state_dict: dict) -> dict:
    """Remove 'module.' prefix from keys if present."""
    normalized = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            normalized[key[7:]] = val
        else:
            normalized[key] = val
    return normalized


def choose_four_runs(seed_dirs: List[Path]) -> List[Path]:
    """Select up to four representative runs by evenly spacing unique P values."""
    by_p: Dict[int, List[Path]] = {}
    for seed_dir in seed_dirs:
        cfg = load_run_config(seed_dir)
        P = int(cfg["P"])
        by_p.setdefault(P, []).append(seed_dir)
    
    sorted_ps = sorted(by_p.keys())
    if len(sorted_ps) == 0:
        return []
    
    # Evenly space P values
    if len(sorted_ps) <= 4:
        selected_ps = sorted_ps
    else:
        indices = np.linspace(0, len(sorted_ps) - 1, 4, dtype=int)
        selected_ps = [sorted_ps[i] for i in indices]
    
    selected_runs = []
    for p in selected_ps:
        # Sort by seed name to get seed0 first
        runs_for_p = sorted(by_p[p], key=lambda x: x.name)
        selected_runs.append(runs_for_p[0])
    
    return selected_runs


def choose_matched_runs(
    fcn2_seed_dirs: List[Path],
    fcn3_seed_dirs: List[Path],
    max_p_values: int = 4,
) -> List[Tuple[int, Path, Path]]:
    """Match FCN2 and FCN3 runs by common P and select up to max_p_values."""
    by_p_fcn2: Dict[int, List[Path]] = {}
    by_p_fcn3: Dict[int, List[Path]] = {}

    for seed_dir in fcn2_seed_dirs:
        cfg = load_run_config(seed_dir)
        p_val = int(cfg["P"])
        by_p_fcn2.setdefault(p_val, []).append(seed_dir)

    for seed_dir in fcn3_seed_dirs:
        cfg = load_run_config(seed_dir)
        p_val = int(cfg["P"])
        by_p_fcn3.setdefault(p_val, []).append(seed_dir)

    common_p = sorted(set(by_p_fcn2).intersection(by_p_fcn3))
    if not common_p:
        return []

    if len(common_p) <= max_p_values:
        selected_p = common_p
    else:
        idx = np.linspace(0, len(common_p) - 1, max_p_values, dtype=int)
        selected_p = [common_p[i] for i in idx]

    matched: List[Tuple[int, Path, Path]] = []
    for p_val in selected_p:
        fcn2_run = sorted(by_p_fcn2[p_val], key=lambda x: x.name)[0]
        fcn3_run = sorted(by_p_fcn3[p_val], key=lambda x: x.name)[0]
        matched.append((p_val, fcn2_run, fcn3_run))
    return matched


def extract_weight_slices(seed_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load checkpoint and extract W0 target and perpendicular slices."""
    checkpoint_path = find_checkpoint_file(seed_dir)
    cfg = load_run_config(seed_dir)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = normalize_state_dict(checkpoint)
    
    W0 = state["W0"]  # shape: (ens, N, d)
    W0 = W0.cpu().numpy() if isinstance(W0, torch.Tensor) else W0
    
    # Target slice: first feature direction
    target = W0[:, :, 0].reshape(-1)
    
    # Perpendicular slice: all other directions (if d > 1)
    if W0.shape[2] > 1:
        perp = W0[:, :, 1:].reshape(-1)
    else:
        perp = np.array([])
    
    return target, perp, cfg


def compute_cumulants(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute first three cumulants of a distribution.
    κ₁ = mean
    κ₂ = variance
    κ₃ = third central moment
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    variance = np.var(data, ddof=0)
    third_moment = np.mean((data - mean) ** 3)
    
    return mean, variance, third_moment


def get_eps_from_cfg(cfg: dict, default: float = 0.4) -> float:
    """Return epsilon value from config, supporting both eps and epsilon keys."""
    if "eps" in cfg:
        return float(cfg["eps"])
    if "epsilon" in cfg:
        return float(cfg["epsilon"])
    return float(default)


def plot_histograms_matched(
    matched_runs: List[Tuple[int, Path, Path]],
    out_path: Path,
    bins: int = 120,
    density: bool = True,
) -> Dict[str, List[float]]:
    """Plot FCN2-vs-FCN3 histograms by matched P and return cumulant table."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False, sharey=False)
    colors = {"FCN2": "#1f77b4", "FCN3": "#d62728"}

    cumulants: Dict[str, List[float]] = {
        "P": [],
        "FCN2_target_k1": [],
        "FCN2_target_k2": [],
        "FCN2_target_k3": [],
        "FCN2_perp_k1": [],
        "FCN2_perp_k2": [],
        "FCN2_perp_k3": [],
        "FCN3_target_k1": [],
        "FCN3_target_k2": [],
        "FCN3_target_k3": [],
        "FCN3_perp_k1": [],
        "FCN3_perp_k2": [],
        "FCN3_perp_k3": [],
    }

    # Hide all axes first; only show used panels
    for row in range(2):
        for col in range(4):
            axes[row, col].set_visible(False)

    for col, (p_val, fcn2_run, fcn3_run) in enumerate(matched_runs[:4]):
        ax_target = axes[0, col]
        ax_perp = axes[1, col]
        ax_target.set_visible(True)
        ax_perp.set_visible(True)

        try:
            fcn2_target, fcn2_perp, cfg2 = extract_weight_slices(fcn2_run)
            fcn3_target, fcn3_perp, cfg3 = extract_weight_slices(fcn3_run)
        except Exception as exc:
            ax_target.text(0.5, 0.5, f"Error for P={p_val}:\n{exc}", ha="center", va="center", transform=ax_target.transAxes)
            ax_perp.text(0.5, 0.5, f"Error for P={p_val}:\n{exc}", ha="center", va="center", transform=ax_perp.transAxes)
            continue

        k1, k2, k3 = compute_cumulants(fcn2_target)
        cumulants["P"].append(float(p_val))
        cumulants["FCN2_target_k1"].append(k1)
        cumulants["FCN2_target_k2"].append(k2)
        cumulants["FCN2_target_k3"].append(k3)
        k1, k2, k3 = compute_cumulants(fcn2_perp)
        cumulants["FCN2_perp_k1"].append(k1)
        cumulants["FCN2_perp_k2"].append(k2)
        cumulants["FCN2_perp_k3"].append(k3)
        k1, k2, k3 = compute_cumulants(fcn3_target)
        cumulants["FCN3_target_k1"].append(k1)
        cumulants["FCN3_target_k2"].append(k2)
        cumulants["FCN3_target_k3"].append(k3)
        k1, k2, k3 = compute_cumulants(fcn3_perp)
        cumulants["FCN3_perp_k1"].append(k1)
        cumulants["FCN3_perp_k2"].append(k2)
        cumulants["FCN3_perp_k3"].append(k3)

        ax_target.hist(
            fcn2_target,
            bins=bins,
            density=density,
            alpha=0.5,
            label="FCN2",
            color=colors["FCN2"],
            edgecolor="black",
            linewidth=0.4,
        )
        ax_target.hist(
            fcn3_target,
            bins=bins,
            density=density,
            alpha=0.5,
            label="FCN3",
            color=colors["FCN3"],
            edgecolor="black",
            linewidth=0.4,
        )

        ax_perp.hist(
            fcn2_perp,
            bins=bins,
            density=density,
            alpha=0.5,
            label="FCN2",
            color=colors["FCN2"],
            edgecolor="black",
            linewidth=0.4,
        )
        ax_perp.hist(
            fcn3_perp,
            bins=bins,
            density=density,
            alpha=0.5,
            label="FCN3",
            color=colors["FCN3"],
            edgecolor="black",
            linewidth=0.4,
        )

        kappa2 = float(cfg2.get("kappa", np.nan))
        eps2 = get_eps_from_cfg(cfg2, default=0.4)
        kappa3 = float(cfg3.get("kappa", np.nan))
        eps3 = get_eps_from_cfg(cfg3, default=0.4)

        ax_target.set_title(
            f"Target, P={p_val}\n"
            f"FCN2: kappa={kappa2:.4g}, eps={eps2:.3g} | FCN3: kappa={kappa3:.4g}, eps={eps3:.3g}",
            fontsize=10,
        )
        ax_perp.set_title(
            f"Perpendicular, P={p_val}\n"
            f"FCN2: kappa={kappa2:.4g}, eps={eps2:.3g} | FCN3: kappa={kappa3:.4g}, eps={eps3:.3g}",
            fontsize=10,
        )

        ax_target.set_xlabel("Weight value")
        ax_target.set_ylabel("Density" if density else "Count")
        ax_perp.set_xlabel("Weight value")
        ax_perp.set_ylabel("Density" if density else "Count")

        ax_target.grid(True, alpha=0.3)
        ax_perp.grid(True, alpha=0.3)
        ax_target.legend(fontsize=8, loc="upper right")
        ax_perp.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved overlaid FCN2-vs-FCN3 histogram figure to {out_path}")
    plt.close(fig)
    return cumulants


def plot_cumulants_vs_p(cumulants: Dict[str, List[float]], out_path: Path) -> None:
    """Plot first three cumulants as functions of P."""
    if not cumulants["P"]:
        print("No cumulant data available; skipping cumulant figure.")
        return

    p_vals = np.asarray(cumulants["P"], dtype=float)
    order = np.argsort(p_vals)
    p_vals = p_vals[order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    labels = {
        "FCN2_target": ("FCN2 target", "#1f77b4", "-"),
        "FCN2_perp": ("FCN2 perp", "#1f77b4", "--"),
        "FCN3_target": ("FCN3 target", "#d62728", "-"),
        "FCN3_perp": ("FCN3 perp", "#d62728", "--"),
    }
    cumulant_names = ["k1", "k2", "k3"]
    ylabels = ["k1 (mean)", "k2 (variance)", "k3 (third central moment)"]

    for ax, cname, ylabel in zip(axes, cumulant_names, ylabels):
        for key, (label, color, linestyle) in labels.items():
            arr = np.asarray(cumulants[f"{key}_{cname}"], dtype=float)[order]
            ax.plot(p_vals, arr, marker="o", linestyle=linestyle, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("P")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ylabel} vs P")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved cumulants-vs-P figure to {out_path}")
    plt.close(fig)


def main():
    """Main entry point."""
    results_dir = Path(__file__).parent  # The current directory contains results_fcn2_erf and results_fcn3_erf
    
    # Collect 4 FCN2 and 4 FCN3 runs
    fcn2_seed_dirs = find_seed_dirs(results_dir, "fcn2")
    fcn3_seed_dirs = find_seed_dirs(results_dir, "fcn3")
    
    print(f"Found {len(fcn2_seed_dirs)} FCN2 seed dirs and {len(fcn3_seed_dirs)} FCN3 seed dirs")

    matched_runs = choose_matched_runs(fcn2_seed_dirs, fcn3_seed_dirs, max_p_values=4)
    print(f"Matched {len(matched_runs)} common P values")

    hist_out = Path(__file__).parent / "plots" / "w0_overlaid_fcn2_vs_fcn3_8panels.png"
    cumulants = plot_histograms_matched(matched_runs, hist_out, bins=120, density=True)

    cumulant_out = Path(__file__).parent / "plots" / "w0_cumulants_vs_P_fcn2_vs_fcn3.png"
    plot_cumulants_vs_p(cumulants, cumulant_out)


if __name__ == "__main__":
    main()
