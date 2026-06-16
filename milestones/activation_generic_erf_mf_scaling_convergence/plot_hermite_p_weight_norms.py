#!/usr/bin/env python3
"""Plot per-neuron weight norms for Hermite runs at fixed P.

Default behavior targets P=1500 runs under ./hermite_activation and
produces one figure per run directory plus an aggregate figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    }
)


def _extract_p_from_name(path: Path) -> Optional[int]:
    m = re.search(r"_P(\d+)_", path.name)
    if m is None:
        return None
    return int(m.group(1))


def find_target_run_dirs(results_root: Path, target_p: int) -> List[Path]:
    run_dirs: List[Path] = []
    for run_dir in sorted(results_root.glob("d*_P*_N*_chi*_kappa*")):
        if not run_dir.is_dir():
            continue
        p_val = _extract_p_from_name(run_dir)
        if p_val != target_p:
            continue
        seed_dirs = [d for d in sorted(run_dir.glob("seed*")) if d.is_dir()]
        if seed_dirs:
            run_dirs.append(run_dir)
    return run_dirs


def _load_state_dict(seed_dir: Path) -> Optional[Dict[str, torch.Tensor]]:
    ckpt = seed_dir / "model_final.pt"
    if not ckpt.exists():
        ckpt = seed_dir / "model.pt"
    if not ckpt.exists():
        return None

    data = torch.load(ckpt, map_location="cpu")
    if isinstance(data, dict) and "model_state_dict" in data:
        data = data["model_state_dict"]
    if not isinstance(data, dict):
        return None

    for key in ("W0", "W1", "A"):
        if key not in data:
            return None

    # Older checkpoints may have an extra leading singleton batch dimension.
    if data["W0"].ndim == 4:
        data["W0"] = data["W0"].squeeze(0)
    if data["W1"].ndim == 4:
        data["W1"] = data["W1"].squeeze(0)
    if data["A"].ndim == 3:
        data["A"] = data["A"].squeeze(0)

    return data


def neuron_norms_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    w0 = state_dict["W0"].detach().to(torch.float32)
    w1 = state_dict["W1"].detach().to(torch.float32)
    a = state_dict["A"].detach().to(torch.float32)

    # Shapes expected: W0 (ens, N1, d), W1 (ens, N2, N1), A (ens, N2, 1 or ens, N2)
    w0_norms = torch.linalg.vector_norm(w0, dim=-1).reshape(-1).cpu().numpy()
    w1_norms = torch.linalg.vector_norm(w1, dim=-1).reshape(-1).cpu().numpy()
    a_abs = a.squeeze(-1).abs().reshape(-1).cpu().numpy()

    return {"W0": w0_norms, "W1": w1_norms, "A": a_abs}


def percentile_summary(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def plot_seed_norms(seed_name: str, norms: Dict[str, np.ndarray], out_path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keys = ["W0", "W1", "A"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for ax, key, color in zip(axes, keys, colors):
        vals = norms[key]
        vals = vals[np.isfinite(vals)]
        vals_sorted = np.sort(vals)
        idx = np.arange(vals_sorted.size)
        ax.plot(idx, vals_sorted, color=color, linewidth=1.4)
        if vals_sorted.size > 0:
            p99 = np.percentile(vals_sorted, 99)
            vmax = vals_sorted[-1]
            ax.axhline(p99, linestyle="--", color="gray", alpha=0.8, linewidth=1.0, label="p99")
            ax.axhline(vmax, linestyle=":", color="black", alpha=0.7, linewidth=1.0, label="max")
            ax.legend(loc="upper left")
        ax.set_xlabel("Neuron index (sorted)")
        ax.set_ylabel("Norm magnitude")
        ax.set_title(f"{key} per-neuron norms")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix} | {seed_name}", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_norms(all_norms: Dict[str, List[np.ndarray]], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keys = ["W0", "W1", "A"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for ax, key, color in zip(axes, keys, colors):
        if not all_norms[key]:
            ax.set_title(f"{key} (no data)")
            continue

        vals = np.concatenate(all_norms[key], axis=0)
        vals = vals[np.isfinite(vals)]
        vals_sorted = np.sort(vals)
        idx = np.arange(vals_sorted.size)

        ax.plot(idx, vals_sorted, color=color, linewidth=1.4)
        p99 = np.percentile(vals_sorted, 99)
        vmax = vals_sorted[-1]
        ax.axhline(p99, linestyle="--", color="gray", alpha=0.8, linewidth=1.0, label="p99")
        ax.axhline(vmax, linestyle=":", color="black", alpha=0.7, linewidth=1.0, label="max")
        ax.legend(loc="upper left")
        ax.set_xlabel("Neuron index (sorted)")
        ax.set_ylabel("Norm magnitude")
        ax.set_title(f"{key} aggregate norms")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hermite per-neuron weight norms for a target P value.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "hermite_activation",
        help="Root directory containing Hermite run folders",
    )
    parser.add_argument("--P", type=int, default=1500, help="Target dataset size P")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / "hermite_weight_norms",
        help="Directory to write plots and summary JSON",
    )
    args = parser.parse_args()

    run_dirs = find_target_run_dirs(args.results_root, args.P)
    if not run_dirs:
        print(f"No Hermite run directories found for P={args.P} under {args.results_root}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"Processing {run_name}")

        seed_dirs = [d for d in sorted(run_dir.glob("seed*")) if d.is_dir()]
        aggregate: Dict[str, List[np.ndarray]] = {"W0": [], "W1": [], "A": []}
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}

        for seed_dir in seed_dirs:
            state_dict = _load_state_dict(seed_dir)
            if state_dict is None:
                print(f"  Skipping {seed_dir.name}: missing or unreadable checkpoint")
                continue

            norms = neuron_norms_from_state_dict(state_dict)
            for key in aggregate:
                aggregate[key].append(norms[key])

            summary[seed_dir.name] = {
                "W0": percentile_summary(norms["W0"]),
                "W1": percentile_summary(norms["W1"]),
                "A": percentile_summary(norms["A"]),
            }

            seed_out = args.out_dir / run_name / f"{seed_dir.name}_weight_norms.png"
            plot_seed_norms(seed_dir.name, norms, seed_out, title_prefix=run_name)
            print(f"  Wrote {seed_out}")

        agg_out = args.out_dir / run_name / "aggregate_weight_norms.png"
        plot_aggregate_norms(
            aggregate,
            agg_out,
            title=f"{run_name} | Aggregate per-neuron norms across seeds",
        )
        print(f"  Wrote {agg_out}")

        summary["aggregate"] = {
            "W0": percentile_summary(np.concatenate(aggregate["W0"], axis=0)) if aggregate["W0"] else {"count": 0},
            "W1": percentile_summary(np.concatenate(aggregate["W1"], axis=0)) if aggregate["W1"] else {"count": 0},
            "A": percentile_summary(np.concatenate(aggregate["A"], axis=0)) if aggregate["A"] else {"count": 0},
        }

        summary_path = args.out_dir / run_name / "weight_norm_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()
