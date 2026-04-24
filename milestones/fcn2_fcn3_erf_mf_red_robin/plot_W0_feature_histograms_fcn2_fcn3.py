#!/usr/bin/env python3
"""Plot W0 feature histograms for FCN2/FCN3 red-robin models.

Creates one large figure containing 16 histograms total:
- 8 models (4 FCN2 + 4 FCN3)
- 2 histograms per model: W0[:, :, 0] and W0[:, :, 1:]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_FCN2_SCAN_DIR = Path(__file__).resolve().parent / "results_fcn2_erf"
DEFAULT_FCN3_SCAN_DIR = Path(__file__).resolve().parent / "results_fcn3_erf"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    match = re.search(
        r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi(?P<chi>[-+]?\d*\.?\d+)_kappa(?P<kappa>[-+]?\d*\.?\d+)",
        run_dir.parent.name,
    )
    if not match:
        raise FileNotFoundError(f"No config.json found and run name did not match expected pattern: {run_dir}")
    return {
        "d": int(match.group("d")),
        "P": int(match.group("P")),
        "N": int(match.group("N")),
        "chi": float(match.group("chi")),
        "kappa": float(match.group("kappa")),
        "seed": int(re.search(r"seed(\d+)", run_dir.name).group(1)) if re.search(r"seed(\d+)", run_dir.name) else 0,
    }


def find_seed_dirs(scan_dir: Path) -> List[Path]:
    candidates = list(scan_dir.glob("**/seed*/model_final.pt")) + list(scan_dir.glob("**/seed*/model.pt"))
    return sorted({p.parent.resolve() for p in candidates})


def find_checkpoint_file(seed_dir: Path) -> Path:
    candidates = [seed_dir / "model_final.pt", seed_dir / "model.pt", seed_dir / "checkpoint.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def normalize_state_dict(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state and "W0" not in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")
    if "W0" in state and state["W0"].ndim == 4:
        state["W0"] = state["W0"].squeeze(0)
    return state


def choose_four_runs(seed_dirs: List[Path]) -> List[Path]:
    """Pick up to four representative runs by evenly spacing unique P values.

    Preference is seed0 for each P when available.
    """
    by_p: Dict[int, List[Path]] = {}
    for seed_dir in seed_dirs:
        try:
            cfg = load_run_config(seed_dir)
            P = int(cfg["P"])
            by_p.setdefault(P, []).append(seed_dir)
        except Exception:
            continue

    if not by_p:
        return []

    sorted_ps = sorted(by_p.keys())
    if len(sorted_ps) <= 4:
        selected_ps = sorted_ps
    else:
        idx = np.linspace(0, len(sorted_ps) - 1, num=4)
        selected_ps = sorted({sorted_ps[int(round(i))] for i in idx})
        if len(selected_ps) < 4:
            for p in sorted_ps:
                if p not in selected_ps:
                    selected_ps.append(p)
                if len(selected_ps) == 4:
                    break
        selected_ps = sorted(selected_ps[:4])

    selected_runs: List[Path] = []
    for p in selected_ps:
        runs = by_p[p]
        runs_sorted = sorted(runs, key=lambda x: load_run_config(x).get("seed", 10**9))
        selected_runs.append(runs_sorted[0])
    return selected_runs


def extract_weight_slices(seed_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    checkpoint = find_checkpoint_file(seed_dir)
    cfg = load_run_config(seed_dir)
    state = torch.load(checkpoint, map_location="cpu")
    state = normalize_state_dict(state)

    if "W0" not in state:
        raise KeyError(f"W0 not found in checkpoint for {seed_dir}")

    W0 = state["W0"]
    if not torch.is_tensor(W0):
        W0 = torch.tensor(W0)

    if W0.ndim != 3:
        raise ValueError(f"Expected W0 with shape (ens, N, d), got {tuple(W0.shape)} in {seed_dir}")
    if W0.shape[-1] < 2:
        raise ValueError(f"Expected d >= 2 for W0 slices, got d={W0.shape[-1]} in {seed_dir}")

    target = W0[:, :, 0].detach().cpu().numpy().reshape(-1)
    perp = W0[:, :, 1:].detach().cpu().numpy().reshape(-1)
    return target, perp, cfg


def plot_histograms(
    run_specs: List[Tuple[str, Path]],
    out_path: Path,
    bins: int = 120,
    density: bool = True,
) -> None:
    n_models = len(run_specs)
    if n_models == 0:
        raise RuntimeError("No models selected to plot.")

    fig, axes = plt.subplots(nrows=n_models, ncols=2, figsize=(18, max(2 * n_models, 10)), constrained_layout=True)
    if n_models == 1:
        axes = np.array([axes])

    for i, (family, run_dir) in enumerate(run_specs):
        target, perp, cfg = extract_weight_slices(run_dir)

        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        color = "tab:blue" if family == "FCN2" else "tab:red"

        ax_left.hist(target, bins=bins, density=density, alpha=0.75, color=color)
        ax_right.hist(perp, bins=bins, density=density, alpha=0.75, color=color)

        label_prefix = (
            f"{family} | P={int(cfg.get('P', -1))} | d={int(cfg.get('d', -1))} "
            f"| N={int(cfg.get('N', cfg.get('n1', -1)))} | seed={int(cfg.get('seed', 0))}"
        )

        ax_left.set_title(f"{label_prefix} | W0[:, :, 0]")
        ax_right.set_title(f"{label_prefix} | W0[:, :, 1:]")

        for ax in (ax_left, ax_right):
            ax.grid(alpha=0.25)
            ax.set_xlabel("weight value")
            ax.set_ylabel("density" if density else "count")

    fig.suptitle("First hidden-layer weight histograms for FCN2/FCN3 red-robin models", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot 16 W0 histograms (4 FCN2 + 4 FCN3, each with target/perp slices)")
    parser.add_argument("--fcn2-scan-dir", type=str, default=str(DEFAULT_FCN2_SCAN_DIR))
    parser.add_argument("--fcn3-scan-dir", type=str, default=str(DEFAULT_FCN3_SCAN_DIR))
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUTPUT_DIR / "w0_feature_histograms_fcn2_fcn3_16panels.png"))
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--count", action="store_true", help="Plot counts instead of densities")
    args = parser.parse_args(argv)

    fcn2_dirs = find_seed_dirs(Path(args.fcn2_scan_dir).resolve())
    fcn3_dirs = find_seed_dirs(Path(args.fcn3_scan_dir).resolve())
    chosen_fcn2 = choose_four_runs(fcn2_dirs)
    chosen_fcn3 = choose_four_runs(fcn3_dirs)

    if len(chosen_fcn2) < 4 or len(chosen_fcn3) < 4:
        print(
            f"Warning: requested 4+4 models, found {len(chosen_fcn2)} FCN2 and {len(chosen_fcn3)} FCN3. "
            "Plot will include all available runs."
        )

    run_specs: List[Tuple[str, Path]] = [("FCN2", p) for p in chosen_fcn2] + [("FCN3", p) for p in chosen_fcn3]
    out_path = Path(args.out_path).resolve()

    print(f"Selected FCN2 runs: {[str(p) for p in chosen_fcn2]}")
    print(f"Selected FCN3 runs: {[str(p) for p in chosen_fcn3]}")

    plot_histograms(run_specs, out_path=out_path, bins=args.bins, density=not args.count)
    print(f"Saved histogram figure to {out_path}")


if __name__ == "__main__":
    main()
