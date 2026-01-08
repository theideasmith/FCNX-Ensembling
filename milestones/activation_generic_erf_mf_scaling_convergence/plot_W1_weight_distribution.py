#!/usr/bin/env python3
"""Plot W1 weight distributions for all discovered runs.

Scans run folders (d<d>_P<P>_N<N>_chi<chi>/seed<seed>/model.pt), loads each model,
and saves a two-panel histogram to <run>/plots/W1_weight_distribution.png:
- Left: flattened W1[:, :, 0] (first input column into layer 2)
- Right: mean across columns 1: (or all columns if only one), flattened

Usage:
    python plot_W1_weight_distribution.py --base-dir <runs_root> [--dims 150] [--suffix "_something"]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def plot_W1_weight_distribution(
    run_dir: Path,
    cfg: Dict[str, int],
    model: FCN3NetworkActivationGeneric,
    out_path: Path,
) -> None:
    """Plot distribution of W1 weights analogous to W0 plot in plot_full_spectrum."""
    try:
        W1 = model.W1.detach().cpu().numpy()  # shape (ens, n2, n1)

        w1_feature0 = W1[:, :, 0].flatten()
        if W1.shape[2] > 1:
            w1_mean_across_input = W1[:, :, 1:].mean(axis=2).flatten()
            mean_label = 'W1.mean over input cols 1:'
        else:
            w1_mean_across_input = W1.mean(axis=2).flatten()
            mean_label = 'W1.mean over input cols (single col)'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(w1_feature0, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', density=True)
        axes[0].set_xlabel('Weight value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(
            f'W1[:, :, 0] distribution\nVar={w1_feature0.var():.4e}, Mean={w1_feature0.mean():.4e}'
        )
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(w1_mean_across_input, bins=50, color='tab:orange', alpha=0.7, edgecolor='black', density=True)
        axes[1].set_xlabel('Weight value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(
            f'{mean_label}\nVar={w1_mean_across_input.var():.4e}, Mean={w1_mean_across_input.mean():.4e}'
        )
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'W1 Weight Distributions (d={cfg["d"]}, P={cfg["P"]}, N={cfg["N"]}, chi={cfg["chi"]})',
            fontsize=12
        )
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W1 weight distribution plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot W1 weight distribution for {run_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot W1 weight distributions for runs.")
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

    for run_dir, cfg in runs:
        model = load_model(run_dir, cfg, device)
        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue
        out_path = run_dir / "plots" / "W1_weight_distribution.png"
        plot_W1_weight_distribution(run_dir, cfg, model, out_path)


if __name__ == "__main__":
    main()
