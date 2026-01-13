#!/usr/bin/env python3
"""
Aggregate eigenvalue trajectories from d-sweep runs and plot over epochs.

- Scans sibling run directories (e.g., d2_P6_N50_chi50) containing eigenvalues_over_time.json
- Computes top eigenvalue and mean/std of remaining eigenvalues at each logged epoch
- Plots both series vs epochs (log scale) for each d
- Saves combined data to d_sweep_eigenvalues_over_epochs.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from Experiment import ExperimentLinear


def find_runs(base_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (d, run_dir) that contain eigenvalue logs."""
    runs: List[Tuple[int, Path]] = []
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        ev_path = run_dir / "eigenvalues_over_time.json"
        if not ev_path.exists():
            continue

        d_val = None
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                d_val = int(cfg.get("d")) if "d" in cfg else None
            except Exception:
                d_val = None
        if d_val is None:
            # Fallback: parse directory name starting with d{d}_
            name = run_dir.name
            if name.startswith("d"):
                num = "".join(ch for ch in name[1:] if ch.isdigit())
                d_val = int(num) if num else None
        if d_val is None:
            continue

        runs.append((d_val, run_dir))

    runs.sort(key=lambda x: x[0])
    return runs


def load_series(ev_path: Path) -> Dict[str, List[float]]:
    """Load eigenvalues_over_time.json and return series for epochs."""
    with open(ev_path, "r") as f:
        raw = json.load(f)

    # Keys are epochs (stringified); sort numerically
    records = []
    for k, vals in raw.items():
        try:
            epoch = int(k)
        except ValueError:
            continue
        eigs = np.array(vals, dtype=float)
        if eigs.size == 0:
            top = np.nan
            mean_rest = np.nan
            std_rest = np.nan
        else:
            eigs_sorted = np.sort(eigs)[::-1]
            top = float(eigs_sorted[0])
            rest = eigs_sorted[1:]
            if rest.size > 0:
                mean_rest = float(rest.mean())
                std_rest = float(rest.std())
            else:
                mean_rest = np.nan
                std_rest = np.nan
        records.append((epoch, top, mean_rest, std_rest))

    records.sort(key=lambda x: x[0])
    epochs = [r[0] for r in records]
    top_series = [r[1] for r in records]
    mean_rest_series = [r[2] for r in records]
    std_rest_series = [r[3] for r in records]

    return {
        "epochs": epochs,
        "top": top_series,
        "mean_rest": mean_rest_series,
        "std_rest": std_rest_series,
    }


def compute_predictions(d: int, P: int, N: int, kappa: float, run_dir: Path) -> Optional[Dict[str, float]]:
    """Compute theoretical predictions using ExperimentLinear."""
    chi = N
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    try:
        exp = ExperimentLinear(
            file=str(run_dir),
            N=N,
            d=d,
            chi=chi,
            P=P,
            ens=50,
            kappa=kappa,
            device=device
        )
        print(exp)

        preds = exp.eig_predictions()
        
        return {
            "lHT": float(preds.lHT) if hasattr(preds, 'lHT') and preds.lHT is not None else None,
            "lHT": float(preds.lHT) if hasattr(preds, 'lHT') and preds.lHT is not None else None,
            "lHP": float(preds.lHP) if hasattr(preds, 'lHP') and preds.lHP is not None else None,
            "lHP": float(preds.lHP) if hasattr(preds, 'lHP') and preds.lHP is not None else None,
        }
    except Exception as e:
        print(f"  Warning: Could not compute predictions for d={d}: {e}")
        return None


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    runs = find_runs(base_dir)
    if not runs:
        print("No d-sweep run directories with eigenvalues_over_time.json found.")
        return

    # Use a colormap with distinct hues for each d
    base_colors = plt.cm.Set1(np.linspace(0, 0.8, len(runs)))
    data_export: Dict[str, Dict[str, List[float]]] = {}

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for idx, (d_val, run_dir) in enumerate(runs):
        ev_path = run_dir / "eigenvalues_over_time.json"
        series = load_series(ev_path)
        data_export[str(d_val)] = series

        epochs = series["epochs"]
        if not epochs:
            continue

        # Use darker shade for top, lighter for mean_rest
        color_dark = base_colors[idx]
        color_light = base_colors[idx] * 0.6 + 0.4  # Lighter version

        # Plot top eigenvalue with markers
        ax.plot(
            epochs,
            series["top"],
            "o-",
            color=color_dark,
            linewidth=0.5,
            markersize=0.5,
            label=f"d={d_val} (top)",
        )

        # Plot mean of rest without markers, with error bars
        ax.errorbar(
            epochs,
            series["mean_rest"],
            yerr=series["std_rest"],
            fmt="-",
            color=color_light,
            linewidth=2,
            alpha=0.85,
            label=f"d={d_val} (mean rest)",
        )

        # Load config to get parameters for predictions
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            
            P = cfg.get("P", 6)
            N = cfg.get("N", 50)
            kappa = cfg.get("kappa", 1.0 / d_val)
            
            # Try to load cached predictions first, otherwise compute
            pred_path = run_dir / "predictions.json"
            preds = None
            # if pred_path.exists():
            #     try:
            #         with open(pred_path, "r") as f:
            #             preds = json.load(f)
            #     except Exception:
            #         pass
            
            # if preds is None:
            if True:
                print(f"  Computing predictions for d={d_val}...")
                preds = compute_predictions(d_val, P, N, kappa, run_dir)
                # Save for future use
                if preds is not None:
                    with open(pred_path, "w") as f:
                        json.dump(preds, f, indent=2)

            print(f"  Predictions for d={d_val}: {preds}")
            if preds is not None:
                xmin, xmax = min(epochs), max(epochs)
                
                # Plot lH1T (training regime) as dashed line
                if preds.get("lHT") is not None:
                    ax.axhline(
                        preds["lHT"],
                        linestyle="--",
                        color=color_dark,
                        linewidth=1.2,
                        alpha=0.6,
                        label=f"d={d_val} lHT (δ=1)",
                    )
                
                # Plot lH1P (population regime) as dashed line
                if preds.get("lHP") is not None:
                    ax.axhline( 
                        preds["lHP"],
                        linestyle=":",
                        color=color_dark,
                        linewidth=1.2,
                        alpha=0.6,
                        label=f"d={d_val} lHP (δ=0)",
                    )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Eigenvalues", fontsize=13)
    ax.set_title("Eigenvalues vs Epochs (d-sweep)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best", ncol=2)

    plt.tight_layout()
    plot_path = plots_dir / "d_sweep_eigenvalues_over_epochs.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    export_path = base_dir / "d_sweep_eigenvalues_over_epochs.json"
    with open(export_path, "w") as f:
        json.dump(data_export, f, indent=2)
    print(f"Saved data export to {export_path}")


if __name__ == "__main__":
    main()
