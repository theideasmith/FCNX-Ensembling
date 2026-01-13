#!/usr/bin/env python3
"""Test script: plot ONLY the Gaussian actions from theory eigenvalues.

Calls Julia eos_fcn3erf.jl to fetch lH1T, lH1P, lH3T, lH3P and plots the
associated -log p(x) curves for N(0, var) at each eigenvalue variance.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_action(x: np.ndarray, var: float) -> np.ndarray:
    """Compute -log p(x) for N(0, var)."""
    if var is None or not np.isfinite(var) or var <= 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    var = np.float64(var)
    return x**2 / (2.0 * var) + 0.5 * np.log(2.0 * np.pi * var)


def fetch_theory(d: int, P: int, N: int, chi: float, kappa: float = 2.0, eps: float = 0.03) -> dict:
    """Call Julia eos_fcn3erf.jl and parse theory eigenvalues."""
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--n2={N}",
        f"--chi={chi}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={to_path}",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(to_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Julia solver failed: {e}")
        return {}
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass

    tgt = data.get("target", {})
    perp = data.get("perpendicular", {})

    return {
        "lH1T": tgt.get("lH1T"),
        "lH1P": perp.get("lH1P"),
        "lH3T": tgt.get("lH3T"),
        "lH3P": perp.get("lH3P"),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian action curves from theory eigenvalues")
    parser.add_argument("--d", type=int, default=150, help="Dimension d")
    parser.add_argument("--P", type=int, default=1200, help="Number of samples P")
    parser.add_argument("--N", type=int, default=1600, help="Hidden layer size N")
    parser.add_argument("--chi", type=float, default=80.0, help="Chi parameter")
    parser.add_argument("--kappa", type=float, default=2.0, help="Kappa parameter")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Epsilon parameter")
    parser.add_argument("--x-range", type=float, default=5.0, help="Range for x-axis (±)")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: test_gaussian_actions.png)")
    args = parser.parse_args()

    # Fetch theory
    print(f"Fetching theory for d={args.d}, P={args.P}, N={args.N}, chi={args.chi}...")
    theory = fetch_theory(args.d, args.P, args.N, args.chi, args.kappa, args.epsilon)

    if not theory:
        print("Error: failed to get theory eigenvalues")
        return

    print(f"Theory eigenvalues:")
    for key, val in theory.items():
        print(f"  {key}: {val}")

    # Generate x range for curves
    x_range = np.float64(args.x_range)
    x = np.linspace(-x_range, x_range, 500, dtype=np.float64)

    # Create 2x4 subplot (linear and log scales side by side)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))

    pairs = [
        ("lH1T", "Target linear (lH1T)", "royalblue"),
        ("lH1P", "Perpendicular linear (lH1P)", "orange"),
        ("lH3T", "Target cubic (lH3T)", "tab:green"),
        ("lH3P", "Perpendicular cubic (lH3P)", "tab:brown"),
    ]

    for col, (key, label, color) in enumerate(pairs):
        var = theory.get(key)
        if var is None or not np.isfinite(var) or var <= 0:
            # Linear scale plot
            ax_lin = axes[0, col]
            ax_lin.text(0.5, 0.5, f"{key} = {var}", ha="center", va="center", fontsize=12, transform=ax_lin.transAxes)
            ax_lin.set_title(f"{label} (linear) [invalid]", fontsize=11)
            ax_lin.grid(True, alpha=0.3)
            # Log scale plot
            ax_log = axes[1, col]
            ax_log.text(0.5, 0.5, f"{key} = {var}", ha="center", va="center", fontsize=12, transform=ax_log.transAxes)
            ax_log.set_title(f"{label} (log) [invalid]", fontsize=11)
            ax_log.grid(True, alpha=0.3)
            continue

        action = gaussian_action(x, var)
        
        # Linear scale plot (top row)
        ax_lin = axes[0, col]
        ax_lin.plot(x, action, color=color, linewidth=2.5)
        ax_lin.set_title(f"{label} (linear): var={var:.3g}", fontsize=11)
        ax_lin.set_xlabel("x")
        ax_lin.set_ylabel("-log p(x)")
        ax_lin.grid(True, alpha=0.3)
        
        # Log scale plot (bottom row)
        ax_log = axes[1, col]
        ax_log.plot(x, action, color=color, linewidth=2.5)
        ax_log.set_title(f"{label} (log): var={var:.3g}", fontsize=11)
        ax_log.set_xlabel("x")
        ax_log.set_ylabel("-log p(x)")
        ax_log.set_yscale("log")
        ax_log.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"Gaussian Actions from Theory Eigenvalues\n(d={args.d}, P={args.P}, N={args.N}, χ={args.chi}, κ={args.kappa})",
        fontsize=14,
    )
    fig.tight_layout()

    out_path = args.out or Path(__file__).parent / "test_gaussian_actions.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)

    # Additional: overlay linear and cubic curves separately
    fig_overlay, axes_overlay = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear projections: lH1T and lH1P
    lin_labels = {
        "lH1T": r"$S[h_{He1}^{*}]$",
        "lH1P": r"$S[h_{He1}^{\perp}]$",
    }
    for key, latex_label in lin_labels.items():
        var = theory.get(key)
        if var is None or not np.isfinite(var) or var <= 0:
            continue
        action = gaussian_action(x, var)
        axes_overlay[0].plot(x, action, linewidth=2.5, label=f"{latex_label} (var={var:.3g})")
    
    axes_overlay[0].set_xlabel("x", fontsize=12)
    axes_overlay[0].set_ylabel("-log p(x)", fontsize=12)
    axes_overlay[0].set_title(r"$h_{He1}$ projections", fontsize=12)
    axes_overlay[0].grid(True, alpha=0.3)
    axes_overlay[0].legend(fontsize=10)
    
    # Cubic projections: lH3T and lH3P
    cubic_labels = {
        "lH3T": r"$S[h_{He3}^{*}]$",
        "lH3P": r"$S[h_{He3}^{\perp}]$",
    }
    for key, latex_label in cubic_labels.items():
        var = theory.get(key)
        if var is None or not np.isfinite(var) or var <= 0:
            continue
        action = gaussian_action(x, var)
        axes_overlay[1].plot(x, action, linewidth=2.5, label=f"{latex_label} (var={var:.3g})")
    
    axes_overlay[1].set_xlabel("x", fontsize=12)
    axes_overlay[1].set_title(r"$h_{He3}$ projections", fontsize=12)
    axes_overlay[1].grid(True, alpha=0.3)
    axes_overlay[1].legend(fontsize=10)
    
    fig_overlay.suptitle(
        r"Gaussian Actions for Eigenfeatures $h_k$" + f"\n(d={args.d}, P={args.P}, N={args.N}, χ={args.chi}, κ={args.kappa})",
        fontsize=13,
    )
    fig_overlay.tight_layout()
    out_overlay = Path(__file__).parent / "test_gaussian_actions_overlay.png"
    fig_overlay.savefig(out_overlay, dpi=200)
    print(f"Saved overlay plot to {out_overlay}")
    plt.close(fig_overlay)


if __name__ == "__main__":
    main()
