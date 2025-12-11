"""
Analyze chi fixed point eigenvalues and compare with FCSLinear.jl predictions.

This script:
1. Loads empirical eigenvalues from chi_fixed_eigenvalues.json
2. Calls Julia FCSLinear solver via juliacall to get theoretical lHT (delta=1) and lHP (delta=0)
3. Plots empirical vs predicted eigenvalue scaling with chi
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import juliacall
from juliacall import Main as jl

# Ensure PyJulia uses the current Python (required by juliacall)
os.environ.setdefault("PYTHONCALL_JULIA_PYTHON", "yes")

JULIA_FCS_LINEAR_PATH = "/home/akiva/FCNX-Ensembling/julia_lib/FCSLinear.jl"


def ensure_julia_loaded():
    """Include the Julia FCSLinear module once per process."""
    if getattr(ensure_julia_loaded, "_loaded", False):
        return
    jl.include(JULIA_FCS_LINEAR_PATH)
    ensure_julia_loaded._loaded = True


def load_empirical_results(results_path):
    """Load empirical eigenvalue results from JSON."""
    with open(results_path, "r") as f:
        data = json.load(f)

    chi_values = []
    lHT_empirical = []  # training eigenvalue (delta=1)
    lHP_empirical = []  # population eigenvalue (delta=0)
    lHT_std = []  # std dev of eigenvalues
    lHP_std = []  # std dev of eigenvalues
    epochs = []
    losses = []

    for entry in data:
        chi_values.append(entry["chi"])
        eigenvals = entry["eigenvalues"]
        # Compute std dev across all eigenvalues as uncertainty estimate
        std_eig = np.std(eigenvals) if len(eigenvals) > 1 else 0.0
        
        # eigenvalues[0] -> lHT (delta=1, training)
        # eigenvalues[1] -> lHP (delta=0, population)
        lHT_empirical.append(entry["eigenvalues"][0])
        lHP_empirical.append(entry["eigenvalues"][1])
        lHT_std.append(std_eig)
        lHP_std.append(std_eig)
        epochs.append(entry["epochs"])
        losses.append(entry["loss"])

    return {
        "chi": np.array(chi_values),
        "lHT_empirical": np.array(lHT_empirical),
        "lHP_empirical": np.array(lHP_empirical),
        "lHT_std": np.array(lHT_std),
        "lHP_std": np.array(lHP_std),
        "epochs": np.array(epochs),
        "losses": np.array(losses),
    }


def solve_theoretical_eigenvalues(chi_values, d=2, P=6, N=256, kappa=1.0):
    """Compute theoretical lHT (delta=1) and lHP (delta=0) via FCSLinear."""
    ensure_julia_loaded()

    lHT_theoretical = []
    lHP_theoretical = []

    for chi in chi_values:
        initial_guess = [1.0, 1.0]  # [lJ, lH]
        i0 = juliacall.convert(jl.Vector[jl.Float64], initial_guess)

        try:
            # Training regime (delta = 1)
            sol_T = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=float(chi),
                d=float(d),
                kappa=float(kappa),
                delta=1.0,
                n1=float(N),
                n2=float(N),
                P=float(P),
                lr=1e-6,
                max_iter=60_000,
                tol=1e-8,
                verbose=False,
                anneal=True,
                anneal_steps=30_000,
            )

            # Population regime (delta = 0)
            sol_P = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=float(chi),
                d=float(d),
                kappa=float(kappa),
                delta=0.0,
                n1=float(N),
                n2=float(N),
                P=float(P),
                lr=1e-6,
                max_iter=60_000,
                tol=1e-8,
                verbose=False,
                anneal=True,
                anneal_steps=30_000,
            )

            if sol_T is None or sol_P is None:
                raise RuntimeError("nlsolve_solver returned None")

            lHT_theoretical.append(float(sol_T[1]))
            lHP_theoretical.append(float(sol_P[1]))

        except Exception as exc:  # noqa: BLE001
            print(f"Error solving for chi={chi}: {exc}")
            lHT_theoretical.append(np.nan)
            lHP_theoretical.append(np.nan)

    return {
        "lHT_theoretical": np.array(lHT_theoretical),
        "lHP_theoretical": np.array(lHP_theoretical),
    }


def plot_eigenvalue_scaling(empirical, theoretical, output_dir):
    """Plot empirical vs theoretical eigenvalue scaling with chi, normalized so (lHT + lHP)/2 = 1."""
    chi = empirical["chi"]

    # Normalize: (lHT + lHP) / 2 = 1 for both theory and experiment
    emp_mean = (empirical["lHT_empirical"] + empirical["lHP_empirical"]) / 2
    theo_mean = (theoretical["lHT_theoretical"] + theoretical["lHP_theoretical"]) / 2
    
    lHT_emp_norm = empirical["lHT_empirical"] / emp_mean
    lHP_emp_norm = empirical["lHP_empirical"] / emp_mean
    lHT_std_norm = empirical["lHT_std"] / emp_mean  # Error scales with normalization
    lHP_std_norm = empirical["lHP_std"] / emp_mean
    
    lHT_theo_norm = theoretical["lHT_theoretical"] / theo_mean
    lHP_theo_norm = theoretical["lHP_theoretical"] / theo_mean

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: lHT scaling (delta=1), normalized
    ax1 = axes[0, 0]
    ax1.errorbar(
        chi,
        lHT_emp_norm,
        yerr=lHT_std_norm,
        fmt="o-",
        label="Empirical lHT",
        linewidth=2,
        markersize=10,
        color="blue",
        capsize=5,
        capthick=1.5,
        alpha=0.8,
    )
    ax1.plot(
        chi,
        lHT_theo_norm,
        "s--",
        label="Theoretical lHT (delta=1)",
        linewidth=2,
        markersize=8,
        color="red",
        alpha=0.7,
    )
    ax1.set_xlabel("χ", fontsize=13)
    ax1.set_ylabel("lHT (normalized)", fontsize=13)
    ax1.set_title("Training Eigenvalue (δ=1) vs χ [normalized]", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Plot 2: lHP scaling (delta=0), normalized
    ax2 = axes[0, 1]
    ax2.errorbar(
        chi,
        lHP_emp_norm,
        yerr=lHP_std_norm,
        fmt="o-",
        label="Empirical lHP",
        linewidth=2,
        markersize=10,
        color="green",
        capsize=5,
        capthick=1.5,
        alpha=0.8,
    )
    ax2.plot(
        chi,
        lHP_theo_norm,
        "s--",
        label="Theoretical lHP (delta=0)",
        linewidth=2,
        markersize=8,
        color="orange",
        alpha=0.7,
    )
    ax2.set_xlabel("χ", fontsize=13)
    ax2.set_ylabel("lHP (normalized)", fontsize=13)
    ax2.set_title("Population Eigenvalue (δ=0) vs χ [normalized]", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    # Plot 3: Relative error in lHT (normalized)
    ax3 = axes[1, 0]
    lHT_error_norm = np.abs(lHT_emp_norm - lHT_theo_norm) / np.abs(lHT_theo_norm)
    ax3.plot(chi, lHT_error_norm * 100, "o-", linewidth=2, markersize=10, color="purple")
    ax3.set_xlabel("χ", fontsize=13)
    ax3.set_ylabel("Relative Error (%)", fontsize=13)
    ax3.set_title("lHT Relative Error (normalized)", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    # Plot 4: Training loss vs chi
    ax4 = axes[1, 1]
    ax4.plot(chi, empirical["losses"], "o-", linewidth=2, markersize=10, color="brown")
    ax4.set_xlabel("χ", fontsize=13)
    ax4.set_ylabel("Final Training Loss", fontsize=13)
    ax4.set_title("Training Loss vs χ", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale("log")
    ax4.set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "chi_eigenvalue_scaling.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Second figure: Direct comparison (normalized)
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot lHT (normalized)
    ax5.scatter(
        lHT_theo_norm,
        lHT_emp_norm,
        s=100,
        alpha=0.7,
        c=chi,
        cmap="viridis",
    )
    lim_h = [
        min(lHT_emp_norm.min(), lHT_theo_norm.min()),
        max(lHT_emp_norm.max(), lHT_theo_norm.max()),
    ]
    ax5.plot(lim_h, lim_h, "r--", linewidth=2, label="Perfect Agreement")
    ax5.set_xlabel("Theoretical lHT (normalized)", fontsize=13)
    ax5.set_ylabel("Empirical lHT (normalized)", fontsize=13)
    ax5.set_title("lHT: Theory vs Empirical [normalized]", fontsize=14, fontweight="bold")
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    cbar_h = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar_h.set_label("χ", fontsize=12)

    # Scatter plot lHP (normalized)
    ax6.scatter(
        lHP_theo_norm,
        lHP_emp_norm,
        s=100,
        alpha=0.7,
        c=chi,
        cmap="plasma",
    )
    lim_k = [
        min(lHP_emp_norm.min(), lHP_theo_norm.min()),
        max(lHP_emp_norm.max(), lHP_theo_norm.max()),
    ]
    ax6.plot(lim_k, lim_k, "r--", linewidth=2, label="Perfect Agreement")
    ax6.set_xlabel("Theoretical lHP (normalized)", fontsize=13)
    ax6.set_ylabel("Empirical lHP (normalized)", fontsize=13)
    ax6.set_title("lHP: Theory vs Empirical [normalized]", fontsize=14, fontweight="bold")
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    cbar_k = plt.colorbar(ax6.collections[0], ax=ax6)
    cbar_k.set_label("χ", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "chi_eigenvalue_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(empirical, theoretical):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Chi Fixed Point Eigenvalue Analysis")
    print("=" * 70)
    print(f"{'χ':<10} {'lHT_emp':<12} {'lHT_theo':<12} {'lHP_emp':<12} {'lHP_theo':<12} {'Loss':<12}")
    print("-" * 70)

    for i, chi in enumerate(empirical["chi"]):
        print(
            f"{chi:<10.0f} "
            f"{empirical['lHT_empirical'][i]:<12.6f} "
            f"{theoretical['lHT_theoretical'][i]:<12.6f} "
            f"{empirical['lHP_empirical'][i]:<12.6f} "
            f"{theoretical['lHP_theoretical'][i]:<12.6f} "
            f"{empirical['losses'][i]:<12.6f}"
        )

    print("-" * 70)
    lHT_mse = np.mean((empirical["lHT_empirical"] - theoretical["lHT_theoretical"]) ** 2)
    lHP_mse = np.mean((empirical["lHP_empirical"] - theoretical["lHP_theoretical"]) ** 2)
    print("\nMean Squared Error:")
    print(f"  lHT: {lHT_mse:.6e}")
    print(f"  lHP: {lHP_mse:.6e}")
    print("=" * 70 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "data" / "results_fixed"
    plots_dir = base_dir / "plots_fixed"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "chi_fixed_eigenvalues.json"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run chi_fixed_points.py first to generate results.")
        return

    print("Loading empirical results...")
    empirical = load_empirical_results(results_file)

    print("Computing theoretical predictions using FCSLinear.jl...")
    d = 2
    P = 6
    N = 256
    kappa = 1.0

    theoretical = solve_theoretical_eigenvalues(empirical["chi"], d=d, P=P, N=N, kappa=kappa)

    print("Generating plots...")
    plot_eigenvalue_scaling(empirical, theoretical, plots_dir)

    print_summary(empirical, theoretical)

    print(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
