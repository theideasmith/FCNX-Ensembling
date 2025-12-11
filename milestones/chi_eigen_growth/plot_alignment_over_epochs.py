"""
Plot alignment (slope of theory-experiment) over continued training epochs.

This script:
1. Loads original theoretical predictions (lHT, lHP)
2. Loads checkpoint eigenvalues from continue_training_chi_fixed.py
3. Computes alignment as the slope of a regression between theory and empirical
4. Plots alignment vs epochs for each chi value
"""

import json
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import juliacall
from juliacall import Main as jl

# Ensure PyJulia uses the current Python
os.environ.setdefault("PYTHONCALL_JULIA_PYTHON", "yes")

JULIA_FCS_LINEAR_PATH = "/home/akiva/FCNX-Ensembling/julia_lib/FCSLinear.jl"


def ensure_julia_loaded():
    """Include the Julia FCSLinear module once per process."""
    if getattr(ensure_julia_loaded, "_loaded", False):
        return
    jl.include(JULIA_FCS_LINEAR_PATH)
    ensure_julia_loaded._loaded = True


def load_theoretical_eigenvalues(chi, d=2, P=6, N=256, kappa=1.0):
    """Compute theoretical lHT and lHP for a single chi value."""
    ensure_julia_loaded()
    
    initial_guess = [1.0, 1.0]  # [lJ, lH]
    i0 = juliacall.convert(jl.Vector[jl.Float64], initial_guess)
    
    try:
        # Training regime (delta=1)
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
        
        # Population regime (delta=0)
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
            return None, None
        
        lHT = float(sol_T[1])
        lHP = float(sol_P[1])
        
        return lHT, lHP
    except Exception as exc:
        print(f"Error computing theoretical eigenvalues for chi={chi}: {exc}")
        return None, None


def compute_alignment(theory_vals, empirical_vals):
    """
    Compute alignment as the slope of lsq fit: empirical = alignment * theory.
    Returns slope (alignment), R² value, and residual.
    """
    # Flatten eigenvalue lists
    theory_flat = np.array(theory_vals).flatten()
    empirical_flat = np.array(empirical_vals).flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(theory_flat) | np.isnan(empirical_flat))
    theory_clean = theory_flat[mask]
    empirical_clean = empirical_flat[mask]
    
    if len(theory_clean) == 0:
        return np.nan, np.nan, np.nan
    
    # Least squares fit: empirical = slope * theory
    slope = np.sum(theory_clean * empirical_clean) / np.sum(theory_clean ** 2)
    
    # R² value
    residuals = empirical_clean - slope * theory_clean
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((empirical_clean - np.mean(empirical_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return slope, r_squared, ss_res


def main():
    base_dir = Path(__file__).resolve().parent
    checkpoint_dir = base_dir / "data" / "checkpoints_fixed"
    plots_dir = base_dir / "plots_fixed"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    chi_values = [1, 64, 128, 192, 256]
    
    # Store alignment data for each chi
    alignment_data = {}
    
    for chi in chi_values:
        checkpoint_path = checkpoint_dir / f"chi_{chi}_continued.json"
        
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            continue
        
        print(f"\nProcessing chi={chi}...")
        
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        
        original_epochs = checkpoint_data["original_epochs"]
        checkpoints = checkpoint_data["checkpoints"]
        
        # Load theoretical predictions
        lHT, lHP = load_theoretical_eigenvalues(chi)
        
        if lHT is None:
            print(f"  Failed to compute theoretical values for chi={chi}")
            continue
        
        # Compute alignment at each checkpoint
        epochs_list = []
        alignment_lHT = []
        alignment_lHP = []
        r2_lHT = []
        r2_lHP = []
        
        for cp in checkpoints:
            epoch = cp["epoch"]
            eigenvalues = cp["eigenvalues"]
            
            # Get max eigenvalue (lH)
            lH_empirical = max(eigenvalues)
            
            # For lHT alignment: use the theoretical lHT
            slope_T, r2_T, _ = compute_alignment([lHT], [lH_empirical])
            
            # For lHP alignment: use the theoretical lHP
            slope_P, r2_P, _ = compute_alignment([lHP], [lH_empirical])
            
            epochs_list.append(epoch)
            alignment_lHT.append(slope_T)
            alignment_lHP.append(slope_P)
            r2_lHT.append(r2_T)
            r2_lHP.append(r2_P)
            
            print(f"  Epoch {epoch:>8}: alignment_T={slope_T:.4f} (R²={r2_T:.4f}), " +
                  f"alignment_P={slope_P:.4f} (R²={r2_P:.4f})")
        
        alignment_data[chi] = {
            "epochs": epochs_list,
            "original_epochs": original_epochs,
            "alignment_lHT": alignment_lHT,
            "alignment_lHP": alignment_lHP,
            "r2_lHT": r2_lHT,
            "r2_lHP": r2_lHP,
        }
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(chi_values)))
    
    for idx, chi in enumerate(chi_values):
        if chi not in alignment_data:
            continue
        
        data = alignment_data[chi]
        epochs = data["epochs"]
        original_eps = data["original_epochs"]
        
        # Plot 1: Alignment over epochs
        ax1 = axes[0]
        ax1.plot(
            epochs,
            data["alignment_lHT"],
            "o-",
            label=f"χ={chi} (lHT)",
            color=colors[idx],
            linewidth=2,
            markersize=6,
        )
        ax1.axvline(original_eps, color=colors[idx], linestyle="--", alpha=0.5)
        
        # Plot 2: R² over epochs
        ax2 = axes[1]
        ax2.plot(
            epochs,
            data["r2_lHT"],
            "o-",
            label=f"χ={chi} (lHT)",
            color=colors[idx],
            linewidth=2,
            markersize=6,
        )
        ax2.axvline(original_eps, color=colors[idx], linestyle="--", alpha=0.5)
    
    # Formatting
    ax1.set_xlabel("Epochs", fontsize=13)
    ax1.set_ylabel("Alignment (Slope)", fontsize=13)
    ax1.set_title("Theory-Empirical Alignment vs Continued Training Epochs", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    
    ax2.set_xlabel("Epochs", fontsize=13)
    ax2.set_ylabel("R² (Goodness of Fit)", fontsize=13)
    ax2.set_title("R² vs Continued Training Epochs", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    
    plt.tight_layout()
    output_path = plots_dir / "alignment_over_epochs.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Alignment Summary at Final Checkpoint")
    print("=" * 80)
    print(f"{'χ':<8} {'Original Epochs':<18} {'Final Alignment (T)':<20} {'Final R² (T)':<15}")
    print("-" * 80)
    
    for chi in chi_values:
        if chi in alignment_data:
            data = alignment_data[chi]
            final_alignment = data["alignment_lHT"][-1] if data["alignment_lHT"] else np.nan
            final_r2 = data["r2_lHT"][-1] if data["r2_lHT"] else np.nan
            print(f"{chi:<8} {data['original_epochs']:<18} {final_alignment:<20.6f} {final_r2:<15.6f}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
