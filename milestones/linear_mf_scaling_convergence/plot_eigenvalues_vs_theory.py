"""
Plot empirical eigenvalues vs theoretical predictions with normalization.

This script:
1. Loads empirical eigenvalues from d_sweep_summary.json
2. Computes theoretical eigenvalues using FCSLinear.jl
3. Normalizes so (lHT + lHP)/2 = 1
4. Plots empirical vs theoretical comparison
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def compute_theoretical_eigenvalues(d, P, N, chi, kappa=1.0):
    """Compute theoretical lHT (delta=1) and lHP (delta=0) for given parameters."""
    ensure_julia_loaded()
    
    initial_guess = [1.0, 1.0]  # [lJ, lH]
    i0 = juliacall.convert(jl.Vector[jl.Float64], initial_guess)
    
    try:
        # Target regime (delta=1)
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
        
        # Perp regime (delta=0)
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
        print(f"Error computing theoretical eigenvalues for d={d}: {exc}")
        return None, None


def main():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "d_sweep_summary.json"
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    # Load empirical data
    with open(data_file, "r") as f:
        data = json.load(f)
    
    dims = np.array(data["dims"])
    largest_eigenvalues = np.array(data["largest_eigenvalues"])
    mean_other_eigenvalues = np.array(data["mean_other_eigenvalues"])
    P = data["P"]
    N = data["N"]
    chi = data["chi"]
    
    print(f"Parameters: P={P}, N={N}, chi={chi}")
    print(f"Dimensions: {dims}")
    
    # Compute theoretical predictions for each d
    lHT_theoretical = []
    lHP_theoretical = []
    
    for d in dims:
        print(f"\nComputing theory for d={d}...")
        lHT, lHP = compute_theoretical_eigenvalues(d, P, N, chi)
        
        if lHT is not None:
            lHT_theoretical.append(lHT)
            lHP_theoretical.append(lHP)
            print(f"  lHT={lHT:.6f}, lHP={lHP:.6f}")
        else:
            lHT_theoretical.append(np.nan)
            lHP_theoretical.append(np.nan)
    
    lHT_theoretical = np.array(lHT_theoretical)
    lHP_theoretical = np.array(lHP_theoretical)
    
    # Normalize: (lHT + lHP) / 2 = 1 for both theory and experiment
    # Treating largest_eigenvalues as lHT and mean_other as lHP
    emp_lHT = largest_eigenvalues
    emp_lHP = mean_other_eigenvalues
    
    emp_mean = (emp_lHT + (dims-1) * emp_lHP) 
    theo_mean = (lHT_theoretical + (dims-1) * lHP_theoretical) 
    
    emp_lHT_norm = emp_lHT / emp_mean
    emp_lHP_norm = emp_lHP / emp_mean
    theo_lHT_norm = lHT_theoretical / theo_mean
    theo_lHP_norm = lHP_theoretical / theo_mean
    
    # Compute power-law fits: eigenvalue ~ d^alpha
    # log(eigenvalue) = alpha * log(d) + log(constant)
    log_dims = np.log(dims)
    
    # Fit lHT empirical
    log_emp_lHT = np.log(emp_lHT_norm)
    fit_emp_lHT = np.polyfit(log_dims, log_emp_lHT, 1)
    alpha_emp_lHT = fit_emp_lHT[0]
    
    # Fit lHT theoretical
    log_theo_lHT = np.log(theo_lHT_norm)
    fit_theo_lHT = np.polyfit(log_dims, log_theo_lHT, 1)
    alpha_theo_lHT = fit_theo_lHT[0]
    
    # Fit lHP empirical
    log_emp_lHP = np.log(emp_lHP_norm)
    fit_emp_lHP = np.polyfit(log_dims, log_emp_lHP, 1)
    alpha_emp_lHP = fit_emp_lHP[0]
    
    # Fit lHP theoretical
    log_theo_lHP = np.log(theo_lHP_norm)
    fit_theo_lHP = np.polyfit(log_dims, log_theo_lHP, 1)
    alpha_theo_lHP = fit_theo_lHP[0]
    
    # Print scaling exponents
    print("\n" + "="*80)
    print("POWER-LAW SCALING EXPONENTS (eigenvalue ~ d^alpha)")
    print("="*80)
    print(f"lHT Empirical:    alpha = {alpha_emp_lHT:.4f}  (lHT ~ d^{alpha_emp_lHT:.4f})")
    print(f"lHT Theoretical:  alpha = {alpha_theo_lHT:.4f}  (lHT ~ d^{alpha_theo_lHT:.4f})")
    print(f"lHP Empirical:    alpha = {alpha_emp_lHP:.4f}  (lHP ~ d^{alpha_emp_lHP:.4f})")
    print(f"lHP Theoretical:  alpha = {alpha_theo_lHP:.4f}  (lHP ~ d^{alpha_theo_lHP:.4f})")
    print("="*80 + "\n")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: lHT vs d (log-log)
    ax1 = axes[0, 0]
    ax1.loglog(dims, emp_lHT_norm, "o", label=f"Empirical lHT (α={alpha_emp_lHT:.3f})", 
               markersize=10, color="blue")
    ax1.loglog(dims, theo_lHT_norm, "s", label=f"Theoretical lHT (α={alpha_theo_lHT:.3f})", 
               markersize=8, color="red", alpha=0.7)
    # Add best fit lines
    d_fit = np.logspace(np.log10(dims.min()), np.log10(dims.max()), 100)
    ax1.loglog(d_fit, np.exp(fit_emp_lHT[1]) * d_fit**alpha_emp_lHT, 
               '--', color='blue', alpha=0.5, linewidth=1.5, label='Emp fit')
    ax1.loglog(d_fit, np.exp(fit_theo_lHT[1]) * d_fit**alpha_theo_lHT, 
               '--', color='red', alpha=0.5, linewidth=1.5, label='Theo fit')
    ax1.set_xlabel("d (dimension)", fontsize=13)
    ax1.set_ylabel("lHT (normalized)", fontsize=13)
    ax1.set_title("Target Eigenvalue Scaling [log-log]", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: lHP vs d (log-log)
    ax2 = axes[0, 1]
    ax2.loglog(dims, emp_lHP_norm, "o", label=f"Empirical lHP (α={alpha_emp_lHP:.3f})", 
               markersize=10, color="green")
    ax2.loglog(dims, theo_lHP_norm, "s", label=f"Theoretical lHP (α={alpha_theo_lHP:.3f})", 
               markersize=8, color="orange", alpha=0.7)
    # Add best fit lines
    ax2.loglog(d_fit, np.exp(fit_emp_lHP[1]) * d_fit**alpha_emp_lHP, 
               '--', color='green', alpha=0.5, linewidth=1.5, label='Emp fit')
    ax2.loglog(d_fit, np.exp(fit_theo_lHP[1]) * d_fit**alpha_theo_lHP, 
               '--', color='orange', alpha=0.5, linewidth=1.5, label='Theo fit')
    ax2.set_xlabel("d (dimension)", fontsize=13)
    ax2.set_ylabel("lHP (normalized)", fontsize=13)
    ax2.set_title("Perp Eigenvalue Scaling [log-log]", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Scatter lHT
    ax3 = axes[1, 0]
    ax3.scatter(theo_lHT_norm, emp_lHT_norm, s=100, alpha=0.7, c=dims, cmap="viridis")
    lim = [min(theo_lHT_norm.min(), emp_lHT_norm.min()),
           max(theo_lHT_norm.max(), emp_lHT_norm.max())]
    ax3.plot(lim, lim, "r--", linewidth=2, label="Perfect Agreement")
    ax3.set_xlabel("Theoretical lHT (normalized)", fontsize=13)
    ax3.set_ylabel("Empirical lHT (normalized)", fontsize=13)
    ax3.set_title("lHT: Theory vs Empirical", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label("d", fontsize=12)
    
    # Plot 4: Scatter lHP
    ax4 = axes[1, 1]
    ax4.scatter(theo_lHP_norm, emp_lHP_norm, s=100, alpha=0.7, c=dims, cmap="plasma")
    lim = [min(theo_lHP_norm.min(), emp_lHP_norm.min()),
           max(theo_lHP_norm.max(), emp_lHP_norm.max())]
    ax4.plot(lim, lim, "r--", linewidth=2, label="Perfect Agreement")
    ax4.set_xlabel("Theoretical lHP (normalized)", fontsize=13)
    ax4.set_ylabel("Empirical lHP (normalized)", fontsize=13)
    ax4.set_title("lHP: Theory vs Empirical", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label("d", fontsize=12)
    
    plt.tight_layout()
    output_path = base_dir / "eigenvalues_vs_theory_normalized.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 100)
    print("Eigenvalue Comparison Summary (Normalized)")
    print("=" * 100)
    print(f"{'d':<6} {'Emp lHT':<12} {'Theo lHT':<12} {'Emp lHP':<12} {'Theo lHP':<12} {'Error lHT (%)':<15} {'Error lHP (%)':<15}")
    print("-" * 100)
    
    for i, d in enumerate(dims):
        emp_ht = emp_lHT_norm[i]
        theo_ht = theo_lHT_norm[i]
        emp_hp = emp_lHP_norm[i]
        theo_hp = theo_lHP_norm[i]
        err_ht = abs(emp_ht - theo_ht) / abs(theo_ht) * 100 if not np.isnan(theo_ht) else np.nan
        err_hp = abs(emp_hp - theo_hp) / abs(theo_hp) * 100 if not np.isnan(theo_hp) else np.nan
        
        print(f"{d:<6} {emp_ht:<12.6f} {theo_ht:<12.6f} {emp_hp:<12.6f} {theo_hp:<12.6f} "
              f"{err_ht:<15.2f} {err_hp:<15.2f}")
    
    print("=" * 100)
    
    # Check normalization
    print("\nNormalization check (should be ~1.0):")
    for i, d in enumerate(dims):
        emp_avg = (emp_lHT_norm[i] + emp_lHP_norm[i]) / 2
        theo_avg = (theo_lHT_norm[i] + theo_lHP_norm[i]) / 2
        print(f"  d={d}: Empirical avg = {emp_avg:.6f}, Theoretical avg = {theo_avg:.6f}")


if __name__ == "__main__":
    main()
