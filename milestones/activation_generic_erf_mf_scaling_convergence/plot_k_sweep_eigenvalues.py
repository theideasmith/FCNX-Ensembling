#!/usr/bin/env python3
"""
Plot learnabilities from eigs.json with ArXiV quality formatting.

Creates comparison plots of theoretical and empirical Hermite learnabilities
as a function of kappa_eff.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Set ArXiV quality styling
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'text.usetex': False,
})

RESULTS_DIR = Path(__file__).parent / "results"
EIGS_FILE = RESULTS_DIR / "eigs.json"


def load_eigs(file_path):
    """Load learnability data from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def group_by_params(data):
    """Group data by (d, P, N, chi)."""
    groups = defaultdict(list)
    for entry in data:
        key = (entry['d'], entry['P'], entry['N'], entry['chi'])
        groups[key].append(entry)
    return groups


def plot_eigenvalues(group_key, group_data, output_dir=RESULTS_DIR):
    """Create comparison plots (log-log, linear-linear, and semilogx) for a group of learnabilities."""
    d, P, N, chi = group_key
    
    # Sort by kappa_eff
    sorted_data = sorted(group_data, key=lambda x: x['kappa_eff'])
    
    kappa_effs = np.array([x['kappa_eff'] for x in sorted_data])
    lH1T_theory = np.array([x['lH1T_theory'] for x in sorted_data])
    eig_H_0 = np.array([x['eig_H_0'] for x in sorted_data])
    lH3T_theory = np.array([x['lH3T_theory'] for x in sorted_data])
    eig_H_d = np.array([x['eig_H_d'] for x in sorted_data])
    
    # Calculate percentage gaps
    gap_H1_pct = 100 * np.abs(eig_H_0 - lH1T_theory) / np.abs(lH1T_theory)
    gap_H3_pct = 100 * np.abs(eig_H_d - lH3T_theory) / np.abs(lH3T_theory)
    
    # ===== LOG-LOG PLOT =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Hermite-1 (He1) Plot =====
    ax1.loglog(kappa_effs, lH1T_theory, 'o-', color='#1f77b4', linewidth=2, 
               markersize=8, label=r'Theory ($\lambda_{H_1}^T$)', zorder=3)
    ax1.loglog(kappa_effs, eig_H_0, 's--', color='#ff7f0e', linewidth=2, 
               markersize=8, label=r'Empirical ($\lambda_{H_1}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH1T_theory, eig_H_0, gap_H1_pct)):
        mid_y = np.sqrt(theory * emp)  # Geometric mean for log scale
        ax1.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax1.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax1.set_ylabel(r'Learnability $\eta_{H_1}$', fontsize=14)
    ax1.set_title(r'Hermite-1 Learnability (Log-Log)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # ===== Hermite-d (He3) Plot =====
    ax2.loglog(kappa_effs, lH3T_theory, 'o-', color='#2ca02c', linewidth=2, 
               markersize=8, label=r'Theory ($\lambda_{H_d}^T$)', zorder=3)
    ax2.loglog(kappa_effs, eig_H_d, 's--', color='#d62728', linewidth=2, 
               markersize=8, label=r'Empirical ($\lambda_{H_d}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH3T_theory, eig_H_d, gap_H3_pct)):
        mid_y = np.sqrt(theory * emp)  # Geometric mean for log scale
        ax2.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax2.set_ylabel(r'Learnability $\eta_{H_d}$', fontsize=14)
    ax2.set_title(f'Hermite-d Learnability (Log-Log)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Overall title with network parameters
    fig.suptitle(
        f'Network: $d={d}, P={P}, N={N}, \\chi={chi}$',
        fontsize=16, fontweight='bold', y=1.00
    )
    
    plt.tight_layout()
    
    # Save log-log
    output_path = output_dir / f"learnabilities_loglog_d{d}_P{P}_N{N}_chi{chi}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    
    output_path_png = output_dir / f"learnabilities_loglog_d{d}_P{P}_N{N}_chi{chi}.png"
    plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path_png}")
    
    plt.close()
    
    # ===== LINEAR-LINEAR PLOT =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Hermite-1 (He1) Plot =====
    ax1.plot(kappa_effs, lH1T_theory, 'o-', color='#1f77b4', linewidth=2, 
             markersize=8, label=r'Theory ($\lambda_{H_1}^T$)', zorder=3)
    ax1.plot(kappa_effs, eig_H_0, 's--', color='#ff7f0e', linewidth=2, 
             markersize=8, label=r'Empirical ($\lambda_{H_1}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH1T_theory, eig_H_0, gap_H1_pct)):
        mid_y = (theory + emp) / 2  # Arithmetic mean for linear scale
        ax1.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax1.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax1.set_ylabel(r'Learnability $\eta_{H_1}$', fontsize=14)
    ax1.set_title(r'Hermite-1 Learnability (Linear)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=0)
    
    # ===== Hermite-d (He3) Plot =====
    ax2.plot(kappa_effs, lH3T_theory, 'o-', color='#2ca02c', linewidth=2, 
             markersize=8, label=r'Theory ($\lambda_{H_d}^T$)', zorder=3)
    ax2.plot(kappa_effs, eig_H_d, 's--', color='#d62728', linewidth=2, 
             markersize=8, label=r'Empirical ($\lambda_{H_d}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH3T_theory, eig_H_d, gap_H3_pct)):
        mid_y = (theory + emp) / 2  # Arithmetic mean for linear scale
        ax2.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax2.set_ylabel(r'Learnability $\eta_{H_d}$', fontsize=14)
    ax2.set_title(f'Hermite-d Learnability (Linear)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(bottom=0)
    
    # Overall title with network parameters
    fig.suptitle(
        f'Network: $d={d}, P={P}, N={N}, \\chi={chi}$',
        fontsize=16, fontweight='bold', y=1.00
    )
    
    plt.tight_layout()
    
    # Save linear-linear
    output_path = output_dir / f"learnabilities_linear_d{d}_P{P}_N{N}_chi{chi}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    
    output_path_png = output_dir / f"learnabilities_linear_d{d}_P{P}_N{N}_chi{chi}.png"
    plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path_png}")
    
    plt.close()
    
    # ===== SEMILOGX PLOT (Log X, Linear Y) =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Hermite-1 (He1) Plot =====
    ax1.semilogx(kappa_effs, lH1T_theory, 'o-', color='#1f77b4', linewidth=2, 
                 markersize=8, label=r'Theory ($\lambda_{H_1}^T$)', zorder=3)
    ax1.semilogx(kappa_effs, eig_H_0, 's--', color='#ff7f0e', linewidth=2, 
                 markersize=8, label=r'Empirical ($\lambda_{H_1}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH1T_theory, eig_H_0, gap_H1_pct)):
        mid_y = (theory + emp) / 2  # Arithmetic mean for linear y scale
        ax1.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax1.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax1.set_ylabel(r'Learnability $\eta_{H_1}$', fontsize=14)
    ax1.set_title(r'Hermite-1 Learnability (SemilogX)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='major', linestyle='-', linewidth=0.5)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=0)
    
    # ===== Hermite-d (He3) Plot =====
    ax2.semilogx(kappa_effs, lH3T_theory, 'o-', color='#2ca02c', linewidth=2, 
                 markersize=8, label=r'Theory ($\lambda_{H_d}^T$)', zorder=3)
    ax2.semilogx(kappa_effs, eig_H_d, 's--', color='#d62728', linewidth=2, 
                 markersize=8, label=r'Empirical ($\lambda_{H_d}$)', zorder=3)
    
    # Add percentage gap labels
    for i, (kappa, theory, emp, gap) in enumerate(zip(kappa_effs, lH3T_theory, eig_H_d, gap_H3_pct)):
        mid_y = (theory + emp) / 2  # Arithmetic mean for linear y scale
        ax2.text(kappa, mid_y, f'{gap:.1f}%', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.set_xlabel(r'$\kappa_{\text{eff}}$', fontsize=14)
    ax2.set_ylabel(r'Learnability $\eta_{H_d}$', fontsize=14)
    ax2.set_title(f'Hermite-d Learnability (SemilogX)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='major', linestyle='-', linewidth=0.5)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(bottom=0)
    
    # Overall title with network parameters
    fig.suptitle(
        f'Network: $d={d}, P={P}, N={N}, \\chi={chi}$',
        fontsize=16, fontweight='bold', y=1.00
    )
    
    plt.tight_layout()
    
    # Save semilogx
    output_path = output_dir / f"learnabilities_semilogx_d{d}_P{P}_N{N}_chi{chi}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    
    output_path_png = output_dir / f"learnabilities_semilogx_d{d}_P{P}_N{N}_chi{chi}.png"
    plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path_png}")
    
    plt.close()


def main():
    if not EIGS_FILE.exists():
        print(f"Error: {EIGS_FILE} not found")
        return 1
    
    print(f"Loading data from {EIGS_FILE}...")
    data = load_eigs(EIGS_FILE)
    
    print(f"Loaded {len(data)} entries")
    
    # Group by parameters
    groups = group_by_params(data)
    print(f"Found {len(groups)} configuration(s)")
    
    # Create plots for each group
    for group_key in sorted(groups.keys()):
        group_data = groups[group_key]
        print(f"\nPlotting: d={group_key[0]}, P={group_key[1]}, N={group_key[2]}, chi={group_key[3]} ({len(group_data)} points)")
        plot_eigenvalues(group_key, group_data)
    
    print(f"\nAll plots saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
