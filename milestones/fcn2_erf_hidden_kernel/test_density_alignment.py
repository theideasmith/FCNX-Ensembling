#!/usr/bin/env python3
"""Test to diagnose the density/action alignment issue."""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data from a Gaussian
np.random.seed(42)
true_sigma = 0.5
true_var = true_sigma ** 2
data = np.random.normal(0, true_sigma, 100000)

# Empirical computation
bins = 50
hist, bin_edges = np.histogram(data, bins=bins, density=True)
bin_widths = np.diff(bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Method 1: density = hist * bin_width (what we're currently doing)
density_method1 = hist * bin_widths
mask1 = density_method1 > 0
action_method1 = -np.log(density_method1[mask1])

# Method 2: just use hist directly (probability density from numpy)
mask2 = hist > 0
action_method2 = -np.log(hist[mask2])

# Theoretical Gaussian action
x_range = np.linspace(-3*true_sigma, 3*true_sigma, 200)
# Action for Gaussian: -log(p(x)) where p(x) = 1/(sigma*sqrt(2*pi)) * exp(-x^2/(2*sigma^2))
gaussian_action = 0.5*np.log(2*np.pi*true_var) + x_range**2 / (2*true_var)

# Also compute what the Gaussian density is at bin centers for comparison
gaussian_density_at_centers = (1.0 / (true_sigma * np.sqrt(2*np.pi))) * np.exp(-bin_centers**2 / (2*true_var))
gaussian_action_at_centers = -np.log(gaussian_density_at_centers)

print("=== Diagnostic Output ===")
print(f"True sigma: {true_sigma:.6f}")
print(f"True var: {true_var:.6f}")
print(f"Empirical var from data: {np.var(data):.6f}")
print()

# At origin (x=0)
print("At x=0:")
print(f"  Gaussian density p(0): {1.0 / (true_sigma * np.sqrt(2*np.pi)):.6f}")
print(f"  Gaussian action -log(p(0)): {0.5*np.log(2*np.pi*true_var):.6f}")
print()

print("Empirical histogram at bins near x=0:")
center_idx = np.argmin(np.abs(bin_centers))
print(f"  hist[{center_idx}] (density from numpy): {hist[center_idx]:.6f}")
print(f"  bin_width: {bin_widths[center_idx]:.6f}")
print(f"  hist * bin_width: {density_method1[center_idx]:.6f}")
print(f"  -log(hist * bin_width): {action_method1[center_idx]:.6f}")
print(f"  -log(hist): {action_method2[center_idx]:.6f}")
print()

# Check: sum of hist*bin_width should be ~1
print(f"Sum of hist * bin_width (should be ~1): {np.sum(hist * bin_widths):.6f}")
print(f"Sum of hist (should be ~bins/50): {np.sum(hist):.6f}")  # Actually sum(hist * bin_width) = 1 when density=True
print()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Both empirical methods vs gaussian at bin centers
ax = axes[0]
ax.plot(bin_centers[mask1], action_method1, 'o-', label=f'Empirical: -log(hist*bin_width)', alpha=0.7)
ax.plot(bin_centers[mask2], action_method2, 's-', label=f'Empirical: -log(hist)', alpha=0.7)
ax.plot(bin_centers, gaussian_action_at_centers, 'x-', linewidth=2, label=f'Gaussian at bin centers', alpha=0.7)
ax.set_xlabel('Weight value w')
ax.set_ylabel('-log(density)')
ax.set_title('Empirical vs Theoretical Action (at bin centers)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Full range Gaussian vs empirical histogram
ax = axes[1]
ax.plot(x_range, gaussian_action, 'k-', linewidth=2, label='Gaussian action (full range)')
ax.plot(bin_centers[mask1], action_method1, 'o', alpha=0.7, label='-log(hist*bin_width)')
ax.set_xlabel('Weight value w')
ax.set_ylabel('-log(density)')
ax.set_title('Full Range Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/density_alignment_test.png', dpi=150)
print("\nSaved plot to density_alignment_test.png")
plt.close()

# Additional check: what if bin_width matters?
print("\n=== Analysis ===")
print("If hist*bin_width gives probability (summing to 1),")
print("then -log(hist*bin_width) is the action of that probability mass.")
print()
print("But for a continuous density, we should use:")
print("-log(probability_density) = -log(hist) when hist is already the density.")
print()
print("The issue: np.histogram with density=True gives PDF values,")
print("NOT probabilities. To get probability, multiply by bin_width.")
print()
print("So the correct formula should be:")
print("-log(density) = -log(hist) [since hist is already the density from density=True]")
