# EK-Comparison Milestone

Train 2,250 neural networks and compare empirical outputs with Eigenvalue-Kernel (EK) theoretical predictions.

## Experiment Setup

- **Network**: 3-layer FCN (d → 512 → 512 → 1) with linear activations
- **Dimensions**: d ∈ {2, 3, 4, ..., 10} (9 values)
- **Samples**: P = ⌊1.5d⌋ per dataset
- **Datasets**: 50 per dimension
- **Ensemble**: 5 networks per dataset
- **Training**: 10,000 epochs each, Langevin dynamics with L2 regularization
- **Total**: 9 × 50 × 5 = 2,250 networks

## Key Parameters

- **Chi (χ)**: 1.0 (standard scaling)
- **EK constant (κ)**: 0.5
- **NNGP eigenvalue (lH)**: 1/d
- **Learning rate**: 1e-5
- **Weight decay**: 1e-4
- **Temperature**: 1.0

## Training

Run `python ek_comparison.py` to:
1. Initialize networks with He initialization (1/fan-in variance)
2. Train with Langevin dynamics: θ ← θ - η∇L - ηλθ + √(2ηT)ξ
3. Save trained models to `data/networks/`
4. Compute loss and eigenvalues for each (d, dataset) pair

## Analysis

The script computes:
- **EK prediction factor**: (1/d) / (1/d + κ)
- **EK loss**: Bias + Variance from EK framework
- **Empirical loss**: Mean squared error from trained networks
- **Comparison**: Plot and analyze agreement between theory and empirics

## Output

- `data/results/config.json` — Experiment configuration
- `data/results/ek_comparison.json` — EK predictions and empirical losses
- `data/results/ensemble_results.json` — Per-network training details
- `plots/*.png` — Visualization of results

## Theory

The EK framework predicts network behavior based on kernel eigenvalues. For linear networks, the NNGP kernel has eigenvalue lH = 1/d. The framework predicts loss as:

$$\text{Loss} = \frac{\chi}{\kappa} \left( \text{Bias}^2 + \text{Variance} \right)$$

where Bias² = (κ/(lH + κ))² and Variance = (κ/(χP)) · (lH/(lH + κ)).

## Commands

```bash
# Train networks
python ek_comparison.py

# Generate plots
python plot_results.py

# Inspect results
python inspect_results.py
```

---
**Status**: Ready to train  
**Estimated time**: 24–48 hours on GPU
