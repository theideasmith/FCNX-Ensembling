#!/usr/bin/env python3
"""
Compare empirical H eigenvalues at initialization vs. theory for erf network.

Parameters: d=10, P=30, N=4000, chi=1.0, kappa=1.0, sigma2=1.0, activation=erf
"""

import sys
from pathlib import Path
import torch
import numpy as np
import tempfile

# Add lib to path
LIB_DIR = str(Path(__file__).parent.parent.parent / "lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from FCN3Network import FCN3NetworkActivationGeneric
from Experiment import Experiment

# Parameters
d = 10
P = 30  # = 3 * d
N = 5000
chi = 1.0
kappa = 1.0
sigma2 = 1.0
activation = "erf"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("ERF Network Eigenvalue Comparison at Initialization")
print("=" * 70)
print(f"Parameters: d={d}, P={P}, N={N}, chi={chi}, kappa={kappa}")
print(f"Activation: {activation}, Device: {device}")
print()

# Initialize network
weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
torch.manual_seed(26)  # Model seed
model = FCN3NetworkActivationGeneric(
    d=d, n1=N, n2=N, P=P, ens=50,
    activation=activation,
    weight_initialization_variance=weight_var,
    device=device,
).to(device)

model.eval()

# Compute empirical H eigenvalues at init, averaged over 50 datasets
print("Computing empirical H eigenvalues over 50 datasets...")
all_H_eigs = []

for dataset_seed in range(50):
    # Generate data with different seed for each dataset
    torch.manual_seed(42 + dataset_seed)
    X = torch.randn(1000, d, device=device, dtype=torch.float32)
    z = X[:, 0]
    Y_target = z.unsqueeze(-1)  # He1 target

    # Compute empirical H eigenvalues for this dataset
    with torch.no_grad():
        H_eig_empirical = model.H_eig(X, X[:,0].unsqueeze(-1), std=False)

    # Handle as list/tensor
    if isinstance(H_eig_empirical, (list, tuple)):
        H_eig_empirical = torch.tensor(H_eig_empirical) if not isinstance(H_eig_empirical[0], torch.Tensor) else torch.stack(H_eig_empirical)
    
    all_H_eigs.append(H_eig_empirical)

# Average over datasets
H_eig_empirical = torch.stack(all_H_eigs).mean(dim=0)

print(f"Data shape (from 50 datasets): X ({P}, {d}), Y ({P}, 1)")
print()

print("Empirical H eigenvalues (averaged over 50 datasets):")
print(f"  H_eig_empirical (via model.H_eig): {H_eig_empirical}")
print(f"  H_eig_empirical std: {torch.stack(all_H_eigs).std(dim=0)}")
print()

# Compute theory predictions using Experiment.eig_predictions()
print("Computing theoretical eigenvalues via Experiment.eig_predictions()...")
with tempfile.TemporaryDirectory() as tmpdir:
    exp = Experiment(
        file=tmpdir,
        N=N,
        d=d,
        chi=chi,
        P=P,
        ens=1,
        device=device,
        kappa=kappa,
        eps=0.0,  # No He3 mixing
    )
    
    predictions = exp.eig_predictions()

print("Theoretical H eigenvalues (via Experiment):")
if predictions is not None:
    print(f"  lH1T (target):       {predictions.lH1T:.10e}")
    print(f"  lH1P (perpendicular): {predictions.lH1P:.10e}")
    print(f"  (Note: lH3T={predictions.lH3T:.6e}, lH3P={predictions.lH3P:.6e})")
else:
    print("  Warning: predictions returned None")
    predictions = None

print()

# Compare
if predictions is not None:
    print("Comparison:")
    # H_eig_empirical is now a tensor; take the first element (target direction)
    H_eig_val = H_eig_empirical[0].item() if H_eig_empirical.dim() > 0 else H_eig_empirical.item()
    print(f"  Empirical H_eig[0]:  {H_eig_val:.10e}")
    print(f"  Theory lH1T:         {predictions.lH1T:.10e}")

    if not np.isnan(H_eig_val) and not np.isnan(predictions.lH1T):
        abs_error = abs(H_eig_val - predictions.lH1T)
        rel_error = abs_error / abs(predictions.lH1T) if predictions.lH1T != 0 else np.inf
        print(f"  Absolute error:      {abs_error:.10e}")
        print(f"  Relative error:      {rel_error:.6%}")
    else:
        print("  Cannot compute error: NaN in comparison")
else:
    print("  Skipping comparison: predictions unavailable")

print()
print("=" * 70)
