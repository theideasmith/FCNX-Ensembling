#!/usr/bin/env python3
"""
Compute the training loss of a saved linear FCN3 model.

This script loads a pre-trained model and computes the MSE loss on the training data.
"""

import os
import sys
import torch
import numpy as np

# Add lib to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from FCN3Network import FCN3NetworkEnsembleLinear
from Experiment import ExperimentLinear
import tempfile

# Configuration matching train_linear_fcn3_and_compare.py
D = 4
N = 20
KAPPA = 1.0
CHI = N
P_TRAIN = 8
ENSEMBLES = 100
DATASETS = 1
NDUP = ENSEMBLES * DATASETS
DATA_SEED = 613
MODEL_SEED = 26

# Model path
model_path = '/home/akiva/FCNX-Ensembling/plots/fcn3_linear_eig_comparison/model_linear.pth'

# Device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Generate training data (same as in train_linear_fcn3_and_compare.py)
print('\nGenerating training data...')
torch.manual_seed(DATA_SEED)
X_train = torch.randn((P_TRAIN, D), dtype=torch.float32, device=device)
z = X_train[:, 0]
He1 = z
Y_train = He1
print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')

# Load model
print(f'\nLoading model from {model_path}...')
if not os.path.exists(model_path):
    print(f'ERROR: Model file not found at {model_path}')
    sys.exit(1)

model = FCN3NetworkEnsembleLinear(
    D, N, N, P_TRAIN,
    ensembles=NDUP,
    weight_initialization_variance=(1/D, 1.0/N, 1.0/(N*CHI))
)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded successfully')

# Compute loss
print('\nComputing loss...')
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    # Custom MSE loss (matching train_linear_fcn3_and_compare.py)
    diff = outputs - Y_train.unsqueeze(-1)
    loss = torch.sum(diff * diff) / ENSEMBLES
    loss_value = loss.item()

    discrepancy = torch.mean(diff.abs(), dim=0)

print(f'\n' + '='*60)
print('Loss Metrics:')
print('='*60)
print(f'Total loss (sum over all):     {loss_value:.8e}')
print(f'Normalized (loss / (P*NDUP)):   {loss_value / (P_TRAIN * NDUP):.8e}')
print(f'Mean discrepancy (over ensembles): {discrepancy.mean(dim = 0).item():.8e}')

# Compute expected discrepancy using Experiment class
print('\nComputing theoretical expected discrepancy...')
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = ExperimentLinear(
            file=tmpdir,
            N=N,
            d=D,
            chi=CHI,
            P=P_TRAIN,
            ens=1,
            device=device,
            kappa=KAPPA
        )
        
        # Get theoretical eigenvalues
        preds = exp.eig_predictions()
        
        if preds is not None and hasattr(preds, 'lHT'):
            lHT = float(preds.lHT) if preds.lHT is not None else 0.0
            lJT = float(preds.lJT) if preds.lJT is not None else 0.0
            lHP = float(preds.lHP) if preds.lHP is not None else 0.0
            lJP = float(preds.lJP) if preds.lJP is not None else 0.0
             # Compute intermediate values from the Mathematica equations
            # lHh == (1/lJ - (1/lJ)^2 lH)
            lHhT = (1/lJT - (1/lJT)**2 * lHT)
            lHhP = (1/lJP - (1/lJP)**2 * lHP)
            # lK == (k/(P χ)) lH/(lH + k/P) - (lH/(lH + k/P))^2 * Δ
            k_over_P = KAPPA 
            denominatorT = lHT + k_over_P
            lKT = (1.0 / denominatorT)**2 
            denominatorP = lHP + k_over_P
            lKP = (1.0 / (lHP + k_over_P))**2
            # lKh == χ^2 lK/lH^2
            ftilde = CHI**2 * 1.0 / denominatorT  + (D - 1) * CHI**2 * 1.0 / denominatorP
            # Expected discrepancy = chi * P * lHT
            expected_discrepancy = ftilde / CHI**2 /P_TRAIN
            
            print(f'  Expected discrepancy (tau): {expected_discrepancy:.8e}')
            print(f'  Ratio (empirical / theoretical): {discrepancy.mean().item() / expected_discrepancy:.4f}')
        else:
            print('  Warning: Could not compute lHT from Experiment')
except Exception as e:
    print(f'  Warning: Failed to compute theoretical predictions: {e}')

print('='*60)

# Additional statistics
print(f'\nOutput statistics:')
print(f'  Shape: {outputs.shape}')
print(f'  Mean: {outputs.mean():.6f}')
print(f'  Std:  {outputs.std():.6f}')
print(f'  Min:  {outputs.min():.6f}')
print(f'  Max:  {outputs.max():.6f}')

print(f'\nTarget statistics:')
print(f'  Shape: {Y_train.shape}')
print(f'  Mean: {Y_train.mean():.6f}')
print(f'  Std:  {Y_train.std():.6f}')
print(f'  Min:  {Y_train.min():.6f}')
print(f'  Max:  {Y_train.max():.6f}')
