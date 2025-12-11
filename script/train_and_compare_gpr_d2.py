#!/usr/bin/env python3
"""
Train an FCN3Erf model on d=2 with specified hyperparameters (N=200, kappa=1.0, chi=1.0, P=30)
for 2 million epochs and compare the neural network predictions with exact GPR (Gaussian Process Regression).

This script:
1. Trains the FCN3Erf model using Langevin dynamics (matching the training loop in train_and_analyze_d_scale_10_15.py).
2. Generates a test set.
3. Computes exact GPR predictions using the dot-product kernel (as in GPKit.py).
4. Compares NN outputs with GPR predictions via:
   - Mean squared error
   - Correlation coefficient
   - Visualization plots (predictions vs targets, NN vs GPR, residuals)
5. Saves results to a results folder.

Usage:
    python3 script/train_and_compare_gpr_d2.py --epochs 2000000 [--cuda] [--dry-run]

    # Dry run (quick test with fewer epochs):
    python3 script/train_and_compare_gpr_d2.py --epochs 100 --dry-run

    # Load pre-trained model and skip training:
    python3 script/train_and_compare_gpr_d2.py --analyze-only [--cuda]
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

# Project layout helpers
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from FCN3Network import FCN3NetworkEnsembleErf
from GPKit import gpr_dot_product_explicit


# ============================================================================
# Hyperparameters
# ============================================================================
D = 4
N = 200
KAPPA = 1.0
CHI = 0.1
P_TRAIN = 8
EPS = 0.03  # Mixing parameter: Y = He1 + eps*He3

# Test set size
P_TEST = 40

# Data and model seeds (for reproducibility)
DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480

# Optimization
DEFAULT_EPOCHS = 2_000_000
ALPHA_T = 0.7  # Fraction of epochs before lr schedule change
LR_SCHEDULE_MULTIPLIER = 3.0  # lr gets divided by this after alphaT fraction


# ============================================================================
# Helper functions
# ============================================================================

def make_folder(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def generate_training_data(P, d, device, seed=DATA_SEED, eps=EPS):
    """Generate synthetic training data: Y = He1(X) + eps * He3(X) where X ~ N(0, I)."""
    torch.manual_seed(seed)
    X = torch.randn((P, d), dtype=torch.float32, device=device)
    z = X[:, 0]
    He1 = z
    He3 = z ** 3 - 3.0 * z
    Y = (He1 + eps * He3).unsqueeze(-1)
    return X, Y, He1, He3


def generate_test_data(P, d, device, seed=DATA_SEED + 1, eps=EPS):
    """Generate synthetic test data."""
    torch.manual_seed(seed)
    X_test = torch.randn((P, d), dtype=torch.float32, device=device)
    z_test = X_test[:, 0]
    He1_test = z_test
    He3_test = z_test ** 3 - 3.0 * z_test
    Y_test = (He1_test + eps * He3_test).unsqueeze(-1)
    return X_test, Y_test, He1_test, He3_test


def train_fcn3erf_langevin(model, X_train, Y_train, epochs, lr, device, model_path=None, dry_run=False):
    """
    Train FCN3Erf using Langevin dynamics.

    Args:
        model: FCN3NetworkEnsembleErf model
        X_train: training input (P, d)
        Y_train: training target (P, 1)
        epochs: number of training epochs
        lr: base learning rate
        device: torch device
        model_path: path to save model checkpoints (if provided)
        dry_run: if True, only do 1 epoch

    Returns:
        loss_history: list of loss values per epoch (sampled)
    """
    model.train()
    
    # Langevin dynamics parameters
    t = 2 * KAPPA
    
    # Weight-decay mapping vector
    weight_decay_vec = t * torch.tensor([float(D), float(N), float(N) * CHI], 
                                                     dtype=torch.float32, device=device)
    
    # Noise buffer and RNG
    noise_buffer = torch.empty(1, device=device, dtype=torch.float32)
    langevin_gen = torch.Generator(device=device)
    
    # LR schedule
    T = max(1, int(epochs * ALPHA_T))
    
    # Loss function
    def custom_mse_loss(outputs, targets):
        diff = outputs - targets
        return torch.sum(diff * diff)
    
    loss_history = []
    print(f'Starting Langevin training: {epochs} epochs, lr={lr:.3e}, T={T}')
    t0 = time.time()
    
    pbar = tqdm(range(1, epochs + 1), desc='Training', unit='epoch')
    for epoch in pbar:
        if dry_run and epoch > 1:
            break
            
        model.zero_grad()
        
        # LR schedule
        effective_lr = lr
        if epoch > T:
            effective_lr = lr / LR_SCHEDULE_MULTIPLIER
        
        # Langevin noise scale
        noise_scale = math.sqrt(2.0 * effective_lr * t)
        langevin_gen.manual_seed(LANGEVIN_SEED + epoch)
        
        # Forward pass
        outputs = model(X_train)
        if outputs.ndim > 1 and outputs.shape[-1] == 1:
            outputs = outputs.view(-1, 1)
        
        loss = custom_mse_loss(outputs, Y_train)
        
        # Backward
        loss.backward()
        
        # Parameter update with Langevin noise and weight decay
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.grad is None:
                    continue
                
                # Langevin noise
                noise_buffer.resize_(param.shape).normal_(0.0, 1.0, generator=langevin_gen).mul_(noise_scale)
                param.add_(noise_buffer)
                
                # Weight decay
                wdec = float(weight_decay_vec[i % len(weight_decay_vec)])
                param.add_(param.data, alpha=-wdec * effective_lr)
                
                # Gradient descent
                param.add_(param.grad, alpha=-effective_lr)
        
        # Update progress bar with loss
        pbar.set_postfix({'loss': f'{float(loss):.6e}'})
        
        if epoch % 10000 == 0 or epoch == epochs:
            loss_history.append((epoch, float(loss)))
            # Save model checkpoint every 10k epochs
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
    
    pbar.close()
    return loss_history
def compute_gpr_predictions(X_train, Y_train, X_test, noise_variance=1.0):
    """
    Compute exact GPR predictions using dot-product kernel.

    Args:
        X_train: (P_train, d)
        Y_train: (P_train, 1)
        X_test: (P_test, d)
        noise_variance: observation noise variance

    Returns:
        Y_pred_gpr: (P_test, 1) predicted means
    """
    Y_train_1d = Y_train.squeeze(-1) if Y_train.dim() > 1 else Y_train
    Y_pred_gpr = gpr_dot_product_explicit(X_train, Y_train_1d.unsqueeze(-1), X_test, 
                                           sigma_0_sq=noise_variance)
    return Y_pred_gpr


def compute_metrics(Y_true, Y_pred_nn, Y_pred_gpr):
    """Compute comparison metrics between NN and GPR."""
    Y_true_np = Y_true.detach().cpu().numpy().squeeze(-1)
    Y_pred_nn_np = Y_pred_nn.detach().cpu().numpy().squeeze(-1)
    Y_pred_gpr_np = Y_pred_gpr.detach().cpu().numpy().squeeze(-1)
    
    # MSE for each model
    mse_nn = np.mean((Y_true_np - Y_pred_nn_np) ** 2)
    mse_gpr = np.mean((Y_true_np - Y_pred_gpr_np) ** 2)
    
    # NN vs GPR difference
    mse_nn_vs_gpr = np.mean((Y_pred_nn_np - Y_pred_gpr_np) ** 2)
    
    # Correlations
    corr_nn = np.corrcoef(Y_true_np, Y_pred_nn_np)[0, 1]
    corr_gpr = np.corrcoef(Y_true_np, Y_pred_gpr_np)[0, 1]
    
    # R^2 (coefficient of determination)
    ss_res_nn = np.sum((Y_true_np - Y_pred_nn_np) ** 2)
    ss_res_gpr = np.sum((Y_true_np - Y_pred_gpr_np) ** 2)
    ss_tot = np.sum((Y_true_np - np.mean(Y_true_np)) ** 2)
    r2_nn = 1.0 - (ss_res_nn / ss_tot)
    r2_gpr = 1.0 - (ss_res_gpr / ss_tot)
    
    return {
        'mse_nn': float(mse_nn),
        'mse_gpr': float(mse_gpr),
        'mse_nn_vs_gpr': float(mse_nn_vs_gpr),
        'corr_nn': float(corr_nn),
        'corr_gpr': float(corr_gpr),
        'r2_nn': float(r2_nn),
        'r2_gpr': float(r2_gpr),
    }


def plot_comparison(Y_true, Y_pred_nn, Y_pred_gpr, metrics, out_dir, prefix="test"):
    """Create visualization plots for a given split (train/test)."""
    Y_true_np = Y_true.detach().cpu().numpy().squeeze(-1)
    Y_pred_nn_np = Y_pred_nn.detach().cpu().numpy().squeeze(-1)
    Y_pred_gpr_np = Y_pred_gpr.detach().cpu().numpy().squeeze(-1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: True vs NN predictions
    ax = axes[0, 0]
    ax.scatter(Y_true_np, Y_pred_nn_np, alpha=0.5, s=20)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('True Y')
    ax.set_ylabel('NN Prediction')
    ax.set_title(f'{prefix.capitalize()} NN Predictions\nMSE={metrics["mse_nn"]:.4e}, R²={metrics["r2_nn"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: True vs GPR predictions
    ax = axes[0, 1]
    ax.scatter(Y_true_np, Y_pred_gpr_np, alpha=0.5, s=20, color='orange')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('True Y')
    ax.set_ylabel('GPR Prediction')
    ax.set_title(f'{prefix.capitalize()} GPR Predictions\nMSE={metrics["mse_gpr"]:.4e}, R²={metrics["r2_gpr"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: NN vs GPR predictions
    ax = axes[1, 0]
    ax.scatter(Y_pred_gpr_np, Y_pred_nn_np, alpha=0.5, s=20, color='green')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('GPR Prediction')
    ax.set_ylabel('NN Prediction')
    ax.set_title(f'NN vs GPR\nMSE={metrics["mse_nn_vs_gpr"]:.4e}')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals_nn = Y_true_np - Y_pred_nn_np
    residuals_gpr = Y_true_np - Y_pred_gpr_np
    ax.hist(residuals_nn, bins=30, alpha=0.6, label='NN residuals', color='blue')
    ax.hist(residuals_gpr, bins=30, alpha=0.6, label='GPR residuals', color='orange')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'comparison_plots_{prefix}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    print(f'Saved comparison plots to {os.path.join(out_dir, filename)}')
    plt.close(fig)


def plot_eigen_spectra(model, X_train, out_dir):
    """Compute and plot network H-eigenvalues and GPR kernel eigenvalues."""
    with torch.no_grad():
        # Network H eigenvalues
        lH = model.H_eig(X_train, torch.linalg.qr(X_train).Q, std=False)
        lH_np = lH.detach().cpu().numpy()
        lH_np = np.sort(lH_np)[::-1]

        # GPR kernel eigenvalues (dot-product kernel)
        K = X_train @ X_train.T / D# (P, P)
        eigvals = torch.linalg.eigvalsh(K) / X_train.shape[0]
        eigvals_np = eigvals.detach().cpu().numpy()
        eigvals_np = np.sort(eigvals_np)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(np.arange(len(lH_np)), lH_np, color='steelblue')
    axes[0].set_title('Network H eigenvalues')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_yscale('log')

    axes[1].bar(np.arange(len(eigvals_np)), eigvals_np, color='darkorange')
    axes[1].set_title('GPR kernel eigenvalues (dot product)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_yscale('log')

    plt.tight_layout()
    fname = os.path.join(out_dir, 'eigen_spectra.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved eigen spectra to {fname}')
    plt.close(fig)


def plot_weight_distribution(model, out_dir):
    """Plot distribution of W0 weights with Gaussian fit."""
    with torch.no_grad():
        W0_weights = model.W0.detach().cpu().numpy().flatten()
    
    # Fit Gaussian
    mu, std2 = np.mean(W0_weights), np.var(W0_weights) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(W0_weights, bins=50, density=True, alpha=0.6, 
                                color='steelblue', label='W0 weights')
    
    # Fitted Gaussian
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std2**0.5)
    ax.plot(x, p, 'r-', linewidth=2, label=f'Fitted Gaussian\nμ={mu:.4f}, σ2={std2:.4f}')
    
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Density')
    ax.set_title(f'W0 Weight Distribution (Read-in Layer)\nFitted std: {std2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'weight_distribution_W0.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved W0 weight distribution to {fname}')
    print(f'  Fitted Gaussian: μ={mu:.6f}, σ={std2:.6f}')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Train FCN3Erf on d=2 and compare with GPR')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--dry-run', action='store_true', help='Quick test (1 epoch, no comparison)')
    parser.add_argument('--analyze-only', action='store_true', help='Load pre-trained model from disk and skip training')
    parser.add_argument('--output-dir', type=str, default='/home/akiva/FCNX-Ensembling/plots/fcn3_d2_gpr_comparison', 
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:1')
        print(f'Using CUDA device: {device}')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')
    
    # Setup output directory
    out_dir = make_folder(args.output_dir)
    
    # Set random seeds
    torch.manual_seed(MODEL_SEED)
    np.random.seed(DATA_SEED)
    
    print('\n' + '='*80)
    print('FCN3Erf Training and GPR Comparison')
    print('='*80)
    print(f'Hyperparameters:')
    print(f'  d={D}, N={N}, P_train={P_TRAIN}, P_test={P_TEST}')
    print(f'  kappa={KAPPA}, chi={CHI}, eps={EPS}')
    print(f'  epochs={args.epochs}, dry_run={args.dry_run}')
    print(f'  Output directory: {out_dir}')
    print('='*80 + '\n')
    
    # Generate training data
    print('Generating training data...')
    torch.manual_seed(DATA_SEED)
    X_train = torch.randn((P_TRAIN, D), dtype=torch.float32, device=device)
    Y_train_list = []
    pbar = tqdm(range(P_TRAIN), desc='  Generating training samples', unit='sample')
    for i in pbar:
        z = X_train[i, 0]
        He1 = z
        He3 = z ** 3 - 3.0 * z
        Y_val = (He1 + EPS * He3).unsqueeze(-1)
        Y_train_list.append(Y_val)
    Y_train = torch.stack(Y_train_list)
    print(f'  Training data shape: X={X_train.shape}, Y={Y_train.shape}')
    
    # Generate test data
    print('Generating test data...')
    torch.manual_seed(DATA_SEED + 1)
    X_test = torch.randn((P_TEST, D), dtype=torch.float32, device=device)
    Y_test_list = []
    pbar = tqdm(range(P_TEST), desc='  Generating test samples', unit='sample')
    for i in pbar:
        z = X_test[i, 0]
        He1 = z
        He3 = z ** 3 - 3.0 * z
        Y_val = (He1 + EPS * He3).unsqueeze(-1)
        Y_test_list.append(Y_val)
    Y_test = torch.stack(Y_test_list)
    print(f'  Test data shape: X={X_test.shape}, Y={Y_test.shape}')
    
    # Create and train model (or load from disk if analyze-only)
    model_path = os.path.join(out_dir, 'model.pth')
    
    if args.analyze_only:
        # Load pre-trained model from disk
        print('\n[analyze-only mode] Loading pre-trained model...')
        if not os.path.exists(model_path):
            print(f'ERROR: Model file not found at {model_path}')
            print(f'Please ensure the model has been trained before using --analyze-only')
            return
        model = FCN3NetworkEnsembleErf(D, N, N, P_TRAIN, ens=1, device=device)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
        loss_history = []
    else:
        # Create and train new model
        print('\nCreating FCN3Erf model...')
        model = FCN3NetworkEnsembleErf(D, N, N, P_TRAIN, ens=1, device=device)
        model.to(device)
        
        # Compute base learning rate
        lr_base = 1e-6 / P_TRAIN
        print(f'Base learning rate: {lr_base:.3e}')
        
        # Train
        print('\nTraining model...')
        loss_history = train_fcn3erf_langevin(model, X_train, Y_train, args.epochs, lr_base, 
                                              device, model_path=model_path, dry_run=args.dry_run)
        
        # Save trained model (final save)
        torch.save(model.state_dict(), model_path)
        print(f'Saved trained model to {model_path}')
        
        if args.dry_run:
            print('\n[dry-run mode] Skipping comparison.')
            return
    
    model.eval()

    # Inference on train set
    print('\nRunning inference on train set...')
    with torch.no_grad():
        Y_pred_nn_train = model(X_train)
        if Y_pred_nn_train.ndim > 1 and Y_pred_nn_train.shape[-1] == 1:
            Y_pred_nn_train = Y_pred_nn_train.view(-1, 1)
    print(f'  Train NN predictions shape: {Y_pred_nn_train.shape}')

    # Inference on test set
    print('\nRunning inference on test set...')
    Y_pred_list = []
    pbar = tqdm(range(P_TEST), desc='  Running inference', unit='sample')
    with torch.no_grad():
        for i in pbar:
            pred = model(X_test[i:i+1])
            if pred.ndim > 1 and pred.shape[-1] == 1:
                pred = pred.view(-1, 1)
            Y_pred_list.append(pred)
    Y_pred_nn = torch.cat(Y_pred_list, dim=0)
    Y_pred_nn = Y_pred_nn / torch.norm(Y_pred_nn)
    print(f'  Test NN predictions shape: {Y_pred_nn.shape}')
    
    # Compute GPR predictions
    print('\nComputing GPR predictions (train/test)...')
    Y_pred_gpr_train = compute_gpr_predictions(X_train, Y_train, X_train, noise_variance=1.0)
    Y_pred_gpr_train = Y_pred_gpr_train / torch.norm(Y_pred_gpr_train)
    Y_pred_gpr = compute_gpr_predictions(X_train, Y_train, X_test, noise_variance=1.0)
    Y_pred_gpr = Y_pred_gpr / torch.norm (Y_pred_gpr)
    print(f'  Train GPR predictions shape: {Y_pred_gpr_train.shape}')
    print(f'  Test  GPR predictions shape: {Y_pred_gpr.shape}')
    
    # Compute metrics
    print('\nComputing metrics...')
    pbar = tqdm(total=6, desc='  Computing', unit='step')
    pbar.update(1)  # MSE train
    pbar.update(1)  # Corr train
    metrics_train = compute_metrics(Y_train, Y_pred_nn_train, Y_pred_gpr_train)
    pbar.update(1)  # R2 train
    pbar.update(1)  # MSE test
    pbar.update(1)  # Corr test
    metrics = compute_metrics(Y_test, Y_pred_nn, Y_pred_gpr)
    pbar.update(1)  # R2 test
    pbar.close()
    
    # Print results
    print('\n' + '='*80)
    print('RESULTS')
    print('='*80)
    print('Train Split:')
    print(f'  NN MSE:  {metrics_train["mse_nn"]:.6e}')
    print(f'  NN Corr: {metrics_train["corr_nn"]:.6f}')
    print(f'  NN R²:   {metrics_train["r2_nn"]:.6f}')
    print(f'  GPR MSE: {metrics_train["mse_gpr"]:.6e}')
    print(f'  GPR Corr:{metrics_train["corr_gpr"]:.6f}')
    print(f'  GPR R²:  {metrics_train["r2_gpr"]:.6f}')
    print(f'  NN vs GPR MSE diff: {metrics_train["mse_nn_vs_gpr"]:.6e}')
    print('\nTest Split:')
    print(f'  NN MSE:  {metrics["mse_nn"]:.6e}')
    print(f'  NN Corr: {metrics["corr_nn"]:.6f}')
    print(f'  NN R²:   {metrics["r2_nn"]:.6f}')
    print(f'  GPR MSE: {metrics["mse_gpr"]:.6e}')
    print(f'  GPR Corr:{metrics["corr_gpr"]:.6f}')
    print(f'  GPR R²:  {metrics["r2_gpr"]:.6f}')
    print(f'  NN vs GPR MSE diff: {metrics["mse_nn_vs_gpr"]:.6e}')
    print('='*80 + '\n')
    
    # Save metrics (train and test)
    metrics_path = os.path.join(out_dir, 'metrics_test.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    metrics_train_path = os.path.join(out_dir, 'metrics_train.json')
    with open(metrics_train_path, 'w') as f:
        json.dump(metrics_train, f, indent=2)
    print(f'Saved metrics to {metrics_path} and {metrics_train_path}')
    
    # Create plots for train and test
    print('\nGenerating comparison plots (train/test)...')
    plot_comparison(Y_train, Y_pred_nn_train, Y_pred_gpr_train, metrics_train, out_dir, prefix='train')
    plot_comparison(Y_test, Y_pred_nn, Y_pred_gpr, metrics, out_dir, prefix='test')

    # Eigenvalue spectra (network vs GPR kernel)
    print('\nComputing and plotting eigenvalue spectra...')
    plot_eigen_spectra(model, X_train, out_dir)
    
    # Weight distribution for W0 (read-in layer)
    print('\nPlotting W0 weight distribution...')
    plot_weight_distribution(model, out_dir)
    
    # Save loss history
    if loss_history:
        loss_path = os.path.join(out_dir, 'loss_history.json')
        with open(loss_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f'Saved loss history to {loss_path}')
    
    print(f'\nAll results saved to: {out_dir}')


if __name__ == '__main__':
    main()
