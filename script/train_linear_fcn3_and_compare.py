#!/usr/bin/env python3
"""
Train a linear FCN3 network with the same parameters as train_and_compare_gpr_d2.py
and plot the resulting eigenvalues (H and J kernels) against theoretical expectations
computed using the ExperimentLinear class.

This script:
1. Sets up hyperparameters matching train_and_compare_gpr_d2.py but with linear activation.
2. Trains a FCN3NetworkEnsembleLinear model using Langevin dynamics.
3. Computes empirical eigenvalues (H_eig, J_eig) from the trained model.
4. Computes theoretical eigenvalue predictions using ExperimentLinear.eig_predictions().
5. Creates comparison plots: empirical vs theoretical for both J and H kernels.

Usage:
    python3 script/train_linear_fcn3_and_compare.py --epochs 100000 [--cuda]
    
    # Dry run (quick test):
    python3 script/train_linear_fcn3_and_compare.py --epochs 100 --dry-run
    
    # Load pre-trained model and skip training:
    python3 script/train_linear_fcn3_and_compare.py --analyze-only [--cuda]
"""

import os
import sys
import json
import time
import math
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Project layout helpers
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from FCN3Network import FCN3NetworkEnsembleLinear
from Experiment import ExperimentLinear, DATA_SEED, MODEL_SEED, LANGEVIN_SEED
from GPKit import gpr_dot_product_explicit

# ============================================================================
# Hyperparameters (matching train_and_compare_gpr_d2.py but linear)
# ============================================================================
D = 4
N = 20
KAPPA = 1.0
CHI = N 
P_TRAIN = 8

# Data and model seeds
DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480

# Optimization
DEFAULT_EPOCHS = 25_000_000
EXTENSION_EPOCHS = 25_000_000

ALPHA_T = 0.7
LR_SCHEDULE_MULTIPLIER = 3.0
ENSEMBLES = 100
DATASETS = 1
NDUP = ENSEMBLES * DATASETS

LR_EXTENSION_FACTOR = 1e-2
# ============================================================================
# Helper functions
# ============================================================================

def make_folder(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def generate_training_data_linear(P, d, device, seed=DATA_SEED):
    """Generate synthetic training data for linear networks: Y = He1(X) where X ~ N(0, I)."""
    torch.manual_seed(seed)
    X = torch.randn((P, d), dtype=torch.float32, device=device)
    z = X[:,0]
    He1 = z
    Y = He1
    return X, Y, He1


def train_fcn3linear_langevin(model, X_train, Y_train, epochs, lr, device, 
                             model_path=None, dry_run=False, tensorboard_writer=None):
    """
    Train FCN3NetworkEnsembleLinear using Langevin dynamics.

    Args:
        model: FCN3NetworkEnsembleLinear model
        X_train: training input (P, d)
        Y_train: training target (P, 1)
        epochs: number of training epochs
        lr: base learning rate
        device: torch device
        model_path: path to save model checkpoints
        dry_run: if True, only do 1 epoch
        tensorboard_writer: SummaryWriter for TensorBoard logging

    Returns:
        loss_history: list of losses (one per epoch)
        eigenvalue_history: dict with 'epochs', 'lJ_first', 'lJ_rest_mean', 'lH_first', 'lH_rest_mean'
    """
    Xinf = generate_training_data_linear(1000, D, device, seed=DATA_SEED)[0]

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
        diff = outputs - targets.unsqueeze(-1)

        return torch.sum(diff * diff)
    
    # History tracking
    loss_history = []
    eigenvalue_history = {
        'epochs': [],
        'lJ_first': [],
        'lJ_rest_mean': [],
        'lH_first': [],
        'lH_rest_mean': []
    }
    
    print(f'Starting Langevin training: {epochs} epochs, lr={lr:.3e}, T={T}')
    print(f'Computing eigenvalues every 10000 epochs')
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
        
        loss = custom_mse_loss(outputs, Y_train)
        
        # Backward
        loss.backward()
        
        # Parameter update with Langevin noise and weight decay
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    # Langevin noise
                    noise_buffer.resize_(param.shape).normal_(
                        0, noise_scale, generator=langevin_gen)
                    param.add_(noise_buffer)
                    
                    # Weight decay
                    param.add_(
                        param.data, alpha=-(weight_decay_vec[i]).item() * effective_lr)
                    
                    # Gradient descent
                    param.add_(param.grad, alpha=-effective_lr)
                else:
                    print("GRAD IS NONE")
        
        # Store loss at every epoch
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Log to TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Training/Loss', loss_val, epoch)
        try:
            # Compute eigenvalues every 10000 epochs
            if epoch % 1000 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    # J eigenvalues
                    lJ = model.J_eig(Xinf, Xinf)
                    lJ_np = lJ.detach().cpu().numpy()
                    lJ_np = np.atleast_1d(lJ_np)
                    # H eigenvalues
                    lH = model.H_eig(Xinf, Xinf, std=False)
                    lH_np = lH.detach().cpu().numpy()
                    lH_np = np.atleast_1d(lH_np)


                    # Store: first eigenvalue and mean of rest
                    lJ_first = float(lJ_np[0]) if len(lJ_np) > 0 else 0.0
                    lJ_rest_mean = float(np.mean(lJ_np[1:])) if len(lJ_np) > 1 else 0.0
                    lH_first = float(lH_np[0]) if len(lH_np) > 0 else 0.0
                    lH_rest_mean = float(np.mean(lH_np[1:])) if len(lH_np) > 1 else 0.0
                    
                    eigenvalue_history['epochs'].append(epoch)
                    eigenvalue_history['lJ_first'].append(lJ_first)
                    eigenvalue_history['lJ_rest_mean'].append(lJ_rest_mean)
                    eigenvalue_history['lH_first'].append(lH_first)
                    eigenvalue_history['lH_rest_mean'].append(lH_rest_mean)
                    
                    # Log to TensorBoard
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar('Eigenvalues/lJ_first', lJ_first, epoch)
                        tensorboard_writer.add_scalar('Eigenvalues/lJ_rest_mean', lJ_rest_mean, epoch)
                        tensorboard_writer.add_scalar('Eigenvalues/lH_first', lH_first, epoch)
                        tensorboard_writer.add_scalar('Eigenvalues/lH_rest_mean', lH_rest_mean, epoch)
                        
                        # Also log all eigenvalues as histograms
                        if len(lJ_np) > 0:
                            tensorboard_writer.add_histogram('Eigenvalues/lJ_all', lJ_np, epoch)
                        if len(lH_np) > 0:
                            tensorboard_writer.add_histogram('Eigenvalues/lH_all', lH_np, epoch)
        except Exception as e:
            print(f"Error computing eigenvalues at epoch {epoch}: {e}") 
                
        try:    
            # Save model checkpoint
            if (epoch % 100000 == 0 or epoch == epochs) and model_path is not None:
                torch.save(model.state_dict(), model_path)
        except Exception as e:
            print(f"Error saving model at epoch {epoch}: {e}")
        
        model.train()
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6e}'})
    
    pbar.close()
    elapsed = time.time() - t0
    print(f'Training completed in {elapsed:.2f}s')
    return loss_history, eigenvalue_history


def compute_empirical_eigenvalues(model, X_train, Y_train, device):
    """
    Compute empirical eigenvalues from trained model.
    
    Args:
        model: trained FCN3NetworkEnsembleLinear
        X_train: training input (P, d)
        Y_train: training target (P, 1)
        device: torch device
    
    Returns:
        dict with arrays: lJ_array (shape d,), lH_array (shape d,)
    """
    model.eval()
    with torch.no_grad():
        # J eigenvalue (readout kernel) - returns d eigenvalues
        lJ = model.J_eig(X_train, X_train)
        lJ_np = lJ.detach().cpu().numpy()
        
        # H eigenvalue (hidden layer kernel) - returns d eigenvalues
        lH = model.H_eig(X_train, X_train, std=False)
        lH_np = lH.detach().cpu().numpy()
    
    return {
        'lJ_array': np.atleast_1d(lJ_np),  # Ensure it's an array
        'lH_array': np.atleast_1d(lH_np),  # Ensure it's an array
    }


def compute_theoretical_eigenvalues(d, n, chi, P, kappa=1.0, device='cpu'):
    """
    Compute theoretical eigenvalue predictions using ExperimentLinear.
    
    Args:
        d: input dimension
        n: hidden width (N1 = N2 = n)
        chi: aspect ratio (N2 / input_dim)
        P: number of training samples
        kappa: temperature parameter
        device: torch device
    
    Returns:
        dict with theoretical predictions, or None if computation fails
    """
    try:
        # Create a temporary directory for ExperimentLinear
        # ExperimentLinear needs a 'file' path (directory)
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentLinear(
                file=tmpdir,
                N=n,
                d=d,
                chi=chi,
                P=P,
                ens=1,
                device=device,
                kappa=kappa
            )
            
            # Compute theoretical predictions
            preds = exp.eig_predictions()
            
            if preds is None:
                print("Warning: eig_predictions() returned None")
                return None
            
            return {
                'lJT': float(preds.lJT) if hasattr(preds, 'lJT') else None,
                'lHT': float(preds.lHT) if hasattr(preds, 'lHT') else None,
                'lJP': float(preds.lJP) if hasattr(preds, 'lJP') else None,
                'lHP': float(preds.lHP) if hasattr(preds, 'lHP') else None,
                'lKT': float(preds.lKT) if hasattr(preds, 'lKT') and preds.lKT is not None else None,
                'lKP': float(preds.lKP) if hasattr(preds, 'lKP') and preds.lKP is not None else None,
            }
    except Exception as e:
        print(f"Error computing theoretical eigenvalues: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_gpr_predictions(X_train, Y_train, device, noise_variance=1.0):
    """
    Compute GPR predictions on training data using dot-product kernel.
    
    Args:
        X_train: training input (P, d)
        Y_train: training target (P, 1)
        device: torch device
        noise_variance: observation noise variance
    
    Returns:
        Y_pred_gpr: (P, 1) GPR predictions on training data
    """
    Y_pred_gpr = gpr_dot_product_explicit(X_train, Y_train, X_train, sigma_0_sq=noise_variance)
    return Y_pred_gpr


def compare_network_with_gpr(model, X_train, Y_train, device):
    """
    Compare network outputs with GPR predictions.
    
    Args:
        model: trained network
        X_train: training input (P, d)
        Y_train: training targets (P, 1)
        device: torch device
    
    Returns:
        dict with comparison metrics
    """
    model.eval()
    with torch.no_grad():
        # Network predictions
        Y_pred_nn = model(X_train).mean(dim=1) # average over ensembles
        if Y_pred_nn.ndim > 1 and Y_pred_nn.shape[-1] == 1:
            Y_pred_nn = Y_pred_nn.view(-1, 1)
        
        # GPR predictions
        Y_pred_gpr = compute_gpr_predictions(X_train, Y_train, device, noise_variance=1.0)

    # Convert to numpy for metrics
    Y_true_np = Y_train.detach().cpu().numpy().squeeze()
    Y_pred_nn_np = Y_pred_nn.detach().cpu().numpy().squeeze()
    Y_pred_gpr_np = Y_pred_gpr.detach().cpu().numpy().squeeze()
    
    # Normalize for fair comparison
    Y_pred_nn_norm = Y_pred_nn_np / np.linalg.norm(Y_pred_nn_np)
    Y_pred_gpr_norm = Y_pred_gpr_np / np.linalg.norm(Y_pred_gpr_np)
    Y_true_norm = Y_true_np / np.linalg.norm(Y_true_np)
    
    # Compute metrics
    mse_nn = np.mean((Y_true_norm - Y_pred_nn_norm) ** 2)
    mse_gpr = np.mean((Y_true_norm - Y_pred_gpr_norm) ** 2)
    
    # Correlation between NN and GPR
    corr_nn_gpr = np.corrcoef(Y_pred_nn_norm, Y_pred_gpr_norm)[0, 1]
    
    # Correlation with targets
    corr_nn_target = np.corrcoef(Y_true_norm, Y_pred_nn_norm)[0, 1]
    corr_gpr_target = np.corrcoef(Y_true_norm, Y_pred_gpr_norm)[0, 1]
    
    # MSE between NN and GPR
    mse_nn_vs_gpr = np.mean((Y_pred_nn_norm - Y_pred_gpr_norm) ** 2)
    
    return {
        'mse_nn': float(mse_nn),
        'mse_gpr': float(mse_gpr),
        'mse_nn_vs_gpr': float(mse_nn_vs_gpr),
        'corr_nn_gpr': float(corr_nn_gpr),
        'corr_nn_target': float(corr_nn_target),
        'corr_gpr_target': float(corr_gpr_target),
        'Y_pred_nn': Y_pred_nn_np,
        'Y_pred_gpr': Y_pred_gpr_np,
        'Y_true': Y_true_np,
    }


def plot_gpr_comparison(comparison_results, out_dir):
    """
    Plot network predictions vs GPR predictions.
    
    Args:
        comparison_results: dict from compare_network_with_gpr
        out_dir: output directory
    """
    Y_true = comparison_results['Y_true']
    Y_pred_nn = comparison_results['Y_pred_nn']
    Y_pred_gpr = comparison_results['Y_pred_gpr']
    
    # Normalize for plotting
    Y_pred_nn_norm = Y_pred_nn / np.linalg.norm(Y_pred_nn)
    Y_pred_gpr_norm = Y_pred_gpr / np.linalg.norm(Y_pred_gpr)
    Y_true_norm = Y_true / np.linalg.norm(Y_true)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: NN vs True
    ax = axes[0, 0]
    ax.scatter(Y_true_norm, Y_pred_nn_norm, alpha=0.6, s=40)
    lims = [min(Y_true_norm.min(), Y_pred_nn_norm.min()), 
            max(Y_true_norm.max(), Y_pred_nn_norm.max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('True Y (normalized)')
    ax.set_ylabel('NN Prediction (normalized)')
    ax.set_title(f'Network Predictions\nMSE={comparison_results["mse_nn"]:.6f}, r={comparison_results["corr_nn_target"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GPR vs True
    ax = axes[0, 1]
    ax.scatter(Y_true_norm, Y_pred_gpr_norm, alpha=0.6, s=40, color='orange')
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('True Y (normalized)')
    ax.set_ylabel('GPR Prediction (normalized)')
    ax.set_title(f'GPR Predictions\nMSE={comparison_results["mse_gpr"]:.6f}, r={comparison_results["corr_gpr_target"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: NN vs GPR
    ax = axes[1, 0]
    ax.scatter(Y_pred_gpr_norm, Y_pred_nn_norm, alpha=0.6, s=40, color='green')
    lims = [min(Y_pred_gpr_norm.min(), Y_pred_nn_norm.min()),
            max(Y_pred_gpr_norm.max(), Y_pred_nn_norm.max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlabel('GPR Prediction (normalized)')
    ax.set_ylabel('NN Prediction (normalized)')
    ax.set_title(f'NN vs GPR\nMSE={comparison_results["mse_nn_vs_gpr"]:.6f}, r={comparison_results["corr_nn_gpr"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals_nn = Y_true_norm - Y_pred_nn_norm
    residuals_gpr = Y_true_norm - Y_pred_gpr_norm
    ax.hist(residuals_nn, bins=20, alpha=0.6, label='NN residuals', color='blue')
    ax.hist(residuals_gpr, bins=20, alpha=0.6, label='GPR residuals', color='orange')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'gpr_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved GPR comparison plot to {fname}')
    plt.close(fig)


def plot_eigenvalue_comparison(empirical, theoretical, out_dir, d):
    """
    Plot empirical vs theoretical eigenvalues across all d dimensions.
    
    For H eigenvalues: first eigenvalue is lHT, rest are degenerate lHP eigenvalues.
    For J eigenvalues: first eigenvalue is lJT, rest are degenerate lJP eigenvalues.
    
    Args:
        empirical: dict with 'lJ_array', 'lH_array' (each shape d,)
        theoretical: dict with scalar predictions 'lJT', 'lHT', 'lJP', 'lHP'
        out_dir: output directory
        d: input dimension (number of eigenvalues per kernel)
    """
    # Get arrays
    lJ_emp = empirical.get('lJ_array', np.array([]))
    lH_emp = empirical.get('lH_array', np.array([]))
    
    # Create figure with two subplots (J and H)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- J Eigenvalues ---
    ax = axes[0]
    x_pos = np.arange(len(lJ_emp))
    
    ax.bar(x_pos, lJ_emp, alpha=0.8, color='steelblue', edgecolor='black', label='Empirical')
    
    # First eigenvalue is lJT (training), rest are lJP (population)
    if theoretical and theoretical.get('lJT') is not None:
        ax.axhline(y=theoretical['lJT'], color='orange', linestyle='--', linewidth=2, 
                  label=f'Theory lJT = {theoretical["lJT"]:.6f}')
    if theoretical and theoretical.get('lJP') is not None:
        ax.axhline(y=theoretical['lJP'], color='green', linestyle='--', linewidth=2, 
                  label=f'Theory lJP = {theoretical["lJP"]:.6f}')
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('J Eigenvalues (Readout Kernel)\nFirst is lJT, rest are degenerate lJP')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- H Eigenvalues ---
    ax = axes[1]
    x_pos = np.arange(len(lH_emp))
    
    ax.bar(x_pos, lH_emp, alpha=0.8, color='steelblue', edgecolor='black', label='Empirical')
    
    # First eigenvalue is lHT (training), rest are lHP (population)
    if theoretical and theoretical.get('lHT') is not None:
        ax.axhline(y=theoretical['lHT'], color='orange', linestyle='--', linewidth=2, 
                  label=f'Theory lHT = {theoretical["lHT"]:.6f}')
    if theoretical and theoretical.get('lHP') is not None:
        ax.axhline(y=theoretical['lHP'], color='green', linestyle='--', linewidth=2, 
                  label=f'Theory lHP = {theoretical["lHP"]:.6f}')
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('H Eigenvalues (Hidden Layer Kernel)\nFirst is lHT, rest are degenerate lHP')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'eigenvalue_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved eigenvalue comparison plot to {fname}')
    plt.close(fig)


def plot_loss_history(loss_history, out_dir):
    """Plot training loss history (loss at every epoch)."""
    if not loss_history:
        return
    
    epochs = np.arange(1, len(loss_history) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, loss_history, linewidth=1, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History (Every Epoch)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'loss_history.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved loss history plot to {fname}')
    plt.close(fig)


def plot_eigenvalue_evolution(eigenvalue_history, theoretical, out_dir):
    """Plot evolution of eigenvalues during training."""
    if not eigenvalue_history or not eigenvalue_history['epochs']:
        return
    
    epochs = eigenvalue_history['epochs']
    lJ_first = eigenvalue_history['lJ_first']
    lJ_rest_mean = eigenvalue_history['lJ_rest_mean']
    lH_first = eigenvalue_history['lH_first']
    lH_rest_mean = eigenvalue_history['lH_rest_mean']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- J Eigenvalues Evolution ---
    ax = axes[0]
    ax.plot(epochs, lJ_first, 'o-', label='lJ[0] (first eigenvalue)', 
            linewidth=2, markersize=6, color='steelblue')
    ax.plot(epochs, lJ_rest_mean, 's-', label='mean(lJ[1:]) (rest eigenvalues)', 
            linewidth=2, markersize=6, color='darkblue')
    
    # Add theoretical lines
    if theoretical and theoretical.get('lJT') is not None:
        ax.axhline(y=theoretical['lJT'], color='orange', linestyle='--', 
                  linewidth=2, label=f'Theory lJT = {theoretical["lJT"]:.6f}')
    if theoretical and theoretical.get('lJP') is not None:
        ax.axhline(y=theoretical['lJP'], color='green', linestyle='--', 
                  linewidth=2, label=f'Theory lJP = {theoretical["lJP"]:.6f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('J Eigenvalues Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- H Eigenvalues Evolution ---
    ax = axes[1]
    ax.plot(epochs, lH_first, 'o-', label='lH[0] (first eigenvalue)', 
            linewidth=2, markersize=6, color='steelblue')
    ax.plot(epochs, lH_rest_mean, 's-', label='mean(lH[1:]) (rest eigenvalues)', 
            linewidth=2, markersize=6, color='darkblue')
    
    # Add theoretical lines
    if theoretical and theoretical.get('lHT') is not None:
        ax.axhline(y=theoretical['lHT'], color='orange', linestyle='--', 
                  linewidth=2, label=f'Theory lHT = {theoretical["lHT"]:.6f}')
    if theoretical and theoretical.get('lHP') is not None:
        ax.axhline(y=theoretical['lHP'], color='green', linestyle='--', 
                  linewidth=2, label=f'Theory lHP = {theoretical["lHP"]:.6f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('H Eigenvalues Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'eigenvalue_evolution.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved eigenvalue evolution plot to {fname}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train linear FCN3 and compare eigenvalues with theory')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, 
                       help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--dry-run', action='store_true', help='Quick test (1 epoch)')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Load pre-trained model and skip training')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/akiva/FCNX-Ensembling/plots/fcn3_linear_eig_comparison',
                       help='Output directory for results')
    parser.add_argument('--checkpoint', action='store_true', 
                       help='Load from checkpoint if available and continue training')
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
    print('Linear FCN3 Training and Eigenvalue Comparison')
    print('='*80)
    print(f'Hyperparameters:')
    print(f'  d={D}, N={N}, P_train={P_TRAIN}')
    print(f'  kappa={KAPPA}, chi={CHI}')
    print(f'  epochs={args.epochs}, dry_run={args.dry_run}')
    print(f'  Output directory: {out_dir}')
    print('='*80 + '\n')
    
    # Generate training data
    print('Generating training data...')
    torch.manual_seed(DATA_SEED)
    X_train, Y_train, He1 = generate_training_data_linear(P_TRAIN, D, device, seed=DATA_SEED)
    print(f'  Training data shape: X={X_train.shape}, Y={Y_train.shape}')
    
    # Model path
    model_path = os.path.join(out_dir, 'model_linear.pth')
    

    if args.analyze_only:
        # Load pre-trained model
        print('\n[analyze-only mode] Loading pre-trained model...')
        if not os.path.exists(model_path):
            print(f'ERROR: Model file not found at {model_path}')
            return
        model = FCN3NetworkEnsembleLinear(
            D, N, N, P_TRAIN, 
            ensembles=NDUP,
            weight_initialization_variance=(1/D, 1.0/N, 1.0/(N*CHI))
        )
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
        loss_history = []
        eigenvalue_history = {'epochs': [], 'lJ_first': [], 'lJ_rest_mean': [], 'lH_first': [], 'lH_rest_mean': []}
    else:
        if args.checkpoint and os.path.exists(model_path):
            print(f'\nCheckpoint found at {model_path}, loading model...')
            model = FCN3NetworkEnsembleLinear(
                D, N, N, P_TRAIN,
                ensembles=NDUP,
                weight_initialization_variance=(1/D, 1.0/N, 1.0/(N*CHI))
            )
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'Loaded model from {model_path}')
        else: 
        # Create and train model
            print('\nCreating linear FCN3 model...')
            model = FCN3NetworkEnsembleLinear(
                D, N, N, P_TRAIN,
                ensembles=NDUP,
                weight_initialization_variance=(1/D, 1.0/N, 1.0/(N*CHI))
            )
            model.to(device)
        

        # Compute base learning rate
        lr_base = 1e-8 / P_TRAIN
        epochs = args.epochs
        print(f'Base learning rate: {lr_base:.3e}')
        if args.checkpoint:
            lr_base *= LR_EXTENSION_FACTOR        
            epochs = EXTENSION_EPOCHS
            print(f'Checkpointing enabled, adjusted learning rate: {lr_base:.3e}, total epochs: {epochs}')
        # Create TensorBoard writer
        # Use same log dir to append to existing logs (won't overwrite)
        tensorboard_dir = os.path.join(out_dir, 'tensorboard')
        # Note: SummaryWriter appends to existing event files in the same directory
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f'TensorBoard logs will be saved to: {tensorboard_dir}')
        print(f'To view: tensorboard --logdir={tensorboard_dir}')
        print(f'(Logs will append to existing runs in this directory)')
        
        # Log hyperparameters to TensorBoard
        hparams = {
            'd': D, 'N': N, 'P': P_TRAIN, 'chi': CHI, 'kappa': KAPPA,
            'lr_base': lr_base, 'temperature': 2.0 * KAPPA,
            'epochs': epochs, 'lr_schedule_alpha_T': ALPHA_T
        }
        writer.add_hparams(hparams, {})
        
        # Train
        print('\nTraining model...')
        loss_history, eigenvalue_history = train_fcn3linear_langevin(
            model, X_train, Y_train, epochs, lr_base, device,
            model_path=model_path, dry_run=args.dry_run, tensorboard_writer=writer
        )
        
        writer.close()
        print(f'\nTensorBoard logs saved. To view, run: tensorboard --logdir={tensorboard_dir}')
        
        # Save trained model
        torch.save(model.state_dict(), model_path)
        print(f'Saved trained model to {model_path}')
        
        if args.dry_run:
            print('\n[dry-run mode] Skipping analysis.')
            return
    
    model.eval()
    
    # Compute empirical eigenvalues
    print('\nComputing empirical eigenvalues...')
    empirical_eigs = compute_empirical_eigenvalues(model, X_train, Y_train, device)
    lJ_emp = empirical_eigs['lJ_array']
    lH_emp = empirical_eigs['lH_array']
    print(f'  Empirical lJ (all {len(lJ_emp)} dimensions): {lJ_emp}')
    print(f'  Empirical lJ (mean): {np.mean(lJ_emp):.8f}')
    print(f'  Empirical lH (all {len(lH_emp)} dimensions): {lH_emp}')
    print(f'  Empirical lH (mean): {np.mean(lH_emp):.8f}')
    
    # Compute theoretical eigenvalues
    print('\nComputing theoretical eigenvalue predictions...')
    theoretical_eigs = compute_theoretical_eigenvalues(D, N, CHI, P_TRAIN, kappa=KAPPA, device=device)
    
    if theoretical_eigs:
        print(f'  Theory lJT (training): {theoretical_eigs["lJT"]:.8f}' if theoretical_eigs["lJT"] else '  Theory lJT: N/A')
        print(f'  Theory lHT (training): {theoretical_eigs["lHT"]:.8f}' if theoretical_eigs["lHT"] else '  Theory lHT: N/A')
        print(f'  Theory lJP (population): {theoretical_eigs["lJP"]:.8f}' if theoretical_eigs["lJP"] else '  Theory lJP: N/A')
        print(f'  Theory lHP (population): {theoretical_eigs["lHP"]:.8f}' if theoretical_eigs["lHP"] else '  Theory lHP: N/A')
    else:
        print('  Warning: Failed to compute theoretical predictions')
    
    # Compare network outputs with GPR
    print('\nComparing network predictions with GPR...')
    gpr_comparison = compare_network_with_gpr(model, X_train, Y_train, device)
    print(f'  NN MSE: {gpr_comparison["mse_nn"]:.6e}')
    print(f'  GPR MSE: {gpr_comparison["mse_gpr"]:.6e}')
    print(f'  NN vs GPR MSE: {gpr_comparison["mse_nn_vs_gpr"]:.6e}')
    print(f'  NN-Target correlation: {gpr_comparison["corr_nn_target"]:.6f}')
    print(f'  GPR-Target correlation: {gpr_comparison["corr_gpr_target"]:.6f}')
    print(f'  NN-GPR correlation: {gpr_comparison["corr_nn_gpr"]:.6f}')
    
    # Save results
    results = {
        'hyperparameters': {
            'd': D,
            'N': N,
            'P': P_TRAIN,
            'chi': CHI,
            'kappa': KAPPA,
            'epochs': args.epochs,
        },
        'empirical': {
            'lJ_array': [float(x) for x in lJ_emp],
            'lJ_mean': float(np.mean(lJ_emp)),
            'lH_array': [float(x) for x in lH_emp],
            'lH_mean': float(np.mean(lH_emp)),
        },
        'theoretical': theoretical_eigs if theoretical_eigs else {},
        'eigenvalue_history': eigenvalue_history,
        'gpr_comparison': {
            'mse_nn': gpr_comparison['mse_nn'],
            'mse_gpr': gpr_comparison['mse_gpr'],
            'mse_nn_vs_gpr': gpr_comparison['mse_nn_vs_gpr'],
            'corr_nn_gpr': gpr_comparison['corr_nn_gpr'],
            'corr_nn_target': gpr_comparison['corr_nn_target'],
            'corr_gpr_target': gpr_comparison['corr_gpr_target'],
        },
    }
    
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')
    
    # Create comparison plots
    print('\nGenerating eigenvalue comparison plot...')
    plot_eigenvalue_comparison(empirical_eigs, theoretical_eigs, out_dir, D)
    
    # Plot GPR comparison
    print('Generating GPR comparison plot...')
    plot_gpr_comparison(gpr_comparison, out_dir)
    
    # Plot eigenvalue evolution during training
    if eigenvalue_history and eigenvalue_history['epochs']:
        print('Generating eigenvalue evolution plot...')
        plot_eigenvalue_evolution(eigenvalue_history, theoretical_eigs, out_dir)
    
    # Plot loss history if available
    if loss_history:
        print('Generating loss history plot...')
        plot_loss_history(loss_history, out_dir)
    
    print(f'\nAll results saved to: {out_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
