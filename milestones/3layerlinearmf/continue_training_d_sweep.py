#!/usr/bin/env python3
"""
Continue training networks from d-sweep, saving eigenvalues periodically.

This script:
1. Loads trained networks from d_sweep.py (trained for 40M epochs)
2. Continues training for 200M additional epochs
3. Saves eigenvalues every checkpoint_interval epochs
4. Stores results indexed by (d, total_epoch)

Usage:
    # Train all d values in parallel:
    python continue_training_d_sweep.py

    # Train specific d value (for manual parallelization):
    python continue_training_d_sweep.py --d 2 --device cuda:0
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import subprocess
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear


def train_network_checkpoint(
    model,
    d,
    P,
    N,
    chi,
    X,
    y,
    device,
    writer=None,
    total_epochs=200_000_000,
    checkpoint_interval=1_000_000,
    lr=1e-5,
    temperature=0.2,
):
    """
    Continue training network, saving eigenvalues at checkpoints.
    
    Args:
        model: Pre-trained network (already partially trained)
        d, P, N, chi: Network parameters
        X, y: Training data
        device: Device to train on
        writer: TensorBoard SummaryWriter (optional)
        total_epochs: Total additional epochs to train
        checkpoint_interval: Save checkpoint every N epochs
        lr: Learning rate (adjusted by P)
        temperature: Temperature parameter (2 * kappa)
    
    Returns:
        List of dicts with (epoch, eigenvalues) at each checkpoint
    """
    model.train()
    loss_fn = nn.MSELoss(reduction='sum')
    lr_adjusted = lr 
    noise_std = np.sqrt(2.0 * lr_adjusted * temperature)
    
    # Weight decay coefficients
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    checkpoint_results = []
    num_checkpoints = total_epochs // checkpoint_interval
    
    # Large eval set for eigenvalues
    Xinf = torch.randn(3000, d, device=device)
    
    print(f"  Training for {total_epochs:,} epochs with {num_checkpoints} checkpoints...")
    print(f"  lr={lr_adjusted:.6e}, temperature={temperature:.6e}, noise_std={noise_std:.6e}")
    
    for checkpoint_idx in range(num_checkpoints):
        # Train for checkpoint_interval epochs
        for step in range(checkpoint_interval):
            model.zero_grad()
            
            output = model(X)  # shape: (P, ensemble)
            # Compute per-ensemble losses
            diff = output - y  # (P, ensemble)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ensemble,)
            loss = per_ensemble_loss.sum()
            
            loss.backward()
            
            # Langevin update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if 'W0' == name:
                        wd = wd_fc1
                    elif 'W1' == name:
                        wd = wd_fc2
                    elif 'A' == name:
                        wd = wd_fc3
                    else:
                        wd = 0
                    
                    noise = torch.randn_like(param) * noise_std
                    param.add_(-lr_adjusted * param.grad)
                    param.add_(-lr_adjusted * wd * param.data)
                    param.add_(noise)
        
        # Save checkpoint
        total_epoch = (checkpoint_idx + 1) * checkpoint_interval
        
        with torch.no_grad():
            try:
                eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                eigvals_list = eigenvalues.tolist()
                lH = float(eigenvalues.max())
            except Exception as e:
                print(f"    Warning: Could not compute eigenvalues at epoch {total_epoch}: {e}")
                eigvals_list = []
                lH = np.nan
        
        checkpoint_results.append({
            "epoch": total_epoch,
            "eigenvalues": eigvals_list,
        })
        
        # Log to TensorBoard
        if writer is not None and len(eigvals_list) > 0:
            # Log top eigenvalue
            writer.add_scalar('eigenvalues/top', lH, total_epoch)
            
            # Log mean and std of remaining eigenvalues
            if len(eigvals_list) > 1:
                remaining = eigenvalues[1:]
                writer.add_scalar('eigenvalues/mean_rest', remaining.mean(), total_epoch)
                writer.add_scalar('eigenvalues/std_rest', remaining.std(), total_epoch)
            
            # Log histogram of all eigenvalues
            writer.add_histogram('eigenvalues/distribution', eigenvalues, total_epoch)
        
        print(f"  Checkpoint {checkpoint_idx + 1}/{num_checkpoints} (epoch {total_epoch:,}): lH={lH:.6f}")
    
    return checkpoint_results


def train_single_d(d: int, device_str: str = "cuda:1", P: int = 6, N: int = 50):
    """Train a single d value - can be called in parallel."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    chi = N
    kappa = 0.1
    
    print(f"\n{'='*60}")
    print(f"Continuing training for d={d} on {device}")
    print(f"{'='*60}")
    
    base_dir = Path(__file__).resolve().parent
    checkpoint_dir = base_dir / "checkpoints_continued"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    tensorboard_dir = base_dir / "runs_continued" / f"d{d}"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"  TensorBoard logging to: {tensorboard_dir}")
    
    # Load configuration
    run_dir = base_dir / f"d{d}_P{P}_N{N}_chi{N}"
    config_path = run_dir / "config.json"
    
    if not config_path.exists():
        print(f"  ERROR: Config file not found: {config_path}")
        return False
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    original_epochs = config.get("epochs", 40_000_000)
    lr = 1e-6 #config.get("lr", 1e-7)
    kappa = config.get("kappa", 0.1)
    temperature = 2 * kappa
    
    print(f"  Original training: {original_epochs:,} epochs")
    print(f"  Parameters: P={P}, N={N}, d={d}, chi={chi}, kappa={kappa}")
    
    # Load pre-trained network
    model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        print(f"  ERROR: Model file not found: {model_path}")
        return False
    
    # Recreate model with same architecture
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkEnsembleLinear(
        d=d, n1=N, n2=N, P=P, ensembles=50,
        weight_initialization_variance=weight_var,
    ).to(device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Recreate data (same seed as d_sweep.py)
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Continue training for 200M epochs
    results = train_network_checkpoint(
        model, d, P, N, chi, X, y, device,
        writer=writer,
        total_epochs=200_000_000,
        checkpoint_interval=1_000_000,
        lr=lr,
        temperature=temperature,
    )
    
    # Close TensorBoard writer
    writer.close()
    
    # Save checkpoint results
    out_path = checkpoint_dir / f"d{d}_continued.json"
    with out_path.open("w") as f:
        json.dump({
            "d": d,
            "P": P,
            "N": N,
            "chi": chi,
            "kappa": kappa,
            "original_epochs": original_epochs,
            "continued_epochs": 200_000_000,
            "checkpoint_interval": 1_000_000,
            "checkpoints": results,
        }, f, indent=2)
    
    print(f"  Saved checkpoint results to {out_path}")
    
    # Also save final model
    final_model_path = checkpoint_dir / f"d{d}_model_continued.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"  Saved final model to {final_model_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Continue training d-sweep networks')
    parser.add_argument('--d', type=int, default=None, 
                       help='Train specific d value (if not set, launches all in parallel)')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--P', type=int, default=6, help='P parameter')
    parser.add_argument('--N', type=int, default=50, help='N parameter')
    args = parser.parse_args()
    
    dims = [2, 6, 8, 10]
    
    if args.d is not None:
        # Train single d value
        if args.d not in dims:
            print(f"ERROR: d={args.d} not in available dims {dims}")
            sys.exit(1)
        success = train_single_d(args.d, args.device, args.P, args.N)
        sys.exit(0 if success else 1)
    
    # Launch parallel training for all d values
    print("Launching parallel training for all d values...")
    print("=" * 60)
    
    script_path = Path(__file__).resolve()
    processes = []
    
    # Assign different GPUs to different d values
    gpu_devices = ["cuda:0", "cuda:1"]
    
    for idx, d in enumerate(dims):
        device = gpu_devices[idx % len(gpu_devices)]
        cmd = [
            sys.executable,
            str(script_path),
            "--d", str(d),
            "--device", device,
            "--P", str(args.P),
            "--N", str(args.N),
        ]
        
        print(f"Launching d={d} on {device}...")
        proc = subprocess.Popen(cmd)
        processes.append((d, proc))
    
    print(f"\nLaunched {len(processes)} parallel training processes")
    print("Waiting for all processes to complete...")
    
    # Wait for all processes
    for d, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"✓ d={d} completed successfully")
        else:
            print(f"✗ d={d} failed with return code {proc.returncode}")
    
    print("\nAll training completed!")


if __name__ == "__main__":
    main()

