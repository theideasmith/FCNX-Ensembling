#!/usr/bin/env python3
"""
Training script for 2-layer linear network with MEAN-FIELD scaling.

Trains a single 2-layer network: Input(d) -> Hidden(n1) with linear -> Output(1)
Tracks eigenvalues of the pre-activation kernel H over training.

Mean-field scaling:
    - Temperature: T_eff = T / chi
    - Readout initialization: variance *= 1/chi  => (1/d, 1/(N*chi))
    - W0 weight decay: wd_A = N * chi * T_eff

Usage:
    python train_fcn2_linear_mf.py --d 10 --P 30 --N 100 --epochs 2000000 --device cuda:0
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter

# Set default dtype
torch.set_default_dtype(torch.float32)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def custom_mse_loss(outputs, targets):
    """MSE loss summed over all samples and ensembles."""
    diff = outputs - targets
    return torch.sum(diff * diff)


def train_fcn2_linear(d, P, N, epochs=2_000_000, log_interval=10_000, 
                      device_str="cuda:1", lr=1e-5, temperature=0.02, chi=1.0,
                      run_dir=None, writer=None):
    """Train 2-layer linear network with MEAN-FIELD scaling.
    
    Args:
        d: Input dimension
        P: Number of training samples
        N: Hidden layer width
        epochs: Number of training iterations
        log_interval: Log eigenvalues every N epochs
        device_str: Device string
        lr: Learning rate
        temperature: Base temperature for weight decay and Langevin noise
        chi: Scaling factor; effective temperature = temperature / chi
        run_dir: Directory to save checkpoints and results
        writer: TensorBoard writer
        
    Returns:
        (final_eigenvalues, eigenvalues_over_time, run_dir)
    """
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Setup directory
    if run_dir is None:
        run_dir = Path(__file__).parent / f"linear_mf_d{d}_P{P}_N{N}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining 2-layer linear network (MEAN-FIELD scaling):")
    print(f"  d={d}, P={P}, N={N}")
    # MEAN-FIELD SCALING: effective temperature = temperature / chi
    effective_temperature = temperature / chi
    print(f"  lr={lr:.6e}, T={temperature:.6f}, chi={chi:.6f}, T_eff={effective_temperature:.6f}")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")
    
    # Generate data: Y = X[:, 0] (first dimension as target)
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].unsqueeze(-1)  # (P, 1)
    
    # Model with MEAN-FIELD initialization
    ens = 5  # ensemble size
    # MEAN-FIELD SCALING: readout variance *= 1/chi => (1/d, 1/(N*chi))
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="linear",
        weight_initialization_variance=(1/d, 1/(N*chi))
    ).to(device)
    
    # Try to load existing checkpoint
    checkpoint_path = run_dir / "checkpoint.pt"
    model_checkpoint = run_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    
    # Try loading full checkpoint first
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        
        # Load training history from JSON files if they exist
        eigenvalues_path = run_dir / "eigenvalues_over_time.json"
        losses_path = run_dir / "losses.json"
        
        if eigenvalues_path.exists():
            with open(eigenvalues_path, "r") as f:
                eigenvalues_over_time = json.load(f)
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
                
    elif model_checkpoint.exists():
        # Fallback to old format
        print(f"Loading model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        
        # Load training state
        eigenvalues_path = run_dir / "eigenvalues_over_time.json"
        losses_path = run_dir / "losses.json"
        
        if eigenvalues_path.exists():
            with open(eigenvalues_path, "r") as f:
                eigenvalues_over_time = json.load(f)
            if eigenvalues_over_time:
                start_epoch = max([int(k) for k in eigenvalues_over_time.keys()])
                print(f"Resuming from epoch {start_epoch}")
        
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
    
    model.train()
    
    # MEAN-FIELD SCALING: weight decay
    # lambda_W0 = d * T_eff
    # lambda_A = N * chi * T_eff (chi factor added)
    wd_W0 = d * effective_temperature
    wd_A = N * chi * effective_temperature  # MF: chi scaling for activation weights
    
    print(f"  Weight decay: wd_W0={wd_W0:.6f}, wd_A={wd_A:.6f}")
    
    # Langevin noise scale (based on effective temperature)
    noise_scale = np.sqrt(2.0 * lr * effective_temperature)
    
    # Large eval set for eigenvalues
    Xinf = torch.randn(3000, d, device=device)
    
    # Compute initial eigenvalues
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                # Move to CPU for eigenvalue computation
                model_cpu = model.cpu()
                Xinf_cpu = Xinf.cpu()
                eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                # Move model back to original device
                model.to(device)
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues: {e}")
                model.to(device)
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        if epoch > 0:
            torch.manual_seed(epoch)
            
            # Forward pass
            output = model(X)  # (P, ens)
            
            # Loss per ensemble
            diff = output - Y  # (P, ens)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ens,)
            loss = per_ensemble_loss.sum()
            
            loss_avg = loss.item() / ens
            loss_std = per_ensemble_loss.std().item()
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('loss/sum_total', loss.item(), epoch)
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Langevin update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    # Weight decay
                    if 'W0' == name:
                        wd = wd_W0
                    elif 'A' == name:
                        wd = wd_A
                    else:
                        wd = 0
                    
                    # Gradient + weight decay + noise
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(noise)
        
        # Logging and checkpointing
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues
                try:
                    # Move to CPU for eigenvalue computation
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    # Move model back to original device
                    model.to(device)
                    
                    # Log to TensorBoard
                    if writer is not None:
                        # Individual eigenvalues
                        eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) 
                                         for i in range(len(eigenvalues))}
                        writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)
                        
                        # Eigenvalue statistics
                        writer.add_scalar('eigenvalues/mean', float(eigenvalues.mean()), epoch)
                        writer.add_scalar('eigenvalues/max', float(eigenvalues.max()), epoch)
                        writer.add_scalar('eigenvalues/min', float(eigenvalues.min()), epoch)
                        writer.add_scalar('eigenvalues/std', float(eigenvalues.std()), epoch)
                        
                        # Eigenvalue histogram
                        writer.add_histogram('eigenvalues/distribution', eigenvalues, epoch)
                        
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                    model.to(device)
                
                # Log loss and eigenvalues
                if epoch > 0:
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
                    if writer is not None:
                        writer.add_scalar('loss/mean', loss_avg, epoch)
                        writer.add_scalar('loss/std', loss_std, epoch)
                    
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}, "
                              f"max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                    else:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")
                else:
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, "
                              f"mean_eig={eigenvalues.mean():.6f}")
                
                # Save checkpoint
                if epoch > 0:
                    # Save model state
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    
                    # Save full checkpoint with metadata
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'd': d, 'P': P, 'N': N, 'ens': ens,
                            'lr': float(lr), 'temperature': float(temperature),
                            'chi': float(chi), 'effective_temperature': float(effective_temperature),
                            'wd_W0': float(wd_W0), 'wd_A': float(wd_A),
                            'readout_variance': float(1/(N*chi)),
                            'scaling': 'mean_field'
                        },
                        'loss': float(loss_avg) if epoch > 0 else None,
                        'loss_std': float(loss_std) if epoch > 0 else None,
                    }
                    if eigenvalues is not None:
                        checkpoint['eigenvalues'] = eigenvalues.tolist()
                    torch.save(checkpoint, run_dir / "checkpoint.pt")
                    
                    # Save eigenvalues and losses
                    with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                        json.dump(eigenvalues_over_time, f, indent=2)
                    
                    with open(run_dir / "losses.json", "w") as f:
                        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, 
        "lr": float(lr), "temperature": float(temperature),
        "chi": float(chi), "effective_temperature": float(temperature / chi),
        "epochs": epochs, "ens": ens,
        "wd_W0": float(wd_W0), "wd_A": float(wd_A),
        "readout_variance": float(1/(N*chi)),
        "scaling": "mean_field"
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-layer linear FCN2 with mean-field scaling")
    parser.add_argument("--d", type=int, default=10, help="Input dimension")
    parser.add_argument("--P", type=int, default=50, help="Number of samples")
    parser.add_argument("--N", type=int, default=200, help="Hidden layer width")
    parser.add_argument("--epochs", type=int, default=2_000_000, help="Number of epochs")
    parser.add_argument("--log-interval", type=int, default=10_000, help="Logging interval")
    parser.add_argument("--device", type=str, default="cpu", help="Device string")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.02, help="Base temperature")
    parser.add_argument("--chi", type=float, default=1.0, help="Chi scaling")
    parser.add_argument("--run-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    train_fcn2_linear(
        d=args.d,
        P=args.P,
        N=args.N,
        epochs=args.epochs,
        log_interval=args.log_interval,
        device_str=args.device,
        lr=args.lr,
        temperature=args.temperature,
        chi=args.chi,
        run_dir=Path(args.run_dir) if args.run_dir else None
    )
