#!/usr/bin/env python3
"""
Train a single network with specified parameters.

Usage:
    python d_sweep_hermite.py --P 1200 --d 100 --N 800 --chi 80 --kappa 0.0125 --lr 3e-5 --epochs 50000000 --device cuda:1
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
import subprocess
import tempfile
from typing import Dict, Optional
import traceback
# Set default dtype to float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def compute_theory(d: int, P: int, N: int, chi: float, kappa: float, eps: float) -> Dict[str, Optional[float]]:
    """Get theoretical predictions by calling Julia eos_fcn3erf.jl and reading JSON output."""
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--n2={N}",
        f"--chi={chi}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={to_path}",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(to_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Julia theory solver failed: {e}")
        data = {}
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass

    tgt = data.get("target", {}) if isinstance(data, dict) else {}
    perp = data.get("perpendicular", {}) if isinstance(data, dict) else {}

    return {
        "lH1T": tgt.get("lH1T"),
        "lH1P": perp.get("lH1P"),
        "lH3T": tgt.get("lH3T"),
        "lH3P": perp.get("lH3P"),
    }

def train_and_track(d, P, N, chi, kappa, lr0, epochs, device_str, eps = 0.03, seed=42, ens=50, log_interval=10_000, to='results'):
    """Train network and track eigenvalues over epochs."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    temperature = 2 * kappa / chi

    # Normalize lr by dataset size

    lr = lr0 / P

    # Setup directories
    base_name = f"d{d}_P{P}_N{N}_chi{chi}_kappa{kappa}"
    parent_dir = Path(__file__).parent
    run_dir = parent_dir /  to / base_name
    run_dir.mkdir(exist_ok=True, parents=True)
    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(exist_ok=True, parents=True)

    # Initialize TensorBoard writer
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=tensorboard_dir / f"seed{seed}")

    print(f"\nTraining: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Compute theory predictions
    # print("\nComputing theory predictions...")
    # theory_H = compute_theory(d, P, N, chi, kappa, eps)
    # print(f"Theory eigenvalues: lH1T={theory_H.get('lH1T'):.6f}, lH1P={theory_H.get('lH1P'):.6f}, lH3T={theory_H.get('lH3T'):.6f}, lH3P={theory_H.get('lH3P'):.6f}")
    
    # Data
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=device)
    X0 = X[:, 0].squeeze(-1).unsqueeze(-1)
    Y = X0 + eps * (X0**3 - 3 * X0)

    # Model (use seed 70 for model initialization)
    torch.manual_seed(70)
    model = FCN3NetworkActivationGeneric(d, N, N, P, ens=ens,
                                         activation="hermite3",
                                         weight_initialization_variance=(1/d, 1/N, 1/(N * chi))).to(device)
    
    # Check if resuming from checkpoint
    model_checkpoint = seed_dir / "model.pt"
    if not model_checkpoint.exists():
        model_checkpoint = seed_dir / "model_final.pt"
    config_path = seed_dir / "config.json"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    if model_checkpoint.exists() and config_path.exists():
        print(f"Loading existing model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        # Load config and resume
        with open(config_path, "r") as f:
            config = json.load(f)
        start_epoch = config.get("current_epoch", 0)
        lr = config.get("lr", lr)  # Load current lr if saved
        print(f"Resuming from epoch {start_epoch}")

        # Load existing logs
        eigenvalues_path = seed_dir / "eigenvalues_over_time.json"
        losses_path = seed_dir / "losses.json"
        if eigenvalues_path.exists():
            with open(eigenvalues_path, "r") as f:
                eigenvalues_over_time = json.load(f)
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
    else:
        # Fresh start
        pass
    
    model.train()
    W0 = model.W0  # shape: (ens, N, d)
    W0_reshaped = W0.view(model.ensembles * N, d)  # shape: (ens*N, d)
    cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * N)  # shape: (d, d)
    eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.detach().cpu().numpy()  # shape: (d,)
    writer.add_scalar('W0_Cov_Eigenvalues/max', eigvals_W0[0], 0)
    writer.add_scalar('W0_Cov_Eigenvalues/mean', eigvals_W0[1:].mean(), 0)
    # Weight decay
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    Xinf = torch.randn(3000, d, device=device)  # large eval set for eigenvalues
    
    # Compute initial eigenvalues at epoch 0 if starting fresh (or forking)
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues[1:].mean():.6f}")
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues at epoch 0: {e}")
    transition_epoch = int(epochs * 0.9)
    epochs = int(epochs * 0.9 + epochs * 0.1 * 3)  # Extend total epochs to allow for post-transition training
    for epoch in range(start_epoch, epochs + 1):  # Resume from start_epoch
        # Forward pass (skip for epoch 0)
        if epoch > 0:
            torch.manual_seed(7 + epoch)  # Langevin dynamics seed

            if epoch > transition_epoch:
                lr = lr0 / ( 3 * P)
            else: 
                lr = lr0 / P

            noise_scale = np.sqrt(2.0 * lr * temperature)
   
            output = model(X)  # shape: (P, ensemble)
            # Compute per-ensemble losses
            diff = output - Y  # (P, ensemble)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ensemble,)
            loss = per_ensemble_loss.sum()
            
            # Compute ensemble-averaged and std loss
            loss_avg = loss.item() / model.ensembles
            loss_std = per_ensemble_loss.std().item()
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Pure Langevin update
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
                    
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(noise)
                        # TensorBoard logging
        if epoch % log_interval ==0 and epoch > 0:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        # Logging and eigenvalue computation
        log_interval = 5000
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues
                try:
                    
                    eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                    writer.add_scalar('Eigenvalues/max', eigenvalues.max(), epoch)
                    writer.add_scalar('Eigenvalues/mean', eigenvalues[1:].mean(), epoch)
                    # try:
                        # eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    # except Exception as e:
                        # print(f"  Warning: Could not store eigenvalues at epoch {epoch}: {e}")
                    
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                
                  
                # Compute eigenvalues of covariance matrix of readin weights and log to tensorboard
                # Along ens*N dimension, and choose the d=0, and the average over the remainder
                try:
                    W0 = model.W0  # shape: (ens, N, d)
                    W0_reshaped = W0.view(model.ensembles * N, d)  # shape: (ens*N, d)
                    cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * N)  # shape: (d, d)
                    eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy()  # shape: (d,)
                    writer.add_scalar('W0_Cov_Eigenvalues/max', eigvals_W0[0], epoch)
                    writer.add_scalar('W0_Cov_Eigenvalues/mean', eigvals_W0[1:].mean(), epoch)
                except Exception as e:
                    print(f"  Warning: Could not compute W0 covariance at epoch {epoch}: {e}")

            # Save progress
            config = {
                "current_epoch": epoch,
                "lr": lr,
                "d": d,
                "P": P,
                "N": N,
                "chi": chi,
                "kappa": kappa,
                "eps": eps,
                "seed": seed,
                "ens": ens,
                "activation": "hermite3",
                "noise_scale": float(noise_scale),
                "loss_avg": loss_avg if epoch > 0 else None,
                "loss_std": loss_std if epoch > 0 else None,
            }
            
            with open(seed_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            torch.save(model.state_dict(), seed_dir / "model.pt")
            
            # Save losses
            if epoch > 0:
                losses[epoch] = loss_avg
                loss_stds[epoch] = loss_std
                with open(seed_dir / "losses.json", "w") as f:
                    json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
            
            # Save eigenvalues
            if eigenvalues is not None:
                try:
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    with open(seed_dir / "eigenvalues_over_time.json", "w") as f:
                        json.dump(eigenvalues_over_time, f, indent=2)
                except Exception as e:
                    print(f"  Warning: Could not save eigenvalues at epoch {epoch}: {e}")

            print(f"  Epoch {epoch:7d}: loss={loss_avg:.6f}, std={loss_std:.6f}")
    
    writer.close()
    return model, eigenvalues_over_time, losses, loss_stds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single network and track eigenvalues over time.")
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--chi", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ens", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10_000)
    parser.add_argument("--to", type=str, default="results")

    args = parser.parse_args()

    train_and_track(
        args.d, args.P, args.N, args.chi, args.kappa,
        args.lr, args.epochs, args.device,
        eps=args.eps, seed=args.seed, ens=args.ens,
        log_interval=args.log_interval, to=args.to
    )