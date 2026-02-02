#!/usr/bin/env python3
"""
Optimized training script with multi-seed dataset support.

Usage:
    python d_sweep_multi_seed.py --P 1200 --d 100 --N 800 --chi 80 --kappa 0.0125 --lr 3e-5 --epochs 50000000 --device cuda:1 --num_seeds 5 --ens 10
"""

import argparse
import datetime
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

sys.path.append('/home/akiva/FCNX-Ensembling/lib')
torch.set_default_dtype(torch.float32)


class FCN3NetworkErfOptimized(torch.nn.Module):
    """
    Three-layer FCN with ERF activation supporting multiple dataset seeds.
    Each seed has its own ensemble of networks.
    """
    
    def __init__(self, d, n1, n2, P, num_seeds=1, ens=1, weight_initialization_variance=(1.0, 1.0, 1.0), device='cuda'):
        super().__init__()
        
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.num_seeds = num_seeds
        self.ens = ens
        self.ensembles = ens
        self.num_samples = P
        self.device = device
        
        v0, v1, v2 = weight_initialization_variance
        
        # Initialize weights with seed dimension: (num_seeds, ens, ...)
        self.W0 = torch.nn.Parameter(
            torch.randn(num_seeds, ens, n1, d, device=device, dtype=torch.float32) * (v0 ** 0.5),
            requires_grad=True
        )
        self.W1 = torch.nn.Parameter(
            torch.randn(num_seeds, ens, n2, n1, device=device, dtype=torch.float32) * (v1 ** 0.5),
            requires_grad=True
        )
        self.A = torch.nn.Parameter(
            torch.randn(num_seeds, ens, n2, device=device, dtype=torch.float32) * (v2 ** 0.5),
            requires_grad=True
        )
    
    def forward(self, X):
        """
        Forward pass with seed dimension.
        X shape: (num_seeds, P, d)
        Output shape: (num_seeds, P, ens)
        """
        # h0 = erf(W0 @ X)  shape: (num_seeds, P, ens, n1)
        h0 = torch.erf(torch.einsum('sqkl,sul->suqk', self.W0, X))
        
        # h1 = erf(W1 @ h0)  shape: (num_seeds, P, ens, n2)
        h1 = torch.erf(torch.einsum('sqkj,suqj->suqk', self.W1, h0))
        
        # output = A @ h1  shape: (num_seeds, P, ens)
        return torch.einsum('sqk,suqk->suq', self.A, h1)
    
    def H_eig(self, X, Y, std=False):
        """
        Compute H kernel eigenvalues efficiently.
        X shape: (num_seeds, P, d)
        Y shape: (num_seeds, P, 1)
        Returns eigenvalues averaged over seeds.
        """
        with torch.no_grad():
            # Compute h0 activation
            h0 = torch.erf(torch.einsum('sqkl,sul->suqk', self.W0, X))
            
            # Compute h1 preactivation
            h1_pre = torch.einsum('sqkj,suqj->suqk', self.W1, h0)
            
            # Kernel per seed and ensemble: K[s,q,u,v] = sum_m h1_pre[s,u,q,m] * h1_pre[s,v,q,m]
            # Shape: (num_seeds, ens, P, P)
            hh_inf_i = torch.einsum('suqm,svqm->squv', h1_pre, h1_pre) / (self.n1 * X.shape[1])
            
            # Average over ensembles for each seed
            hh_inf = torch.mean(hh_inf_i, dim=1)  # shape: (num_seeds, P, P)
            
            # Average over seeds
            hh_inf = torch.mean(hh_inf, dim=0)  # shape: (P, P)
            
            # Compute eigenvalue
            Y_flat = Y.squeeze()  # shape: (num_seeds, P)
            Y_mean = Y_flat.mean(dim=0)  # Average over seeds: shape (P,)
            
            norm = torch.sum(Y_mean * Y_mean) / X.shape[1]
            
            # Ls = Y^T @ K @ Y
            Ls = torch.einsum('u,uv,v->', Y_mean, hh_inf, Y_mean) / X.shape[1]
            lsT = Ls / norm
            
            if std:
                # Compute std over seeds
                Ls_per_seed = []
                for s in range(self.num_seeds):
                    Y_s = Y_flat[s]
                    norm_s = torch.sum(Y_s * Y_s) / X.shape[1]
                    # Use per-seed kernel averaged over ensembles
                    hh_s = torch.mean(hh_inf_i[s], dim=0)
                    Ls_s = torch.einsum('u,uv,v->', Y_s, hh_s, Y_s) / X.shape[1]
                    Ls_per_seed.append(Ls_s / norm_s)
                
                std_ls = torch.std(torch.stack(Ls_per_seed))
                return lsT, std_ls
            
            return lsT


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


def train_and_track(d, P, N, chi, kappa, lr0, epochs, device_str, storage_dir, eps=0.03, num_seeds=1, ens=50, 
                    base_seed=42, log_interval=10_000):
    """Train network with multiple dataset seeds and track eigenvalues over epochs."""
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    temperature = 2 * kappa / chi
    
    lr = lr0 / P
    
    # Setup directories
    base_name = f"d{d}_P{P}_N{N}_chi{chi}_kappa{kappa}_nseeds{num_seeds}_ens{ens}"
    parent_dir = Path(__file__).parent
    run_dir = parent_dir / storage_dir / base_name
    run_dir.mkdir(exist_ok=True, parents=True)
    seed_dir = run_dir / f"base_seed{base_seed}"
    seed_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize TensorBoard writer
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    human_readable_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=tensorboard_dir / f"{human_readable_timestamp}_base_seed{base_seed}")
    
    print(f"\nTraining: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Number of seeds: {num_seeds}, Ensemble size (per seed): {ens}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Compute theory predictions
    print("\nComputing theory predictions...")
    theory_H = compute_theory(d, P, N, chi, kappa, eps)
    print(f"Theory eigenvalues: lH1T={theory_H.get('lH1T'):.6f}, lH1P={theory_H.get('lH1P'):.6f}, "
          f"lH3T={theory_H.get('lH3T'):.6f}, lH3P={theory_H.get('lH3P'):.6f}")
    
    # Generate data for multiple seeds
    X = torch.zeros(num_seeds, P, d, device=device)
    Y = torch.zeros(num_seeds, P, 1, device=device)
    
    for s in range(num_seeds):
        torch.manual_seed(base_seed + s)
        X_s = torch.randn(P, d, device=device)
        X0_s = X_s[:, 0].unsqueeze(-1)
        Y_s = X0_s + eps * (X0_s**3 - 3 * X0_s)
        
        X[s] = X_s
        Y[s] = Y_s
    
    # Model (use seed 70 for model initialization)
    torch.manual_seed(70)
    model = FCN3NetworkErfOptimized(
        d, N, N, P, num_seeds=num_seeds, ens=ens,
        weight_initialization_variance=(1/d, 1/N, 1/(N * chi))
    ).to(device)
    
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
        
        with open(config_path, "r") as f:
            config = json.load(f)
        start_epoch = config.get("current_epoch", 0)
        lr = config.get("lr", lr)
        print(f"Resuming from epoch {start_epoch}")
        
    model.train()
    
    # Weight decay coefficients
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    
    # Large eval set for eigenvalues
    Xinf = torch.zeros(num_seeds, 3000, d, device=device)
    for s in range(num_seeds):
        torch.manual_seed(base_seed + s + 1000)
        Xinf[s] = torch.randn(3000, d, device=device)
    
    loss = None
    for epoch in range(start_epoch, epochs + 1):
        if epoch > 0:
            torch.manual_seed(7 + epoch)
            
            # Adjust learning rate in final 10%
            if epoch > epochs * 0.9:
                lr = lr0 / (3 * P)
            else:
                lr = lr0 / P
            
            noise_scale = np.sqrt(2.0 * lr * temperature)
            
            # Forward pass
            output = model(X)  # shape: (num_seeds, P, ens)
            
            # Compute loss
            diff = output - Y  # (num_seeds, P, ens)
            per_seed_ensemble_loss = torch.sum(diff * diff, dim=1)  # (num_seeds, ens)
            loss = per_seed_ensemble_loss.sum()
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Langevin update with weight decay and noise
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    if name == 'W0':
                        wd = wd_fc1
                    elif name == 'W1':
                        wd = wd_fc2
                    elif name == 'A':
                        wd = wd_fc3
                    else:
                        wd = 0
                    
                    noise = torch.randn_like(param) * noise_scale
                    
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param)
                    param.add_(noise)
        
        # Logging
        if epoch % log_interval == 0 and epoch > 0:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues of covariance matrix of read-in weights
                try:
                    W0 = model.W0  # shape: (num_seeds, ens, N, d)
                    W0_reshaped = W0.view(num_seeds * ens * N, d)  # shape: (num_seeds*ens*N, d)
                    cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (num_seeds * ens * N)
                    eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy()
                    writer.add_scalar('W0_Cov_Eigenvalues/max', eigvals_W0[0], epoch)
                    writer.add_scalar('W0_Cov_Eigenvalues/mean', eigvals_W0[1:].mean(), epoch)
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute/log W0 covariance eigenvalues at epoch {epoch}: {e}")
    
    # Save final model
    torch.save(model.state_dict(), seed_dir / "model_final.pt")
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, "kappa": float(kappa),
        "lr": float(lr), "epochs": epochs, "chi": chi,
        "base_seed": base_seed, "num_seeds": num_seeds, "ens": ens, 
        "current_epoch": epochs
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save eigenvalues over time
    try:
        with open(seed_dir / "eigenvalues_over_time.json", "w") as f:
            json.dump(eigenvalues_over_time, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not save final eigenvalues: {e}")
    
    # Save losses
    with open(seed_dir / "losses.json", "w") as f:
        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
    
    writer.close()
    
    # Get final eigenvalue
    final_eigenvalue = None
    try:
        if epochs in eigenvalues_over_time:
            final_eigenvalue = eigenvalues_over_time[epochs]
    except Exception as e:
        print(f"  Warning: Could not retrieve final eigenvalue: {e}")
    
    return final_eigenvalue, eigenvalues_over_time, run_dir


def main():
    parser = argparse.ArgumentParser(description='Train networks with multiple dataset seeds')
    parser.add_argument('--P', type=int, required=True, help='Number of data points per seed')
    parser.add_argument('--d', type=int, required=True, help='Input dimension')
    parser.add_argument('--N', type=int, required=True, help='Hidden layer size')
    parser.add_argument('--chi', type=float, required=True, help='Chi parameter')
    parser.add_argument('--kappa', type=float, required=True, help='Kappa parameter')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--device', type=str, required=True, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of dataset seeds')
    parser.add_argument('--ens', type=int, default=50, help='Ensemble size per seed')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed for dataset generation')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick test with epochs=1 and delete results afterwards')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon parameter for cubic target generation')
    parser.add_argument('--to', type=str, default='results', help='Directory to store results')
    args = parser.parse_args()
    
    epochs = 1 if args.dry_run else args.epochs
    
    print(f"\n{'='*60}")
    print(f"Starting training with P={args.P}, d={args.d}, N={args.N}, chi={args.chi}, kappa={args.kappa}, "
          f"lr={args.lr}, epochs={epochs}, num_seeds={args.num_seeds}, ens={args.ens}, base_seed={args.base_seed} on {args.device}")
    if args.dry_run:
        print("DRY RUN MODE: Running with epochs=1 and will delete results afterwards")
    print(f"{'='*60}")
    
    final_eig, eigs_over_time, run_dir = train_and_track(
        d=args.d,
        P=args.P,
        N=args.N,
        chi=args.chi,
        kappa=args.kappa,
        lr0=args.lr,
        storage_dir=args.to,
        epochs=epochs,
        device_str=args.device,
        num_seeds=args.num_seeds,
        ens=args.ens,
        base_seed=args.base_seed,
        eps=args.eps
    )
    
    print(f"\nTraining completed!")
    
    if args.dry_run:
        import shutil
        print(f"Deleting dry-run results from {run_dir}")
        shutil.rmtree(run_dir)
        print("Dry-run cleanup complete.")


if __name__ == "__main__":
    main()