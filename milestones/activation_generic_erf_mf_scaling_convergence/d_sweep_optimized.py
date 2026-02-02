#!/usr/bin/env python3
"""
Optimized training script with streamlined ERF network implementation.

Usage:
    python d_sweep_optimized.py --P 1200 --d 100 --N 800 --chi 80 --kappa 0.0125 --lr 3e-5 --epochs 50000000 --device cuda:1
"""


import argparse
from csv import writer
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
import sys
sys.path.append('/home/akiva/FCNX-Ensembling/lib')  # Adjust this path as needed
# Set default dtype to float32
torch.set_default_dtype(torch.float32)
from FCN3Network import FCN3NetworkErfOptimized

class FCN3NetworkErfOptimized(torch.nn.Module):
    """
    Highly optimized three-layer FCN with ERF activation.
    Eliminates unnecessary function calls and intermediate computations.
    """
    
    def __init__(self, d, n1, n2, P, ens=1, weight_initialization_variance=(1.0, 1.0, 1.0), device='cuda'):
        super().__init__()
        
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.ens = ens
        self.ensembles = ens
        self.num_samples = P
        self.device = device
        
        v0, v1, v2 = weight_initialization_variance
        
        # Initialize weights directly with correct std
        self.W0 = torch.nn.Parameter(
            torch.randn(ens, n1, d, device=device, dtype=torch.float32) * (v0 ** 0.5),
            requires_grad=True
        )
        self.W1 = torch.nn.Parameter(
            torch.randn(ens, n2, n1, device=device, dtype=torch.float32) * (v1 ** 0.5),
            requires_grad=True
        )
        self.A = torch.nn.Parameter(
            torch.randn(ens, n2, device=device, dtype=torch.float32) * (v2 ** 0.5),
            requires_grad=True
        )
    
    def forward(self, X):
        """
        Optimized forward pass: X -> erf(W0@X) -> erf(W1@h0) -> A@h1
        X shape: (P, d)
        Output shape: (P, ens)
        """
        # h0 = erf(W0 @ X)  shape: (P, ens, n1)
        h0 = torch.erf(torch.einsum('qkl,ul->uqk', self.W0, X))
        
        # h1 = erf(W1 @ h0)  shape: (P, ens, n2)
        h1 = torch.erf(torch.einsum('qkj,uqj->uqk', self.W1, h0))
        
        # output = A @ h1  shape: (P, ens)
        return torch.einsum('qk,uqk->uq', self.A, h1)
    
    def H_eig(self, X, Y, std=False):
        """
        Compute H kernel eigenvalues efficiently.
        H kernel is based on h1_preactivation (before final erf).
        """
        with torch.no_grad():
            # Compute h0 activation
            h0 = torch.erf(torch.einsum('qkl,ul->uqk', self.W0, X))
            
            # Compute h1 preactivation (no final erf)
            h1_pre = torch.einsum('qkj,uqj->uqk', self.W1, h0)
            
            # Kernel per ensemble: K_i[u,v] = sum_m h1_pre[u,i,m] * h1_pre[v,i,m]
            # Shape: (ens, P, P)
            hh_inf_i = torch.einsum('uqm,vqm->quv', h1_pre, h1_pre) / (self.n1 * X.shape[0])
            
            # Average over ensembles
            hh_inf = torch.mean(hh_inf_i, dim=0)
            
            # Compute eigenvalue: lambda = Y^T K Y / (Y^T Y)
            Y_flat = Y.squeeze()
            norm = torch.sum(Y_flat * Y_flat, dim=0) / X.shape[0]
            
            # Ls = Y^T @ K @ Y
            Ls = torch.einsum('ul,uv,vl->l', Y_flat, hh_inf, Y_flat) / X.shape[0]
            lsT = Ls / norm
            
            if std:
                Ls_i = torch.einsum('ul,quv,vl->ql', Y_flat, hh_inf_i, Y_flat) / X.shape[0]
                std_ls = torch.std(Ls_i / norm, dim=0)
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


def train_and_track(d, P, N, chi, kappa, lr0, epochs, device_str, eps=0.03, seed=42, ens=50, log_interval=10_000):
    """Train network and track eigenvalues over epochs."""
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    temperature = 2 * kappa / chi
    
    # Normalize lr by dataset size
    lr = lr0 / P
    
    # Setup directories
    base_name = f"d{d}_P{P}_N{N}_chi{chi}_kappa{kappa}"
    parent_dir = Path(__file__).parent
    run_dir = parent_dir / base_name
    run_dir.mkdir(exist_ok=True, parents=True)
    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize TensorBoard writer
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    human_readable_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=tensorboard_dir / f"{human_readable_timestamp}_seed{seed}")
    
    print(f"\nTraining: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Compute theory predictions
    print("\nComputing theory predictions...")
    theory_H = compute_theory(d, P, N, chi, kappa, eps)
    print(f"Theory eigenvalues: lH1T={theory_H.get('lH1T'):.6f}, lH1P={theory_H.get('lH1P'):.6f}, "
          f"lH3T={theory_H.get('lH3T'):.6f}, lH3P={theory_H.get('lH3P'):.6f}")
    
    # Data
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=device)
    X0 = X[:, 0].unsqueeze(-1)
    Y = X0 + eps * (X0**3 - 3 * X0)
    
    # Model (use seed 70 for model initialization)
    torch.manual_seed(70)
    model = FCN3NetworkErfOptimized(
        d, N, N, P, ens=ens,
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
        
        # Load config and resume
        with open(config_path, "r") as f:
            config = json.load(f)
        start_epoch = config.get("current_epoch", 0)
        lr = config.get("lr", lr)
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
    
    model.train()
    
    # Weight decay coefficients
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    Xinf = torch.randn(3000, d, device=device)  # large eval set for eigenvalues
    
    # Compute initial eigenvalues at epoch 0 if starting fresh
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues[1:].mean():.6f}")
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues at epoch 0: {e}")
    
    loss = None
    for epoch in range(start_epoch, epochs + 1):
        # Forward pass (skip for epoch 0)
        if epoch > 0:
            torch.manual_seed(7 + epoch)  # Langevin dynamics seed
            
            # Adjust learning rate in final 10%
            if epoch > epochs * 0.9:
                lr = lr0 / (3 * P)
            else:
                lr = lr0 / P
            
            noise_scale = np.sqrt(2.0 * lr * temperature)
            
            # Forward pass
            output = model(X)  # shape: (P, ensemble)
            
            # Compute per-ensemble losses
            diff = output - Y  # (P, ensemble)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ensemble,)
            loss = per_ensemble_loss.sum()
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Pure Langevin update with weight decay and noise
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    # Set weight decay based on parameter
                    if name == 'W0':
                        wd = wd_fc1
                    elif name == 'W1':
                        wd = wd_fc2
                    elif name == 'A':
                        wd = wd_fc3
                    else:
                        wd = 0
                    
                    # Generate noise
                    noise = torch.randn_like(param) * noise_scale
                    
                    # Langevin update: param -= lr * (grad + wd * param) + noise
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param)
                    param.add_(noise)
        
        # Logging and eigenvalue computation
        log_interval = 10000
        if epoch % log_interval == 0 and epoch > 0:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % log_interval == 0:
            with torch.no_grad():    
                # Compute eigenvalues of covariance matrix of read-in weights
                try:
                    W0 = model.W0  # shape: (ens, N, d)
                    W0_reshaped = W0.view(model.ensembles * N, d)  # shape: (ens*N, d)
                    cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * N)  # shape: (d, d)
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
        "seed": seed, "ens": ens, "current_epoch": epochs
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
    
    # Close TensorBoard writer
    writer.close()
    
    # Get final eigenvalues
    final_eigenvalues = None
    try:
        if epochs in eigenvalues_over_time:
            final_eigenvalues = np.array(eigenvalues_over_time[epochs])
    except Exception as e:
        print(f"  Warning: Could not retrieve final eigenvalues: {e}")
    
    return final_eigenvalues, eigenvalues_over_time, run_dir


def main():
    parser = argparse.ArgumentParser(description='Train a single network with specified parameters')
    parser.add_argument('--P', type=int, required=True, help='Number of data points')
    parser.add_argument('--d', type=int, required=True, help='Input dimension')
    parser.add_argument('--N', type=int, required=True, help='Hidden layer size')
    parser.add_argument('--chi', type=int, required=True, help='Chi parameter')
    parser.add_argument('--kappa', type=float, required=True, help='Kappa parameter')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--device', type=str, required=True, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset generation')
    parser.add_argument('--ens', type=int, default=50, help='Ensemble size')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick test with epochs=1 and delete results afterwards')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon parameter for cubic target generation')
    args = parser.parse_args()
    
    epochs = 1 if args.dry_run else args.epochs
    
    print(f"\n{'='*60}")
    print(f"Starting training with P={args.P}, d={args.d}, N={args.N}, chi={args.chi}, kappa={args.kappa}, "
          f"lr={args.lr}, epochs={epochs}, seed={args.seed}, ens={args.ens} on {args.device}")
    if args.dry_run:
        print("DRY RUN MODE: Running with epochs=1 and will delete results afterwards")
    print(f"{'='*60}")
    
    final_eigs, eigs_over_time, run_dir = train_and_track(
        d=args.d,
        P=args.P,
        N=args.N,
        chi=args.chi,
        kappa=args.kappa,
        lr0=args.lr,
        epochs=epochs,
        device_str=args.device,
        seed=args.seed,
        ens=args.ens,
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