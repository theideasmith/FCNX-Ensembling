#!/usr/bin/env python3
"""
Minimal GPR convergence study: train networks at different widths
and compare empirical H eigenvalues with GPR on P samples.

Usage: python train_gpr_convergence.py P N d
  P: number of training samples
  N: hidden width (N1=N2=N)
  d: input dimension
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

# Assume these are in lib/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear
from GPKit import gpr_dot_product_explicit

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def plot_gpr_vs_model(gpr_pred, model_out, epoch, out_path):
    """Plot GPR predictions vs model outputs with y=x and best-fit line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(gpr_pred, model_out, s=20, alpha=0.6, label="model vs GPR")
    
    # y=x line
    mn = min(gpr_pred.min(), model_out.min())
    mx = max(gpr_pred.max(), model_out.max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label="y = x")
    
    # best-fit line
    slope, intercept = np.polyfit(gpr_pred, model_out, 1)
    x_line = np.linspace(mn, mx, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5,
            label=f"best fit (slope={slope:.4f})")
    
    ax.set_xlabel("GPR prediction")
    ax.set_ylabel("Model output")
    ax.set_title(f"GPR vs Model (epoch {epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    
    return slope

def main():
    if len(sys.argv) != 4 + 1:
        print("Usage: python train_gpr_convergence.py P N d k chi")
        sys.exit(1)
    
    P = int(sys.argv[1])
    N = int(sys.argv[2])
    d = int(sys.argv[3])
    k = 0.01/d**2
    chi = float(sys.argv[4])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 1e-7 / P
    epochs = 50_000
    temperature = 2 * k

    # Setup
    exp_dir = Path(f"/home/akiva/FCNX-Ensembling/milestones/gpr_convergence_linear/P{P}_N{N}_d{d}_k{k}_chi{chi}_kappa1overd2")
  
    exp_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = exp_dir / "runs"
    tb_dir.mkdir(exist_ok=True)
      # Save config 
    with open(exp_dir / "config.txt", "w") as f:
        f.write(f"P={P}\n")
        f.write(f"N={N}\n")
        f.write(f"d={d}\n")
        f.write(f"k={k}\n")
        f.write(f"chi={chi}\n")
        f.write(f"lr={lr}\n")
        f.write(f"epochs={epochs}\n")
        f.close()
    writer = SummaryWriter(str(tb_dir))
    
    print(f"Training: P={P}, N={N}, d={d}, χ={chi}, lr={lr}, kappa={k}, epochs={epochs}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:,0].squeeze(-1).unsqueeze(-1)

    X_large = torch.randn(400, d, device=device)
    
    # GPR setup with dot product kernel (sigma_0_sq=1.0 for normalization)
    sigma_0_sq = 2 * k
    
    # Model
    model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=10, weight_initialization_variance=(1/d, 1/N, 1/N)).to(device)
    model.train()
    # Weight decay lambda_i = temperature * fan_in
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop with pure Langevin dynamics
    noise_scale = np.sqrt(2.0 * lr * temperature)
    
    for epoch in range(epochs):
        # Forward pass
        output = model(X)

        loss = custom_mse_loss(output, Y)

        # Backward
        model.zero_grad()
        loss.backward()
        
        # Pure Langevin update: theta -= lr * (grad + lambda * theta) + sqrt(2*lr*T) * xi
        with torch.no_grad():
            for name, param in model.named_parameters():

                if param.grad is None:
                    print("Grad is none")
                    continue
                if 'W0' == name:
                    wd = wd_fc1
                elif 'W1' == name:
                    wd = wd_fc2
                elif 'A' == name:
                    wd = wd_fc3

                
                # Langevin update: theta -= lr * grad - lr * lambda * theta + sqrt(2*lr*T) * xi
                noise = torch.randn_like(param) * noise_scale
                param.add_(-lr * param.grad)
                param.add_(-lr * wd * param.data)
                param.add_(noise)
        
        if epoch % 10_000 == 0:
            writer.add_scalar("loss", loss.item(), epoch)
            torch.save(model.state_dict(), exp_dir / "model.pt")

            # Compute H eigenvalues and slope
            with torch.no_grad():

                h_eigs = model.H_eig(X_large, X_large)
                h_eigs_np = h_eigs.cpu().numpy()
                
                # Compute slope between GPR and model outputs using gpr_dot_product_explicit
                gpr_pred = gpr_dot_product_explicit(X, Y, X, sigma_0_sq).cpu().numpy().ravel()
                model_out = model(X).mean(axis=-1).detach().cpu().numpy().flatten()
                print(gpr_pred.shape, model_out.shape)
                slope = plot_gpr_vs_model(gpr_pred, model_out, epoch, exp_dir / "gpr_vs_model.png")
            
            # Put the eigenvalues as different lines in the same plot:
            writer.add_scalars("h_eigenvalues", {f"eig_{i}": val for i, val in enumerate(h_eigs_np)}, epoch)
            writer.add_scalar("gpr_model_slope", slope, epoch)
            print(f"Epoch {epoch}/{epochs}: loss={loss.item():.6f}, H_eig mean={h_eigs_np.mean():.6f}, slope={slope:.6f}")
    try: 
        # Final H eigenvalues
        with torch.no_grad():
            # X = X.squeeze()
            h_eigs_final = model.H_eig(X, X).cpu().numpy()
            print(f"Final H_eig: mean={h_eigs_final.mean():.6f}, std={h_eigs_final.std():.6f}")
            print(f"GPR-Model slope: {slope:.6f}")
    
    except Exception as e:
        print("Error computing final H eigenvalues:", e)
        h_eigs_final = np.array([np.nan])
    
    try:
        # Final slope
        with torch.no_grad():
            gpr_pred = gpr_dot_product_explicit(X, Y, X, sigma_0_sq)
            model_out = model(X).detach().cpu().numpy().flatten()
            slope = np.polyfit(gpr_pred.cpu().numpy().ravel(), model_out, 1)[0]
    except Exception as e:
        print("Error computing final slope:", e)
        slope = np.nan

    try: 
        writer.add_scalar("h_eigenvalue_mean", h_eigs_final.mean(), 0)
        writer.add_scalar("h_eigenvalue_std", h_eigs_final.std(), 0)
        writer.close()

    except Exception as e:
        print("Error logging final H eigenvalues:", e)
    
    try:
        torch.save(model.state_dict(), exp_dir / "model.pt")
    except Exception as e:
        print("Error saving model:", e)
    print(f"Saved to {exp_dir}")

if __name__ == "__main__":
    main()
