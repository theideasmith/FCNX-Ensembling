#!/usr/bin/env python3
"""
Minimal ensemble size comparison:
- Train networks with d=4, P=7, N=500, kappa=25
- Vary ensemble sizes: 1, 15, 50, 100
- Plot model vs GPR with ensemble std error bars (4 subplots)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear
from GPKit import gpr_dot_product_explicit

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def train_network(d, P, N, kappa, ensemble_size, epochs=100_000):
    """Train network and return model and data for plotting."""
    k = kappa
    chi = 1.0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 1e-7 / P
    temperature = 2 * k
    
    print(f"Training: d={d}, P={P}, N={N}, kappa={kappa:.6e}, ensemble={ensemble_size}, lr={lr:.6e}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Model
    model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=ensemble_size,
                                      weight_initialization_variance=(1/d, 1/N, 1/N)).to(device)
    model.train()
    
    # Weight decay
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    
    for epoch in range(epochs):
        output = model(X)
        diff = output - Y
        loss = torch.sum(diff * diff) 
        
        model.zero_grad()
        loss.backward()
        
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
        
        if epoch % 10_000 == 0:
            print(f"  Epoch {epoch:6d}: loss={loss.item() / ensemble_size:.6e}")
    
    print(f"  Training complete")
    return model, X, Y

def plot_ensemble_comparison():
    """Train networks with different ensemble sizes and plot."""
    d, P, N, kappa = 4, 7, 500, 5.0
    ensemble_sizes = [300, 600, 900]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, ens_size in enumerate(ensemble_sizes):
        model, X, Y = train_network(d, P, N, kappa, ens_size)
        
        # Compute predictions
        with torch.no_grad():
            gpr_pred = gpr_dot_product_explicit(X, Y, X, 2*kappa).cpu().numpy().ravel()
            model_out_full = model(X).detach().cpu().numpy()  # (P, ensemble)
            model_out = model_out_full.mean(axis=-1)
            model_std = model_out_full.std(axis=-1)
        
        # Compute slope
        slope = np.polyfit(gpr_pred, model_out, 1)[0]
        
        # Plot
        ax = axes[idx]
        ax.errorbar(gpr_pred, model_out, yerr=model_std, fmt='o', markersize=8, 
                   elinewidth=1.5, capsize=3, alpha=0.6, label="model ± std")
        
        # y=x line
        mn = min(gpr_pred.min(), model_out.min() - model_std.max())
        mx = max(gpr_pred.max(), model_out.max() + model_std.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label="y = x")
        
        # best-fit line
        slope_val, intercept = np.polyfit(gpr_pred, model_out, 1)
        x_line = np.linspace(mn, mx, 100)
        ax.plot(x_line, slope_val * x_line + intercept, 'r-', linewidth=1.5,
                label=f"best fit (slope={slope_val:.4f})")
        
        ax.set_xlabel("GPR prediction")
        ax.set_ylabel("Model output")
        ax.set_title(f"Ensemble Size = {ens_size}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"d={d}, P={P}, N={N}, κ={kappa}", fontsize=14, y=0.995)
    fig.tight_layout()
    
    out_path = Path(__file__).parent / f"ensemble_size_comparison_kappa_{kappa}.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_ensemble_comparison()
