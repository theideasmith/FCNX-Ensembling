#!/usr/bin/env python3
"""
Arcsin kernel slope/loss sweep with erf network and Hermite-3 targets.
Replicates minimal_slope_sweep but uses:
- FCN3NetworkEnsembleErf
- Target: y = x0 + eps * He3(x0) with eps=0.03
- Kernel: arcsin J = 2/pi * arcsin( 2 <x_i, x_j> / ((1+2||x_i||^2)(1+2||x_j||^2)) )
- Sweep kappa over [0.001, 1.0, 10.0]
- Ensemble size = 50
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Defaults
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleErf

def hermite3(x):
    return (x**3 - 3.0 * x) / np.sqrt(6.0)

def arcsin_kernel(X):
    """Compute arcsin kernel matrix as provided in the snippet."""
    d = X.shape[1]
    XXT = torch.einsum('ui,vi->uv', X, X) 
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    return (2 / torch.pi) * torch.arcsin(arg)

def arcsin_gpr_predict(X, Y, sigma0_sq):
    """Simple GP prediction with arcsin kernel on training data."""
    J = arcsin_kernel(X)
    n = J.shape[0]
    K = J + sigma0_sq * torch.eye(n, device=X.device)
    alpha = torch.linalg.solve(K, Y)
    return J @ alpha

from torch import nn
criterion = nn.MSELoss(reduction='sum')

def custom_mse_loss(outputs, targets):

    return criterion(outputs, targets)

def plot_gpr_vs_model(gpr_pred, model_out, model_std, epoch, out_path, d, P, N, kappa):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Ensure 1D vectors for plotting
    gpr_flat = np.asarray(gpr_pred).ravel()
    model_flat = np.asarray(model_out).ravel()
    std_flat = np.asarray(model_std).ravel()
    ax.errorbar(gpr_flat, model_flat, yerr=std_flat, fmt='o', markersize=6, alpha=0.6,
                elinewidth=1.2, capsize=2.5, label="model ± std")
    mn = min(gpr_flat.min(), (model_flat - std_flat).min())
    mx = max(gpr_flat.max(), (model_flat + std_flat).max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.4, label="y = x")
    slope, intercept = np.polyfit(gpr_flat, model_flat, 1)
    x_line = np.linspace(mn, mx, 200)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.4,
            label=f"best fit (slope={slope:.4f})")
    ax.set_xlabel("GPR prediction")
    ax.set_ylabel("Model output")
    ax.set_title(f"GPR vs Model (epoch {epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"d={d}, P={P}, N={N}, κ={kappa}", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return slope

def train_and_track(d, P, N, kappa, eps=0.03, ens=50, epochs=50_000, log_interval=10_000):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 5e-5 
    temperature = 2 * kappa
    chi = 1.0
    run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_k{kappa:.3g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    x0 = X[:, 0]
    Y = x0 + eps * hermite3(x0)
    sigma0_sq = 2 * kappa

    # One-time sanity plot: GPR predictor vs target as a function of x0
    with torch.no_grad():
        gpr_init = arcsin_gpr_predict(X, Y, sigma0_sq).cpu().numpy().ravel()
        x0_np = x0.cpu().numpy().ravel()
        Y_np = Y.cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x0_np, Y_np, s=25, alpha=0.6, label="target")
    ax.scatter(x0_np, gpr_init, s=25, alpha=0.6, label="GPR pred")
    ax.set_xlabel("x0")
    ax.set_ylabel("y")
    ax.set_title(f"Initial GPR vs target (d={d}, P={P}, κ={kappa})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(run_dir / "gpr_vs_target_initial.png"), dpi=150)
    plt.close(fig)
    model = FCN3NetworkEnsembleErf(d, N, N, P, ens=ens,
                                   weight_initialization_variance=(1/d, 1/N, 1/N),
                                   device=device).to(device)
    model.train()
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    noise_scale = np.sqrt(2.0 * lr * temperature)
    slopes = {}
    losses = {}
    loss_stds = {}
    for epoch in range(epochs):
        output_full = model.forward_no_unsqueeze(X).squeeze(-1)  # (P, ens)

        diff = output_full - Y
        per_ens_loss = torch.sum(diff * diff, dim=0)  # (ens,)
        loss = per_ens_loss.sum()
        loss_avg = loss.item() / model.ens
        loss_std = per_ens_loss.std().item()
        model.zero_grad()
        loss_avg = custom_mse_loss(output_full.mean(dim=1), Y)
        loss = loss_avg
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: param {name} has no grad")
                    continue
                if 'W0' == name:
                    wd = wd_fc1
                elif 'W1' == name:
                    wd = wd_fc2
                elif 'A' == name:
                    wd = wd_fc3
                else:
                    print("Unknown parameter name:", name)
                    wd = 0.0

                noise = torch.randn_like(param) * noise_scale
                param.add_(-lr * param.grad)
                param.add_(-lr * wd * param.data)
                param.add_(noise)
        if epoch % log_interval == 0:
            with torch.no_grad():
                gpr_pred = arcsin_gpr_predict(X, Y, sigma0_sq).cpu().numpy().ravel()
                model_out_full = model.forward_no_unsqueeze(X).detach().cpu().numpy()
                model_out = model_out_full.mean(axis=-1).ravel()
                model_std = model_out_full.std(axis=-1).ravel()
                slope = plot_gpr_vs_model(gpr_pred, model_out, model_std, epoch,
                                          run_dir / "gpr_vs_model.png", d, P, N, kappa)
                slopes[epoch] = float(slope)
                losses[epoch] = float(loss_avg)
                loss_stds[epoch] = float(loss_std)

                # Overwrite x-vs-target/model plot for quick visual
                x0_np = x0.cpu().numpy().ravel()
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.scatter(x0_np, Y.cpu().numpy().ravel(), s=18, alpha=0.6, label="target")
                ax2.scatter(x0_np, model_out, s=18, alpha=0.6, label="model")
                ax2.set_xlabel("x0")
                ax2.set_ylabel("y")
                ax2.set_title(f"x0 vs target/model (epoch {epoch})")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                fig2.tight_layout()
                fig2.savefig(str(run_dir / "x_vs_target_model.png"), dpi=150)
                plt.close(fig2)
                torch.save(model.state_dict(), run_dir / f"model.pt")
                print(f"Epoch {epoch:6d}: loss_avg={loss_avg:.6e}, loss_std={loss_std:.6e}, slope={slope:.6f}")
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    # Final metrics
    with torch.no_grad():
        gpr_pred_final = arcsin_gpr_predict(X, Y, sigma0_sq).cpu().numpy().ravel()
        model_out_full = model.forward_no_unsqueeze(X).detach().cpu().numpy()
        model_out_final = model_out_full.mean(axis=-1).ravel()
        model_std_final = model_out_full.std(axis=-1).ravel()
        final_slope = np.polyfit(gpr_pred_final, model_out_final, 1)[0]
        final_loss_avg = float(custom_mse_loss(torch.tensor(model_out_final), torch.tensor(gpr_pred_final)).item())
    config = {"d": d, "P": P, "N": N, "kappa": float(kappa), "lr": float(lr),
              "epochs": epochs, "chi": chi, "eps": eps, "ensemble": ens}
    metrics = {"final_slope": float(final_slope), "final_loss": final_loss_avg}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return slopes, losses, loss_stds

def main():
    dims = [5]
    kappas = [0.2]
    N = 512
    epochs = 30000
    log_interval = 1_000
    eps = 0.03
    ens = 1
    all_slopes = {}
    all_losses = {}
    all_loss_stds = {}
    for d in dims:
        P =2*d**3
        all_slopes[d] = {}
        all_losses[d] = {}
        all_loss_stds[d] = {}
        for kappa in kappas:
            key = f"k_{kappa:.3g}"
            slopes, losses, loss_stds = train_and_track(d, P, N, kappa, eps=eps, ens=ens,
                                                        epochs=epochs, log_interval=log_interval)
            all_slopes[d][key] = slopes
            all_losses[d][key] = losses
            all_loss_stds[d][key] = loss_stds
    # Save summaries
    out_dir = Path(__file__).parent
    with open(out_dir / "slopes_data.json", "w") as f:
        json.dump(all_slopes, f, indent=2)
    with open(out_dir / "losses_data.json", "w") as f:
        json.dump(all_losses, f, indent=2)
    with open(out_dir / "loss_stds_data.json", "w") as f:
        json.dump(all_loss_stds, f, indent=2)
    # Plots
    colors = ['b', 'g', 'r']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, d in enumerate(dims):
        ax = axes[idx]
        for color, kappa in zip(colors, kappas):
            key = f"k_{kappa:.3g}"
            epochs_list = sorted(all_slopes[d][key].keys())
            slope_vals = [all_slopes[d][key][e] for e in epochs_list]
            ax.plot(epochs_list, slope_vals, marker='o', color=color, linewidth=2,
                    label=f"κ={kappa}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Slope (Model vs GP)")
        ax.set_title(f"d={d}, P={d+3}, N={N}, ens={ens}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "slopes_comparison.png"), dpi=150)
    plt.close(fig)
    # Loss plot with std error bars
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for idx, d in enumerate(dims):
        ax = axes2[idx]
        for color, kappa in zip(colors, kappas):
            key = f"k_{kappa:.3g}"
            epochs_list = sorted(all_losses[d][key].keys())
            loss_vals = [all_losses[d][key][e] for e in epochs_list]
            loss_std_vals = [all_loss_stds[d][key][e] for e in epochs_list]
            ax.errorbar(epochs_list, loss_vals, yerr=loss_std_vals, marker='o', color=color,
                        linewidth=2, capsize=3, capthick=1, alpha=0.8, label=f"κ={kappa}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Ensemble mean ± std)")
        ax.set_yscale('log')
        ax.set_title(f"d={d}, P={d+3}, N={N}, ens={ens}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(out_dir / "losses_comparison.png"), dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    main()
