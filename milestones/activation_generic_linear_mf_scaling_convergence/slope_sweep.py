#!/usr/bin/env python3
"""
Linear mean-field scaling convergence study:
- Train networks with d=2, P=6
- chi = N (changed from chi=1)
- Train with kappa = 1/6 and 2
- Train for 2 million epochs
- Track slope across epochs
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric
from GPKit import gpr_dot_product_explicit

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def plot_Y_vs_model(Y, model_out, model_std, epoch, out_path, d, P, N, kappa):
    """Plot Y vs model outputs with error bars and best-fit line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter with error bars showing ensemble std of model output
    ax.errorbar(Y, model_out, yerr=model_std, fmt='o', markersize=6, alpha=0.5, 
                elinewidth=1, capsize=2, label="model ± std (ensemble)")
    
    # y=x line
    mn = min(Y.min(), model_out.min() - model_std.max())
    mx = max(Y.max(), model_out.max() + model_std.max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label="y = x")
    
    # best-fit line
    slope, intercept = np.polyfit(Y, model_out, 1)
    x_line = np.linspace(mn, mx, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5,
            label=f"best fit (slope={slope:.4f})")
    
    ax.set_xlabel("Target")
    ax.set_ylabel("Model output")
    ax.set_title(f"Target vs Model (epoch {epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add supertitle with hyperparameters
    fig.suptitle(f"d={d}, P={P}, N={N}, κ={kappa:.6e}", fontsize=12, y=0.98)
    
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    
    return slope

def train_and_track(d, P, N, kappa, epochs=2_000_000, log_interval=50_000):
    """Train network and return list of slopes across epochs."""
    k = kappa
    chi = N  # Changed from chi=1.0 to chi=N
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 1e-5
    temperature = 2 * k / chi
    
    # Setup output directory for this run
    run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_k{kappa:.6e}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Training: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Model
    model = FCN3NetworkActivationGeneric(d, N, N, P, ens=5,
                                         activation="linear",
                                         weight_initialization_variance=(1/d, 1/N, 1/(N*chi))).to(device)
    model.train()
    
    # Weight decay
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    slopes = {}  # epoch -> slope
    losses = {}  # epoch -> loss averaged over ensemble
    loss_stds = {}  # epoch -> loss std over ensemble
    
    for epoch in range(epochs):
        # Forward pass
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
        
        # Logging
        if epoch % log_interval == 0:
            with torch.no_grad():
                # gpr_pred = gpr_dot_product_explicit(X, Y, X, 2*k).cpu().numpy().ravel()
                # Get full model output (P, ensemble)
                model_out_full = model(X).detach().cpu().numpy()  # shape: (P, ensemble)
                model_out = model_out_full.mean(axis=-1)  # mean over ensemble
                model_std = model_out_full.std(axis=-1)  # std over ensemble
                slope = plot_Y_vs_model(Y.detach().cpu().numpy().ravel(), model_out, model_std, epoch, 
                                         run_dir / "Y_vs_model.png", d, P, N, kappa)
                slopes[epoch] = float(slope)
                losses[epoch] = float(loss_avg)
                loss_stds[epoch] = float(loss_std)
                
                # Save model checkpoint
                torch.save(model.state_dict(), run_dir / f"model.pt")
                
                print(f"  Epoch {epoch:7d}: loss_avg={loss_avg:.6e}, loss_std={loss_std:.6e}, slope={slope:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Compute final metrics
    with torch.no_grad():
        # Final loss (ensemble-averaged)
        final_output = model(X)
        final_loss_sum = custom_mse_loss(final_output, Y).item()
        final_loss_avg = final_loss_sum / model.ensembles
        
        # Final slope with ensemble std
        gpr_pred_final = gpr_dot_product_explicit(X, Y, X, 2*k).cpu().numpy().ravel()
        model_out_full_final = model(X).detach().cpu().numpy()  # shape: (P, ensemble)
        model_out_final = model_out_full_final.mean(axis=-1)
        model_std_final = model_out_full_final.std(axis=-1)
        final_slope = np.polyfit(gpr_pred_final, model_out_final, 1)[0]
        
        # Average discrepancy (mean absolute difference between model and GPR)
        avg_discrepancy = np.mean(np.abs(model_out_final - gpr_pred_final))
        
        # Normalized loss term: (sum_reduction_loss / ensembles) / P * chi / kappa
        normalized_loss = final_loss_avg / P * chi / kappa
    
    final_metrics = {
        "final_loss_avg": float(final_loss_avg),
        "final_slope": float(final_slope),
        "avg_discrepancy": float(avg_discrepancy),
        "normalized_loss": float(normalized_loss),
        "kappa": float(kappa),
        "d": d,
        "P": P,
        "N": N,
        "chi": chi
    }
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, "kappa": float(kappa), 
        "lr": float(lr), "epochs": epochs, "chi": chi
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save final metrics
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Compute and plot eigenvalues using H_eig
    with torch.no_grad():
        try:
            eigenvalues = model.H_eig(X, X).cpu().numpy()
            
            # Create bar plot of eigenvalues
            fig_eig, ax_eig = plt.subplots(figsize=(10, 6))
            ax_eig.bar(range(len(eigenvalues)), eigenvalues, alpha=0.7, color='steelblue')
            ax_eig.set_xlabel("Eigenvalue Index")
            ax_eig.set_ylabel("Eigenvalue")
            ax_eig.set_title(f"Kernel Eigenvalues (d={d}, P={P}, N={N}, κ={kappa:.6e})")
            ax_eig.grid(True, alpha=0.3, axis='y')
            fig_eig.tight_layout()
            fig_eig.savefig(str(run_dir / "kernel_eigenvalues.png"), dpi=150)
            plt.close(fig_eig)
            
            print(f"  Computed {len(eigenvalues)} eigenvalues")
            
            # Save eigenvalues to JSON
            eigenvalues_data = {
                "eigenvalues": eigenvalues.tolist(),
                "num_eigenvalues": len(eigenvalues)
            }
            with open(run_dir / "eigenvalues.json", "w") as f:
                json.dump(eigenvalues_data, f, indent=2)
                
        except Exception as e:
            print(f"  Warning: Could not compute eigenvalues with H_eig: {e}")
    
    return slopes, losses, loss_stds, final_metrics

def main():
    d = 2
    P = 6
    N = 30
    kappas = [1/6, 2.0]
    epochs = 2_000_000
    log_interval = 50_000
    
    all_slopes = {}  # kappa -> slopes
    all_losses = {}  # kappa -> losses
    all_loss_stds = {}  # kappa -> loss_stds
    all_metrics = {}  # kappa -> final_metrics
    
    # Train all networks
    for kappa in kappas:
        key = f"k_{kappa:.6e}"
        slopes, losses, loss_stds, final_metrics = train_and_track(d, P, N, kappa, epochs=epochs, log_interval=log_interval)
        all_slopes[key] = slopes
        all_losses[key] = losses
        all_loss_stds[key] = loss_stds
        all_metrics[key] = final_metrics
        print()
    
    # Save slopes and losses to JSON
    output_file = Path(__file__).parent / "slopes_data.json"
    with open(output_file, "w") as f:
        json.dump(all_slopes, f, indent=2)
    print(f"Saved slopes to {output_file}")
    
    loss_file = Path(__file__).parent / "losses_data.json"
    with open(loss_file, "w") as f:
        json.dump(all_losses, f, indent=2)
    print(f"Saved losses to {loss_file}")
    
    # Save final metrics summary
    metrics_file = Path(__file__).parent / "final_metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved final metrics to {metrics_file}")
    
    # Plot slopes comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'r']
    
    for color, kappa in zip(colors, kappas):
        key = f"k_{kappa:.6e}"
        slopes_data = all_slopes[key]
        epochs_list = sorted(slopes_data.keys())
        slope_vals = [slopes_data[e] for e in epochs_list]
        ax.plot(epochs_list, slope_vals, marker='o', label=f"κ={kappa:.6e}", color=color, linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Slope (Model vs GPR)")
    ax.set_title(f"d={d}, P={P}, N={N}, chi={N}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_file = Path(__file__).parent / "slopes_comparison.png"
    fig.savefig(str(plot_file), dpi=150)
    print(f"Saved plot to {plot_file}")
    plt.close()
    
    # Plot losses with error bars
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for color, kappa in zip(colors, kappas):
        key = f"k_{kappa:.6e}"
        losses_data = all_losses[key]
        loss_stds_data = all_loss_stds[key]
        epochs_list = sorted(losses_data.keys())
        loss_vals = [losses_data[e] for e in epochs_list]
        loss_std_vals = [loss_stds_data[e] for e in epochs_list]
        ax2.errorbar(epochs_list, loss_vals, yerr=loss_std_vals, marker='o', label=f"κ={kappa:.6e}", 
                   color=color, linewidth=2, capsize=3, capthick=1, alpha=0.7)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (Ensemble-Averaged ± Std)")
    ax2.set_yscale('log')
    ax2.set_title(f"d={d}, P={P}, N={N}, chi={N}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    loss_plot_file = Path(__file__).parent / "losses_comparison.png"
    fig2.savefig(str(loss_plot_file), dpi=150)
    print(f"Saved loss plot to {loss_plot_file}")
    plt.close()
    
    # Plot normalized loss vs kappa
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    kappas_list = []
    normalized_losses = []
    for kappa in kappas:
        key = f"k_{kappa:.6e}"
        kappas_list.append(kappa)
        normalized_losses.append(all_metrics[key]["normalized_loss"])
    
    ax3.plot(kappas_list, normalized_losses, marker='o', markersize=10, linewidth=2.5, color='purple')
    
    ax3.set_xlabel("κ (kappa)", fontsize=12)
    ax3.set_ylabel("Loss / P × χ / κ", fontsize=12)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_title(f"Normalized Loss Term vs Kappa (d={d}, P={P}, N={N}, chi={N})", fontsize=14)
    ax3.grid(True, alpha=0.3, which='both')
    
    fig3.tight_layout()
    normalized_plot_file = Path(__file__).parent / "normalized_loss_vs_kappa.png"
    fig3.savefig(str(normalized_plot_file), dpi=150)
    print(f"Saved normalized loss plot to {normalized_plot_file}")
    plt.close()

if __name__ == "__main__":
    main()
