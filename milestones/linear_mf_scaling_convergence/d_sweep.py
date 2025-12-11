#!/usr/bin/env python3
"""
D-sweep for linear mean-field scaling convergence:
- Sweep d = 2, 6, 8, 10 with P=6, N=1028, chi=N
- kappa = 1/d for each d
- Train for 2 million epochs
- Track eigenvalues every 10K epochs
- Plot largest eigenvalue and mean of others vs d
- Compare with theoretical predictions from ExperimentLinear.eig_predictions()
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
from FCN3Network import FCN3NetworkEnsembleLinear
from GPKit import gpr_dot_product_explicit
from Experiment import ExperimentLinear

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def train_and_track(d, P, N, epochs=2_000_000, log_interval=10_000):
    """Train network and track eigenvalues over epochs."""
    kappa = 0.1
    chi = N  # Mean-field scaling
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 1e-7 / P
    temperature = 2 * kappa
    
    # Setup output directory for this run
    run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi{N}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Model
    model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=50, 
                                      weight_initialization_variance=(1/d, 1/N, 1/(N * chi))).to(device)
    model.train()
    
    # Weight decay
    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    
    # Training loop
    noise_scale = np.sqrt(2.0 * lr * temperature)
    losses = {}  # epoch -> loss averaged over ensemble
    loss_stds = {}  # epoch -> loss std over ensemble
    eigenvalues_over_time = {}  # epoch -> list of eigenvalues
    Xinf = torch.randn(3000, d, device=device)  # large eval set for eigenvalues
    for epoch in range(epochs + 1):  # +1 to include epoch 0
        # Forward pass (skip for epoch 0)
        if epoch > 0:
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
        
        # Logging and eigenvalue computation
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues
                try:
                    
                    eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    
                    # Plot eigenvalues and save (overwrite each time)
                    fig_eig, ax_eig = plt.subplots(figsize=(10, 6))
                    ax_eig.bar(range(len(eigenvalues)), eigenvalues, alpha=0.7, color='steelblue')
                    ax_eig.set_xlabel("Eigenvalue Index", fontsize=12)
                    ax_eig.set_ylabel("Eigenvalue", fontsize=12)
                    ax_eig.set_title(f"Kernel Eigenvalues at Epoch {epoch}\n(d={d}, P={P}, N={N}, κ={kappa:.4f})", fontsize=13)
                    ax_eig.grid(True, alpha=0.3, axis='y')
                    fig_eig.tight_layout()
                    fig_eig.savefig(str(run_dir / "eigenvalues_over_epochs.png"), dpi=150)
                    plt.close(fig_eig)
                    
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                
                if epoch > 0:
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}, "
                              f"max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                    else:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")
                else:
                    if eigenvalues is not None:
                        print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                
                # Save checkpoint
                if epoch > 0:
                    torch.save(model.state_dict(), run_dir / "model.pt")
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, "kappa": float(kappa), 
        "lr": float(lr), "epochs": epochs, "chi": chi
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save eigenvalues over time
    with open(run_dir / "eigenvalues_over_time.json", "w") as f:
        json.dump(eigenvalues_over_time, f, indent=2)
    
    # Save losses
    with open(run_dir / "losses.json", "w") as f:
        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
    
    # Get final eigenvalues
    final_eigenvalues = None
    if epochs in eigenvalues_over_time:
        final_eigenvalues = np.array(eigenvalues_over_time[epochs])
    
    return final_eigenvalues, eigenvalues_over_time

def plot_predictions_vs_empirical(d, P, N, run_dir):
    """Plot theoretical predictions from ExperimentLinear against final empirical eigenvalues."""
    kappa = 1.0 / d
    chi = N
    
    print(f"\nComputing theoretical predictions for d={d}...")
    
    # Create ExperimentLinear instance
    exp = ExperimentLinear(
        file=str(run_dir),
        N=N,
        d=d,
        chi=chi,
        P=P,
        ens=50,
        kappa=kappa,
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    )
    
    try:
        # Get predictions
        preds = exp.eig_predictions()
        
        # Load final eigenvalues
        with open(run_dir / "eigenvalues_over_time.json", "r") as f:
            eig_data = json.load(f)
        
        # Get final epoch eigenvalues
        final_epoch = max([int(k) for k in eig_data.keys()])
        empirical_eigs = np.array(eig_data[str(final_epoch)])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot empirical eigenvalues as bars
        ax.bar(range(len(empirical_eigs)), empirical_eigs, alpha=0.6, 
               color='steelblue', label='Empirical Eigenvalues')
        
        # Plot theoretical predictions as horizontal lines
        # lH1T and lH3T (training regime)
        if hasattr(preds, 'lH1T') and preds.lH1T is not None:
            ax.axhline(y=preds.lH1T, color='red', linestyle='-', linewidth=2.5, 
                      label=f'lH1T (pred) = {preds.lH1T:.4f}')
        
        # lH1P and lH3P (population regime)
        if hasattr(preds, 'lH1P') and preds.lH1P is not None:
            ax.axhline(y=preds.lH1P, color='purple', linestyle='-', linewidth=2.5, 
                      label=f'lH1P (pred) = {preds.lH1P:.4f}')
        
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Eigenvalue", fontsize=12)
        ax.set_title(f"Empirical vs Predicted Eigenvalues\n(d={d}, P={P}, N={N}, κ={kappa:.4f}, χ={chi})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        fig.savefig(str(run_dir / "eigenvalues_vs_predictions.png"), dpi=150)
        plt.close(fig)
        
        print(f"  Predictions: lH1T={preds.lH1T:.6f}, lH1P={preds.lH1P:.6f}")
        print(f"  Empirical: max={empirical_eigs.max():.6f}, mean={empirical_eigs.mean():.6f}")
        
        # Save predictions
        pred_dict = {
            "lH1T": float(preds.lH1T) if hasattr(preds, 'lH1T') else None,
            "lH3T": float(preds.lH3T) if hasattr(preds, 'lH3T') else None,
            "lH1P": float(preds.lH1P) if hasattr(preds, 'lH1P') else None,
            "lH3P": float(preds.lH3P) if hasattr(preds, 'lH3P') else None,
        }
        with open(run_dir / "predictions.json", "w") as f:
            json.dump(pred_dict, f, indent=2)
            
        return preds
        
    except Exception as e:
        print(f"  Error computing predictions for d={d}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    dims = [2, 6, 8, 10]
    P = 6
    N = 50
    epochs = 2_000_000
    log_interval = 10_000
    
    all_results = {}  # d -> {"final_eigs": array, "eigs_over_time": dict}
    
    # Train all networks
    for d in dims:
        print(f"\n{'='*60}")
        print(f"Starting training for d={d}")
        print(f"{'='*60}")
        
        final_eigs, eigs_over_time = train_and_track(d, P, N, epochs=epochs, log_interval=log_interval)
        all_results[d] = {
            "final_eigs": final_eigs,
            "eigs_over_time": eigs_over_time
        }
    
    # Generate prediction plots for each model
    print(f"\n{'='*60}")
    print("Generating theoretical prediction plots")
    print(f"{'='*60}")
    
    for d in dims:
        run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi{N}"
        plot_predictions_vs_empirical(d, P, N, run_dir)
    
    # Plot largest eigenvalue and mean of others vs d
    print(f"\n{'='*60}")
    print("Creating summary plots")
    print(f"{'='*60}")
    
    largest_eigs = []
    mean_other_eigs = []
    
    for d in dims:
        final_eigs = all_results[d]["final_eigs"]
        if final_eigs is not None and len(final_eigs) > 0:
            largest_eigs.append(final_eigs.max())
            if len(final_eigs) > 1:
                # Mean of all except the largest
                other_eigs = np.sort(final_eigs)[:-1]
                mean_other_eigs.append(other_eigs.mean())
            else:
                mean_other_eigs.append(0.0)
        else:
            largest_eigs.append(np.nan)
            mean_other_eigs.append(np.nan)
    
    # Plot largest eigenvalue vs d
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dims, largest_eigs, marker='o', markersize=10, linewidth=2.5, color='darkblue')
    ax1.set_xlabel("Dimension d", fontsize=12)
    ax1.set_ylabel("Largest Eigenvalue", fontsize=12)
    ax1.set_title(f"Largest Eigenvalue vs Dimension\n(P={P}, N={N}, χ={N}, κ=1/d)", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(str(Path(__file__).parent / "largest_eigenvalue_vs_d.png"), dpi=150)
    plt.close(fig1)
    
    # Plot mean of other eigenvalues vs d
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(dims, mean_other_eigs, marker='s', markersize=10, linewidth=2.5, color='darkred')
    ax2.set_xlabel("Dimension d", fontsize=12)
    ax2.set_ylabel("Mean of Other Eigenvalues", fontsize=12)
    ax2.set_title(f"Mean of Non-Largest Eigenvalues vs Dimension\n(P={P}, N={N}, χ={N}, κ=1/d)", fontsize=13)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(Path(__file__).parent / "mean_other_eigenvalues_vs_d.png"), dpi=150)
    plt.close(fig2)
    
    # Save summary data
    summary = {
        "dims": dims,
        "largest_eigenvalues": largest_eigs,
        "mean_other_eigenvalues": mean_other_eigs,
        "P": P,
        "N": N,
        "chi": N
    }
    with open(Path(__file__).parent / "d_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {Path(__file__).parent / 'd_sweep_summary.json'}")
    print(f"Plots saved to {Path(__file__).parent}")
    print("\nDone!")

if __name__ == "__main__":
    main()
