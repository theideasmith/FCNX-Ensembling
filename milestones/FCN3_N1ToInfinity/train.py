#!/usr/bin/env python3
"""
Training script: FCN3_N1ToInfinity
- Fixed: n1=256, n2=20, chi=n2=20, lr=1e-6, epochs=30M
- Sweep: d ∈ {2, 6}
- P = 3*d (following convention)
- Train for 30 million epochs
- Track eigenvalues every 10K epochs
- Compare with theoretical predictions from ExperimentLinear.eig_predictions()

Usage:
    # Train all d values in parallel:
    python train.py

    # Train specific d value:
    python train.py --d 2 --device cuda:0
"""

import argparse
import subprocess
import signal
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear, FCN3NetworkEnsembleErf
from Experiment import ExperimentLinear

def train_and_track(d, P, n1, n2, chi, epochs=30_000_000, log_interval=10_000, device_str="cuda:1", writer=None):
    """Train network and track eigenvalues over epochs."""
    kappa = 1.0 / chi
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    lr = 1e-6
    temperature = 2 * kappa 
    
    # Setup output directory for this run
    run_dir = Path(__file__).parent / f"d{d}_P{P}_n1{n1}_n2{n2}_chi{chi}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining: d={d}, P={P}, n1={n1}, n2={n2}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Model
    model = FCN3NetworkEnsembleErf(d, n1, n2, P, ens=50, 
                                      weight_initialization_variance=(1/d, 1/n1, 1/(n2 * chi))).to(device)
    model.train()
    
    # Weight decay
    wd_fc1 = d * temperature
    wd_fc2 = n1 * temperature
    wd_fc3 = n2 * temperature * chi
    
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
            
            # Log total sum loss to TensorBoard
            if writer is not None:
                writer.add_scalar('loss/sum_total', loss.item(), epoch)
            
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
                    
                    # Log to TensorBoard - plot all eigenvalues in one chart
                    if writer is not None:
                        # Create dict with all eigenvalues for single plot
                        eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) for i in range(len(eigenvalues))}
                        writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)
                    
                    # Plot eigenvalues over time in log-log scale
                    if len(eigenvalues_over_time) > 1:
                        fig_eig, ax_eig = plt.subplots(figsize=(12, 7))
                        
                        # Extract epochs and eigenvalues
                        epochs_list = sorted([int(e) for e in eigenvalues_over_time.keys()])
                        eig_array = np.array([eigenvalues_over_time[e] for e in epochs_list])
                        
                        # Plot each eigenvalue series
                        for i in range(eig_array.shape[1]):
                            ax_eig.plot(epochs_list, eig_array[:, i], '-', linewidth=1.5, alpha=0.7, label=f'λ_{i}' if i < 5 else '')
                        
                        ax_eig.set_xscale('log')
                        ax_eig.set_yscale('log')
                        ax_eig.set_xlabel("Epochs", fontsize=12)
                        ax_eig.set_ylabel("Eigenvalue", fontsize=12)
                        ax_eig.set_title(f"Eigenvalues vs Epochs (log-log)\n(d={d}, P={P}, n1={n1}, n2={n2}, κ={kappa:.4f})", fontsize=13)
                        ax_eig.grid(True, alpha=0.3, which='both')
                        if eig_array.shape[1] <= 5:
                            ax_eig.legend(fontsize=10, loc='best')
                        fig_eig.tight_layout()
                        fig_eig.savefig(str(run_dir / "eigenvalues_over_epochs.png"), dpi=150)
                        plt.close(fig_eig)
                    
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                
                if epoch > 0:
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
                    # Log loss to TensorBoard
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
                        print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                
                # Save checkpoint
                if epoch > 0:
                    torch.save(model.state_dict(), run_dir / "model.pt")
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save config
    config = {
        "d": d, "P": P, "n1": n1, "n2": n2, "kappa": float(kappa), 
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

def plot_predictions_vs_empirical(d, P, n1, n2, chi, run_dir):
    """Plot theoretical predictions from ExperimentLinear against final empirical eigenvalues."""
    kappa = 1.0
    
    print(f"\nComputing theoretical predictions for d={d}...")
    
    # Create ExperimentLinear instance
    exp = ExperimentLinear(
        file=str(run_dir),
        N=n2,  # Use n2 as the reference for ExperimentLinear
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
        if hasattr(preds, 'lH1T') and preds.lH1T is not None:
            ax.axhline(y=preds.lH1T, color='red', linestyle='-', linewidth=2.5, 
                      label=f'lH1T (pred) = {preds.lH1T:.4f}')
        
        if hasattr(preds, 'lH1P') and preds.lH1P is not None:
            ax.axhline(y=preds.lH1P, color='purple', linestyle='-', linewidth=2.5, 
                      label=f'lH1P (pred) = {preds.lH1P:.4f}')
        
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Eigenvalue", fontsize=12)
        ax.set_title(f"Empirical vs Predicted Eigenvalues\n(d={d}, P={P}, n1={n1}, n2={n2}, κ={kappa:.4f}, χ={chi})", fontsize=13)
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
    parser = argparse.ArgumentParser(description='FCN3_N1ToInfinity training (n1=256, n2=20, epochs=30M)')
    parser.add_argument('--d', type=int, default=None,
                       help='Train specific d value (if not set, launches all in parallel)')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    args = parser.parse_args()
    
    # Fixed parameters
    dims = [2, 6]
    n1 = 1028
    n2 = 20
    chi = n2
    epochs = 30_000_000
    log_interval = 10_000
    lr = 1e-5
    
    # Single d training mode
    if args.d is not None:
        if args.d not in dims:
            print(f"ERROR: d={args.d} not in available dims {dims}")
            sys.exit(1)
        
        d = args.d
        P = 3 * d
        print(f"\n{'='*60}")
        print(f"Starting training for d={d} on {args.device}")
        print(f"Fixed: n1={n1}, n2={n2}, chi={chi}, lr={lr}, epochs={epochs}")
        print(f"{'='*60}")
        
        # Setup TensorBoard
        tensorboard_dir = Path(__file__).parent / "runs" / f"d{d}"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        print(f"TensorBoard logging to: {tensorboard_dir}")
        
        final_eigs, eigs_over_time = train_and_track(
            d, P, n1, n2, chi, epochs=epochs, log_interval=log_interval,
            device_str=args.device, writer=writer
        )
        
        writer.close()
        
        # Generate prediction plot
        run_dir = Path(__file__).parent / f"d{d}_P{P}_n1{n1}_n2{n2}_chi{chi}"
        plot_predictions_vs_empirical(d, P, n1, n2, chi, run_dir)
        
        print(f"\nTraining completed for d={d}")
        sys.exit(0)
    
    # Parallel training mode
    print(f"Launching parallel training for d values {dims}")
    print(f"Fixed: n1={n1}, n2={n2}, chi={chi}, lr={lr}, epochs={epochs}")
    print("=" * 60)
    
    script_path = Path(__file__).resolve()
    processes = []
    
    # Signal handler to terminate children on SIGINT
    def signal_handler(sig, frame):
        print("\n\nReceived SIGINT, terminating all child processes...")
        for d, proc in processes:
            if proc.poll() is None:  # Process still running
                print(f"  Terminating d={d} (PID {proc.pid})")
                proc.terminate()
        
        # Wait a bit for graceful termination
        import time
        time.sleep(2)
        
        # Force kill if still alive
        for d, proc in processes:
            if proc.poll() is None:
                print(f"  Force killing d={d} (PID {proc.pid})")
                proc.kill()
        
        print("All child processes terminated.")
        sys.exit(1)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Assign different GPUs to different d values
    gpu_devices = ["cuda:0"]
    
    for idx, d in enumerate(dims):
        device = gpu_devices[idx % len(gpu_devices)]
        cmd = [
            sys.executable,
            str(script_path),
            "--d", str(d),
            "--device", device,
        ]
        
        print(f"Launching d={d} on {device}...")
        proc = subprocess.Popen(cmd)
        processes.append((d, proc))
    
    print(f"\nLaunched {len(processes)} parallel training processes")
    print("Waiting for all processes to complete...")
    print("Press Ctrl+C to terminate all processes")
    
    # Wait for all processes
    try:
        for d, proc in processes:
            proc.wait()
            if proc.returncode == 0:
                print(f"✓ d={d} completed successfully")
            else:
                print(f"✗ d={d} failed with return code {proc.returncode}")
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    print("\nAll training completed!")

if __name__ == "__main__":
    main()
