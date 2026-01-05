#!/usr/bin/env python3
"""
D-sweep for linear mean-field scaling convergence:
- Sweep d = 2, 6, 8, 10 with P=3*d, N=50, chi=N
- kappa = 0.1 for each d
- Train for 300M epochs
- Track eigenvalues every 10K epochs
- Plot largest eigenvalue and mean of others vs d
- Compare with theoretical predictions from ExperimentLinear.eig_predictions()

Usage:
    # Train all d values in parallel:
    python d_sweep.py

    # Train specific d value:
    python d_sweep.py --d 2 --device cuda:0
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
from FCN3Network import FCN3NetworkActivationGeneric
from GPKit import gpr_dot_product_explicit
from Experiment import ExperimentLinear

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def train_and_track(d, P, N, epochs=300_000_000, log_interval=10_000, device_str="cuda:1", writer=None):
    """Train network and track eigenvalues over epochs."""
    

    chi = N  # Mean-field scaling
    kappa = 1.0 / chi
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    lr = 1e-5  
    temperature = 2 * kappa
    
    # Setup output directory for this run
    run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi{N}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining: d={d}, P={P}, N={N}, kappa={kappa:.6e}, chi={chi}, lr={lr:.6e}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].squeeze(-1).unsqueeze(-1)
    
    # Model
    model = FCN3NetworkActivationGeneric(d, N, N, P, ens=50,
                                         activation="linear",
                                         weight_initialization_variance=(1/d, 1/N, 1/(N * chi))).to(device)
    
    # Try to load existing model checkpoint
    model_checkpoint = run_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    if model_checkpoint.exists():
        print(f"Loading existing model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        
        # Try to load existing training state
        eigenvalues_path = run_dir / "eigenvalues_over_time.json"
        losses_path = run_dir / "losses.json"
        eigenvalues_over_time = {}
        try:
            if eigenvalues_path.exists():
                with open(eigenvalues_path, "r") as f:
                    eigenvalues_over_time = json.load(f)
                # Find last checkpoint epoch
                if eigenvalues_over_time:
                    start_epoch = max([int(k) for k in eigenvalues_over_time.keys()])
                    print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"  Warning: Could not load eigenvalues_over_time: {e}")
            eigenvalues_over_time = {}
            
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
        else:
            losses = {}
            loss_stds = {}
    else:
        eigenvalues_over_time = {}
        losses = {}
        loss_stds = {}
    
    model.train()
    
    # Weight decay
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
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues at epoch 0: {e}")
    
    for epoch in range(start_epoch, epochs + 1):  # Resume from start_epoch
        # Forward pass (skip for epoch 0)
        if epoch >= 0:
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
        import traceback
        # Logging and eigenvalue computation
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues
                try:
                    
                    eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                    try:

                        eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    except Exception as e:
                        print(f"  Warning: Could not store eigenvalues at epoch {epoch}: {e}")
                    
                    # Log to TensorBoard - plot all eigenvalues in one chart
                    if writer is not None:
                        # Create dict with all eigenvalues for single plot
                        eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) for i in range(len(eigenvalues))}
                        writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)
                    
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                
                # Check if eigenvalues exist before using
                try:
                    eigenvalues_exist = eigenvalues is not None
                except:
                    eigenvalues_exist = False
                
                if epoch >= 0:
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
                    # Log loss to TensorBoard
                    if writer is not None:
                        writer.add_scalar('loss/mean', loss_avg, epoch)
                        writer.add_scalar('loss/std', loss_std, epoch)
                    
                    if eigenvalues_exist:
                        try:
                            print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}, "
                                  f"max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                        except:
                            print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")
                    else:
                        print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")
                else:
                    if eigenvalues_exist:
                        try:
                            print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                        except:
                            pass
                
                # Save checkpoint
                if epoch > 0:
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    
                    # Also save intermediate results periodically
                    try:
                        with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                            json.dump(eigenvalues_over_time, f, indent=2)
                    except Exception as save_e:
                        print(f"  Warning: Could not save eigenvalues: {save_e}")
                    
                    with open(run_dir / "losses.json", "w") as f:
                        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
    
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
    try:
        with open(run_dir / "eigenvalues_over_time.json", "w") as f:
            json.dump(eigenvalues_over_time, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not save final eigenvalues: {e}")
    
    # Save losses
    with open(run_dir / "losses.json", "w") as f:
        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
    
    # Get final eigenvalues
    final_eigenvalues = None
    try:
        if epochs in eigenvalues_over_time:
            final_eigenvalues = np.array(eigenvalues_over_time[epochs])
    except Exception as e:
        print(f"  Warning: Could not retrieve final eigenvalues: {e}")
    
    return final_eigenvalues, eigenvalues_over_time

def plot_predictions_vs_empirical(d, P, N, run_dir):
    """Plot theoretical predictions from ExperimentLinear against final empirical eigenvalues."""
    kappa = 1.0 
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
        empirical_eigs = None
        try:
            with open(run_dir / "eigenvalues_over_time.json", "r") as f:
                eig_data = json.load(f)
            
            # Get final epoch eigenvalues
            if eig_data:
                final_epoch = max([int(k) for k in eig_data.keys()])
                empirical_eigs = np.array(eig_data[str(final_epoch)])
        except Exception as eig_e:
            print(f"  Warning: Could not load empirical eigenvalues: {eig_e}")
        
        if empirical_eigs is None:
            print(f"  No empirical eigenvalues available for plotting")
            return None
        
        # # Create plot
        # fig, ax = plt.subplots(figsize=(10, 6))
        
        # # Plot empirical eigenvalues as bars
        # ax.bar(range(len(empirical_eigs)), empirical_eigs, alpha=0.6, 
        #        color='steelblue', label='Empirical Eigenvalues')
        
        # # Plot theoretical predictions as horizontal lines
        # # lH1T and lH3T (training regime)
        # if hasattr(preds, 'lHT') and preds.lHT is not None:
        #     ax.axhline(y=preds.lHT, color='red', linestyle='-', linewidth=2.5, 
        #               label=f'lH1T (pred) = {preds.lH1T:.4f}')
        
        # # lH1P and lH3P (population regime)
        # if hasattr(preds, 'lH1P') and preds.lHP is not None:
        #     ax.axhline(y=preds.lHP, color='purple', linestyle='-', linewidth=2.5, 
        #               label=f'lH1P (pred) = {preds.lHP:.4f}')
        
        # ax.set_xlabel("Eigenvalue Index", fontsize=12)
        # ax.set_ylabel("Eigenvalue", fontsize=12)
        # ax.set_title(f"Empirical vs Predicted Eigenvalues\n(d={d}, P={P}, N={N}, κ={kappa:.4f}, χ={chi})", fontsize=13)
        # ax.legend(fontsize=10)
        # ax.grid(True, alpha=0.3, axis='y')
        
        # fig.tight_layout()
        # fig.savefig(str(run_dir / "eigenvalues_vs_predictions.png"), dpi=150)
        # plt.close(fig)
        
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
    parser = argparse.ArgumentParser(description='D-sweep training with parallel execution')
    parser.add_argument('--d', type=int, default=None,
                       help='Train specific d value (if not set, launches all in parallel)')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    args = parser.parse_args()
    
    dims = [200, 300, 400]

    epochs = 1_000_000
    log_interval = 10_000
    
    # Single d training mode
    if args.d is not None:
        if args.d not in dims:
            print(f"ERROR: d={args.d} not in available dims {dims}")
            sys.exit(1)
        
        d = args.d
        P = 3 * d
        N = 6 * d
        print(f"\n{'='*60}")
        print(f"Starting training for d={d} on {args.device}")
        print(f"{'='*60}")
        
        # Setup TensorBoard
        tensorboard_dir = Path(__file__).parent / "runs" / f"d{d}"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        print(f"TensorBoard logging to: {tensorboard_dir}")
        
        final_eigs, eigs_over_time = train_and_track(
            d, P, N, epochs=epochs, log_interval=log_interval,
            device_str=args.device, writer=writer
        )
        
        writer.close()
        
        # Generate prediction plot
        run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi{N}"
        plot_predictions_vs_empirical(d, P, N, run_dir)
        
        print(f"\nTraining completed for d={d}")
        sys.exit(0)
    
    # Parallel training mode
    print("Launching parallel training for all d values...")
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
    gpu_devices = ["cuda:0", "cuda:1"]
    
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
