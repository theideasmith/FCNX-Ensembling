#!/usr/bin/env python3
"""
D-sweep for erf mean-field scaling convergence:
- Sweep d = 2, 6, 8, 10 with P=3*d, N=50, chi=N
- kappa = 1/chi for each d
- Train for 300M epochs
- Track eigenvalues every 10K epochs
- Plot largest eigenvalue and mean of others vs d
- Compare with theoretical predictions from ExperimentErf.eig_predictions()

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
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleErf
from GPKit import gpr_dot_product_explicit
from Experiment import Experiment

# Optional theoretical solver (may be missing in some environments)
try:
    from FCS import nlsolve_solver
except Exception:
    nlsolve_solver = None

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def train_and_track(d, P, N, epochs=300_000_000, log_interval=25_000, device_str="cuda:1", writer=None):
    """Train network with erf activation and track eigenvalues over epochs."""
    

    chi = N  # Mean-field scaling
    kappa = 1.0 / chi
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    lr = 1e-5 / P 
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
    Y1 = X[:, 0].squeeze(-1).unsqueeze(-1)
    Y3 = (Y1**3 - 3 * Y1) * 1/6
    Y = Y1 + 0.03 * Y3
    # Model with erf activation
    model = FCN3NetworkEnsembleErf(d, N, N, P, ens=10, 
                                    weight_initialization_variance=(1/d, 1/N, 1/(N * chi))).to(device)
    model.device = device
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
    Xinf = torch.randn(800, d, device=device)  # large eval set for eigenvalues
    for epoch in range(epochs + 1):  # +1 to include epoch 0
        # Forward pass (skip for epoch 0)
        if epoch > 0:
            if epoch > 200_000_000:
                lr = 5e-6 / P
            output = model(X)  # shape: (P, ensemble)
            # Compute per-ensemble losses
            diff = output - Y  # (P, ensemble)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ensemble,)
            loss = per_ensemble_loss.sum()
            
            # Compute ensemble-averaged and std loss
            loss_avg = loss.item() / model.ens
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
                # Compute eigenvalues (full H) and linear/cubic projections on GPU (or current device)
                try:
                    Y1_inf = Xinf
                    Y3_inf = (Y1_inf**3 - 3 * Y1_inf) * 1/6
                    eigenvalues = model.H_eig(Xinf, Xinf).detach().cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()

                    # Linear (Y1) and cubic (Y3) eigenvalues per provided recipe on the same device as Xinf
                    try:
                        eigs_lin_cub = compute_linear_cubic_eigenvalues(
                            SimpleNamespace(model=model), Xinf, Y1_inf, Y3_inf
                        )
                    except Exception as sub_e:
                        eigs_lin_cub = None
                        print(f"  Warning: failed Y1/Y3 eigen computation at epoch {epoch}: {sub_e}")

                    # Log to TensorBoard - all eigenvalues + Y1/Y3 means
                    if writer is not None:
                        eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) for i in range(len(eigenvalues))}
                        writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)

                        if eigs_lin_cub is not None:
                            eig_lin = eigs_lin_cub["eig_linear"].cpu().numpy()
                            eig_cub = eigs_lin_cub["eig_cubic"].cpu().numpy()
                            writer.add_scalar('eigenvalues/Y1_mean', float(eig_lin.mean()), epoch)
                            writer.add_scalar('eigenvalues/Y3_mean', float(eig_cub.mean()), epoch)
                            # Also log first few components for inspection
                            log_len = min(5, eig_lin.shape[0])
                            y1_dict = {f'eigY1_{i}': float(eig_lin[i]) for i in range(log_len)}
                            y3_dict = {f'eigY3_{i}': float(eig_cub[i]) for i in range(log_len)}
                            writer.add_scalars('eigenvalues/Y1_components', y1_dict, epoch)
                            writer.add_scalars('eigenvalues/Y3_components', y3_dict, epoch)

                    # Plot eigenvalues over time in log-log scale (same as before)
                    if len(eigenvalues_over_time) > 1:
                        fig_eig, ax_eig = plt.subplots(figsize=(12, 7))

                        epochs_list = sorted([int(e) for e in eigenvalues_over_time.keys()])
                        eig_array = np.array([eigenvalues_over_time[e] for e in epochs_list])

                        for i in range(eig_array.shape[1]):
                            ax_eig.plot(epochs_list, eig_array[:, i], '-', linewidth=1.5, alpha=0.7, label=f'λ_{i}' if i < 5 else '')

                        ax_eig.set_xscale('log')
                        ax_eig.set_yscale('log')
                        ax_eig.set_xlabel("Epochs", fontsize=12)
                        ax_eig.set_ylabel("Eigenvalue", fontsize=12)
                        ax_eig.set_title(f"Eigenvalues vs Epochs (log-log)\n(d={d}, P={P}, N={N}, κ={kappa:.4f})", fontsize=13)
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
        "d": d, "P": P, "N": N, "kappa": float(kappa), 
        "lr": float(lr), "epochs": epochs, "chi": chi,
        "activation": "erf"
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
    """Plot theoretical predictions from ExperimentErf against final empirical eigenvalues."""
    chi = N
    kappa = 1.0 / chi
    
    print(f"\nComputing theoretical predictions for d={d}...")
    
    # Create ExperimentErf instance
    exp = Experiment(
        file=str(run_dir),
        N=N,
        d=d,
        chi=chi,
        P=P,
        ens=10,
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
        # lHT (target direction) and lHP (perpendicular direction)
        if hasattr(preds, 'lHT') and preds.lHT is not None:
            ax.axhline(y=preds.lHT, color='red', linestyle='-', linewidth=2.5, 
                      label=f'lHT (target) = {preds.lHT:.4f}')
        
        # lHP (perpendicular direction)
        if hasattr(preds, 'lHP') and preds.lHP is not None:
            ax.axhline(y=preds.lHP, color='purple', linestyle='-', linewidth=2.5, 
                      label=f'lHP (perpendicular) = {preds.lHP:.4f}')
        
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Eigenvalue", fontsize=12)
        ax.set_title(f"Empirical vs Predicted Eigenvalues (Erf)\n(d={d}, P={P}, N={N}, κ={kappa:.4f}, χ={chi})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        fig.savefig(str(run_dir / "eigenvalues_vs_predictions.png"), dpi=150)
        plt.close(fig)
        
        print(f"  Predictions: lHT={preds.lHT:.6f}, lHP={preds.lHP:.6f}")
        print(f"  Empirical: max={empirical_eigs.max():.6f}, mean={empirical_eigs.mean():.6f}")
        
        # Save predictions
        pred_dict = {
            "lHT": float(preds.lHT) if hasattr(preds, 'lHT') else None,
            "lHP": float(preds.lHP) if hasattr(preds, 'lHP') else None,
        }
        with open(run_dir / "predictions.json", "w") as f:
            json.dump(pred_dict, f, indent=2)
            
        return preds
        
    except Exception as e:
        print(f"  Error computing predictions for d={d}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _loglog_slope(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    """Least-squares slope on log-log axes."""
    mask = (~np.isnan(xs)) & (~np.isnan(ys)) & (xs > 0) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or ys.size < 2:
        return None
    X = np.log(xs)
    Y = np.log(ys)
    A = np.vstack([X, np.ones_like(X)]).T
    m, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m)
import traceback as stacktrace

def compute_linear_cubic_eigenvalues(experimentMF, X: torch.Tensor, Y1: torch.Tensor, Y3: torch.Tensor):
    """Compute eigenvalues for linear (Y1) and cubic (Y3) eigenfunctions.

    This follows the provided QB + SVD workflow and returns empirical eigenvalues:
    - sigma_diag: diagonal of Sigma / |X|
    - eig_linear: eigenvalues associated with Y1
    - eig_cubic: eigenvalues associated with Y3
    """
    try:
        with torch.no_grad():
            Q, Z = experimentMF.model.H_random_QB(X, k=500, p=10)
            Ut, _S, V = torch.linalg.svd(Z.T)
            m, n = Z.shape[1], Z.shape[0]
            k = min(m, n)
            Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
            Sigma[:k, :k] = torch.diag(_S)
            U = torch.matmul(Q, Ut)
            sigma_diag = Sigma.diagonal() / X.shape[0]

            Y1_norm = Y1 / torch.norm(Y1, dim=0)
            left_eigenvaluesY1 = experimentMF.model.H_eig(X, Y1_norm)

            Y3_norm = Y3 / torch.norm(Y3, dim=0)
            left_eigenvaluesY3 = (
                torch.matmul(Y3_norm.t(), U)
                @ torch.diag(_S)
                @ torch.matmul(U.T, Y3_norm)
            ).diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]
        return {
            "sigma_diag": sigma_diag.detach().cpu(),
            "eig_linear": left_eigenvaluesY1.detach().cpu(),
            "eig_cubic": left_eigenvaluesY3.detach().cpu(),
        }
    except Exception as e:
        print(f"  Error in compute_linear_cubic_eigenvalues: {e}")
        stacktrace.print_exc()
        raise e

 


def theoretical_linear_cubic(d: int, P: int, N: int, chi: int):
    """Get theoretical eigenvalues (linear, cubic) using nlsolve_solver if available."""
    if nlsolve_solver is None:
        return None, None
    try:
        sol = nlsolve_solver(d=d, P=P, N=N, chi=chi)
        lH1T = getattr(sol, "lH1T", None)
        lH3T = getattr(sol, "lH3T", None)
        return lH1T, lH3T
    except Exception as e:
        print(f"  Warning: nlsolve_solver failed for d={d}: {e}")
        return None, None


def plot_linear_cubic_scaling(d_vals, emp_lin, emp_cub, th_lin, th_cub, out_path: Path):
    """Plot empirical/theoretical linear and cubic eigenvalues vs d on log-log axes."""
    d_arr = np.array(d_vals, dtype=float)
    emp_lin_arr = np.array(emp_lin, dtype=float)
    emp_cub_arr = np.array(emp_cub, dtype=float)
    th_lin_arr = np.array(th_lin, dtype=float)
    th_cub_arr = np.array(th_cub, dtype=float)

    slope_emp_lin = _loglog_slope(d_arr, emp_lin_arr)
    slope_emp_cub = _loglog_slope(d_arr, emp_cub_arr)
    slope_th_lin = _loglog_slope(d_arr, th_lin_arr)
    slope_th_cub = _loglog_slope(d_arr, th_cub_arr)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(d_arr, emp_lin_arr, 'o-', color='tab:blue', linewidth=2.0, markersize=8,
              label=f"Empirical $\\lambda_H^{{(1)}}$ (slope={slope_emp_lin:.3f})" if slope_emp_lin is not None else "Empirical $\\lambda_H^{(1)}$")
    ax.loglog(d_arr, emp_cub_arr, '^-', color='tab:green', linewidth=2.0, markersize=8, alpha=0.8,
              label=f"Empirical $\\lambda_H^{{(3)}}$ (slope={slope_emp_cub:.3f})" if slope_emp_cub is not None else "Empirical $\\lambda_H^{(3)}$")
    ax.loglog(d_arr, th_lin_arr, 's--', color='tab:red', linewidth=2.0, markersize=7,
              label=f"Theory $\\lambda_H^{{(1)}}$ (slope={slope_th_lin:.3f})" if slope_th_lin is not None else "Theory $\\lambda_H^{(1)}$")
    ax.loglog(d_arr, th_cub_arr, 'D--', color='tab:orange', linewidth=2.0, markersize=7, alpha=0.8,
              label=f"Theory $\\lambda_H^{{(3)}}$ (slope={slope_th_cub:.3f})" if slope_th_cub is not None else "Theory $\\lambda_H^{(3)}$")

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Linear vs Cubic Eigenvalues vs $d$ (log-log)")
    if len(d_arr) > 0:
        ax.set_xlim(left=d_arr[0] * 0.8)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved linear/cubic scaling plot to {out_path}")
    return slope_emp_lin, slope_emp_cub, slope_th_lin, slope_th_cub


def analyze_linear_cubic_scaling(dims, N, base_dir: Path, device_str: str = "cuda:1"):
    """Load saved runs, compute linear/cubic eigenvalues, and plot scaling vs d.

    Assumes run folders named d{d}_P{3d}_N{N}_chi{N} with saved model.pt/model_final.pt.
    Uses the same data generation seed (42) and sets Y1=X (linear) and Y3=X^3 (cubic).
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    d_list = []
    emp_lin_list = []
    emp_cub_list = []
    th_lin_list = []
    th_cub_list = []

    for d in dims:
        P = 3 * d
        run_dir = base_dir / f"d{d}_P{P}_N{N}_chi{N}"
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            model_path = run_dir / "model_final.pt"
        if not model_path.exists():
            print(f"  Skipping d={d}: no model checkpoint found")
            continue

        # Load model
        model = FCN3NetworkEnsembleErf(d, N, N, P, ens=10,
                                       weight_initialization_variance=(1/d, 1/N, 1/(N * N))).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        torch.manual_seed(42)
        X = torch.randn(P, d, device=device)
        Y1 = X  # linear eigenfunctions
        Y3 = X ** 3  # cubic eigenfunctions

        exp_wrapper = SimpleNamespace(model=model)
        eigs = compute_linear_cubic_eigenvalues(exp_wrapper, X, Y1, Y3)

        # Aggregate as mean eigenvalue per family for scaling visualization
        emp_lin_val = float(eigs["eig_linear"].mean())
        emp_cub_val = float(eigs["eig_cubic"].mean())

        lH1T, lH3T = theoretical_linear_cubic(d=d, P=P, N=N, chi=N)

        d_list.append(d)
        emp_lin_list.append(emp_lin_val)
        emp_cub_list.append(emp_cub_val)
        th_lin_list.append(float(lH1T) if lH1T is not None else np.nan)
        th_cub_list.append(float(lH3T) if lH3T is not None else np.nan)

    if not d_list:
        print("No runs found for linear/cubic analysis.")
        return

    # Sort by d for clean plotting
    order = np.argsort(np.array(d_list))
    d_list = list(np.array(d_list)[order])
    emp_lin_list = list(np.array(emp_lin_list)[order])
    emp_cub_list = list(np.array(emp_cub_list)[order])
    th_lin_list = list(np.array(th_lin_list)[order])
    th_cub_list = list(np.array(th_cub_list)[order])

    out_path = base_dir / "plots" / "linear_cubic_scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_linear_cubic_scaling(d_list, emp_lin_list, emp_cub_list, th_lin_list, th_cub_list, out_path)

def main():
    parser = argparse.ArgumentParser(description='D-sweep training with parallel execution (erf activation)')
    parser.add_argument('--d', type=int, default=None,
                       help='Train specific d value (if not set, launches all in parallel)')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--analyze-linear-cubic', action='store_true',
                        help='Only analyze/plot linear & cubic eigenvalue scaling across saved runs')
    args = parser.parse_args()
    
    dims = [4, 6, 8, 10]
    N = 50
    epochs = 300_000_000
    log_interval = 10_000
    
    # Analysis-only mode
    if args.analyze_linear_cubic:
        base_dir = Path(__file__).parent
        analyze_linear_cubic_scaling(dims, N, base_dir, device_str=args.device)
        return

    # Single d training mode
    if args.d is not None:
        if args.d not in dims:
            print(f"ERROR: d={args.d} not in available dims {dims}")
            sys.exit(1)
        
        d = args.d
        P = 10 * d
        print(f"\n{'='*60}")
        print(f"Starting training for d={d} with erf activation on {args.device}")
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
    print("Launching parallel training for all d values with erf activation...")
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
