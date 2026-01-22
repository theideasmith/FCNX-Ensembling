#!/usr/bin/env python3
"""
Training script for 2-layer erf network with eigenvalue tracking.

Trains a single 2-layer network: Input(d) -> Hidden(n1) with erf -> Output(1)
Tracks eigenvalues of the pre-activation kernel H over training.

Usage:
    python train_fcn2_erf.py --d 10 --P 30 --N 100 --epochs 10000000 --device cuda:0
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
from torch.utils.tensorboard import SummaryWriter

# Set default dtype
torch.set_default_dtype(torch.float32)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def custom_mse_loss(outputs, targets):
    """MSE loss summed over all samples and ensembles."""
    diff = outputs - targets
    return torch.sum(diff * diff)


def _pred_record(epoch, targets, outputs):
    """Create a serializable record of predictions and alignment stats."""
    y_true = targets.squeeze(-1).detach().cpu().numpy()
    y_pred = outputs.mean(dim=1).detach().cpu().numpy()
    y_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    var_y = np.mean((y_true - y_mean) ** 2)
    if var_y == 0:
        slope = float('nan')
        intercept = float('nan')
    else:
        cov = np.mean((y_true - y_mean) * (y_pred - y_pred_mean))
        slope = cov / var_y
        intercept = y_pred_mean - slope * y_mean
    return {
        "epoch": int(epoch),
        "y_true": y_true.tolist(),
        "y_pred_mean": y_pred.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def train_fcn2(d, P, N, eps=0.03, epochs=10_000_000, log_interval=10_000, ens=50,
               device_str="cuda:1", base_lr=1e-5, temperature=0.02, chi=1.0,
               run_dir=None, writer=None, dataset_seed=42, activation="erf"):
    """Train 2-layer erf network and track H eigenvalues.
    
    Args:
        d: Input dimension
        P: Number of training samples
        N: Hidden layer width
        epochs: Number of training iterations
        log_interval: Log eigenvalues every N epochs
        device_str: Device string
        lr: Learning rate
        temperature: Base temperature for weight decay and Langevin noise
        chi: Scaling factor; effective temperature = temperature / chi
        run_dir: Directory to save checkpoints and results
        writer: TensorBoard writer
        dataset_seed: Random seed for data generation
        
    Returns:
        (final_eigenvalues, eigenvalues_over_time, run_dir)
    """
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Setup directory
    if run_dir is None:
        run_dir = Path(__file__).parent / f"d{d}_P{P}_N{N}_chi_{chi}_lr_{base_lr}_T_{temperature}_seed_{dataset_seed}_eps_{eps}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nTraining 2-layer erf network:")
    print(f"  d={d}, P={P}, N={N}")


    effective_temperature = temperature / chi 
    lr = base_lr / P 


    print(f"  lr={lr:.6e}, T={temperature:.6f}, chi={chi:.6f}, T_eff={effective_temperature:.6f}")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")
    
    # Generate data: Y = X[:, 0] (first dimension as target)
    torch.manual_seed(dataset_seed)
    X = torch.randn(P, d, device=device)
    z = X[:, 0].unsqueeze(-1)  # (P, 1)
    z3 = (z ** 3 - 3 * z)/6**0.5  # Cubic nonlinearity
    Y = z + eps * z3  # (P, 1)
    
    # Model
    ens = ens  # ensemble size
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation=activation,
        weight_initialization_variance=(1/d, 1/(N * chi))
    ).to(device)
    
    # Try to load existing checkpoint
    checkpoint_path = run_dir / "checkpoint.pt"
    model_checkpoint = run_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    pred_vs_true = {}
    pred_vs_true_path = run_dir / "pred_vs_true.json"
    
    # Try loading full checkpoint first
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        
        # Load training history from JSON files if they exist
        eigenvalues_path = run_dir / "eigenvalues_over_time.json"
        losses_path = run_dir / "losses.json"
        
        if eigenvalues_path.exists():
            with open(eigenvalues_path, "r") as f:
                eigenvalues_over_time = json.load(f)
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
        if pred_vs_true_path.exists():
            with open(pred_vs_true_path, "r") as f:
                pred_vs_true = {int(k): v for k, v in json.load(f).items()}
                
    elif model_checkpoint.exists():
        # Fallback to old format
        print(f"Loading model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        
        # Load training state
        eigenvalues_path = run_dir / "eigenvalues_over_time.json"
        losses_path = run_dir / "losses.json"
        
        if eigenvalues_path.exists():
            with open(eigenvalues_path, "r") as f:
                eigenvalues_over_time = json.load(f)
            if eigenvalues_over_time:
                start_epoch = max([int(k) for k in eigenvalues_over_time.keys()])
                print(f"Resuming from epoch {start_epoch}")
        
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
                losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
                loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}
        if pred_vs_true_path.exists():
            with open(pred_vs_true_path, "r") as f:
                pred_vs_true = {int(k): v for k, v in json.load(f).items()}
    
    model.train()
    
    # Weight decay: lambda_W0 = d * T, lambda_A = N * T
    wd_W0 = d * effective_temperature
    wd_A = N * effective_temperature * chi
    

    # Log predictions every N epochs
    pred_log_interval = 10000
    
    # Large eval set for eigenvalues
    Xinf = torch.randn(3000, d, device=device)
    
    # Compute initial eigenvalues
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                # Move to CPU for eigenvalue computation
                model_cpu = model.cpu()
                Xinf_cpu = Xinf.cpu()
                eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                # Move model back to original device
                model.to(device)
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues: {e}")
                model.to(device)
    print("Initial loss")
    with torch.no_grad():
        output = model(X)
        diff = output - Y
        per_ensemble_loss = torch.sum(diff * diff, dim=0)
        loss = per_ensemble_loss.sum()
        loss_avg = loss.item() / ens
        loss_std = per_ensemble_loss.std().item()
        print(f"  loss={loss_avg:.6e}±{loss_std:.6e}")
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('loss/sum_total', loss.item(), 0)
            writer.add_scalar('loss/mean', loss_avg, 0)
            writer.add_scalar('loss/std', loss_std, 0)
        if 0 % pred_log_interval == 0 and 0 not in pred_vs_true:
            rec = _pred_record(0, Y, output)
            pred_vs_true[0] = rec
            with open(pred_vs_true_path, "w") as f:
                json.dump(pred_vs_true, f, indent=2)
    last_output = output

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        if epoch > 0:
            lr = base_lr / P
            # # 1. Define the annealing schedule
            # anneal_start = int(epochs * 0.75)
            # anneal_end = int(epochs * 0.80)  # Spend 5% of time cooling down
            # target_lr = base_lr / (3 * P)
            # current_base_lr = base_lr / P
            # if epoch == anneal_start:
            #     # Hard save model to a filename indicating start of annealing
            #     anneal_checkpoint_path = run_dir / f"checkpoint_anneal_start_epoch_{epoch}.pt"
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #     }, anneal_checkpoint_path)
            #     print(f"  Saved anneal start checkpoint to {anneal_checkpoint_path}")
            # if epoch < anneal_start:
            #     lr = current_base_lr
            # elif epoch < anneal_end:
            #     # Linear interpolation between high LR and low LR
            #     fraction = (epoch - anneal_start) / (anneal_end - anneal_start)
            #     lr = current_base_lr + fraction * (target_lr - current_base_lr)
            # else:
            #     lr = target_lr

            # 2. Re-calculate noise scale based on the smoothly changing LR
            torch.manual_seed(epoch)
            noise_scale = np.sqrt(2.0 * lr * effective_temperature)



            # Forward pass
            output = model(X)  # (P, ens)
            
            # Loss per ensemble
            diff = output - Y  # (P, ens)
            per_ensemble_loss = torch.sum(diff * diff, dim=0)  # (ens,)
            loss = per_ensemble_loss.sum() 
            
            loss_avg = loss.item() / ens
            loss_std = per_ensemble_loss.std().item()
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('loss/sum_total', loss.item(), epoch)
                writer.add_scalar('learning_rate/lr', lr, epoch)
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Langevin update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    # Weight decay
                    if 'W0' == name:
                        wd = wd_W0
                    elif 'A' == name:
                        wd = wd_A
                    else:
                        wd = 0
                    
                    # Gradient + weight decay + noise
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(noise)
                    last_output = output
        
        # Logging and checkpointing
        if epoch % log_interval == 0:
            with torch.no_grad():
                # Compute eigenvalues
                try:
                    # Move to CPU for eigenvalue computation
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    eigenvalues = model_cpu.H_eig(Xinf_cpu, Xinf_cpu).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    # Move model back to original device
                    model.to(device)
                    
                    # Log to TensorBoard
                    if writer is not None:
                        # # Individual eigenvalues
                        # eigenvalue_dict = {f'eig_{i}': float(eigenvalues[i]) 
                        #                  for i in range(len(eigenvalues))}
                        # writer.add_scalars('eigenvalues/all', eigenvalue_dict, epoch)
                        
                        # Eigenvalue statistics
                        writer.add_scalar('eigenvalues/perp_mean', float(eigenvalues[1:].mean()), epoch)
                        writer.add_scalar('eigenvalues/max', float(eigenvalues.max()), epoch)
                        writer.add_scalar('eigenvalues/min', float(eigenvalues.min()), epoch)
                        writer.add_scalar('eigenvalues/std', float(eigenvalues.std()), epoch)
                        
                        # Eigenvalue histogram
                        writer.add_histogram('eigenvalues/distribution', eigenvalues, epoch)
                        
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                    model.to(device)
                
                # Compute h0_activation projections
                try:
                    import traceback
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    
                    # Get hidden layer activations: shape (P, ens, n1)
                    h0_act = model_cpu.h0_activation(Xinf_cpu)
                    P_dim = h0_act.shape[0]
                    
                    # Compute projection directions
                    x0_target = Xinf_cpu[:, 0]
                    x0_target_normed = x0_target 
                    x1_perp = Xinf_cpu[:, 1] if d > 1 else torch.randn_like(Xinf_cpu[:, 0])
                    x1_perp_normed = x1_perp / x1_perp.norm() 
                    
                    # Hermite cubic polynomials for target and perp: (x^3 - 3x)/sqrt(6)
                    h3_target = (x0_target**3 - 3.0 * x0_target) 
                    h3_target_normed = h3_target 
                    h3_perp = (x1_perp**3 - 3.0 * x1_perp)
                    h3_perp_normed = h3_perp 
                    
                    # Project activations onto target/perp directions per ensemble
                    # h0_act shape: (P, ens, n1)
                    proj_lin_target = torch.einsum('pqn,p->qn', h0_act, x0_target_normed) / P_dim
                    proj_lin_perp = torch.einsum('pqn,p->qn', h0_act, x1_perp_normed) / P_dim
                    proj_cubic_target = torch.einsum('pqn,p->qn', h0_act, h3_target_normed)  / P_dim
                    proj_cubic_perp = torch.einsum('pqn,p->qn', h0_act, h3_perp_normed) / P_dim
                    
                    # Compute variances
                    var_lin_target = float(torch.var(proj_lin_target).item())
                    var_lin_perp = float(torch.var(proj_lin_perp).item())
                    var_cubic_target = float(torch.var(proj_cubic_target).item())
                    var_cubic_perp = float(torch.var(proj_cubic_perp).item())
                    
                    # Log variances to TensorBoard
                    if writer is not None:
                        writer.add_scalar('Projections/He1_target_var', var_lin_target, epoch)
                        writer.add_scalar('Projections/He1_perp_var', var_lin_perp, epoch)
                        writer.add_scalar('Projections/He3_target_var', var_cubic_target, epoch)
                        writer.add_scalar('Projections/He3_perp_var', var_cubic_perp, epoch)
                        
                        # Log histograms to TensorBoard
                        writer.add_histogram('Projections/He1_target', proj_lin_target, epoch)
                        writer.add_histogram('Projections/He1_perp', proj_lin_perp, epoch)
                        writer.add_histogram('Projections/He3_target', proj_cubic_target, epoch)
                        writer.add_histogram('Projections/He3_perp', proj_cubic_perp, epoch)
                        
                    print(f"  Projections - He1_target: {var_lin_target:.3g}, He1_perp: {var_lin_perp:.3g}, He3_target: {var_cubic_target:.3g}, He3_perp: {var_cubic_perp:.3g}")
                    
                    model.to(device)
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute/log projections at epoch {epoch}: {e}")
                    model.to(device)
                
                # Log loss and eigenvalues
                if epoch > 0:
                    
                    losses[epoch] = float(loss_avg)
                    loss_stds[epoch] = float(loss_std)
                    
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
                        print(f"  Epoch {epoch:7d} (init): max_eig={eigenvalues.max():.6f}, "
                              f"mean_eig={eigenvalues.mean():.6f}")

                # Save checkpoint
                if epoch % (2 * log_interval) == 0:
                    # Save model state
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    
                    # Save full checkpoint with metadata
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'd': d, 'P': P, 'N': N, 'ens': ens,
                            'lr': float(lr), 'temperature': float(temperature),
                            'chi': float(chi), 'effective_temperature': float(effective_temperature)
                        },
                        'loss': float(loss_avg) if epoch > 0 else None,
                        'loss_std': float(loss_std) if epoch > 0 else None,
                    }
                    if eigenvalues is not None:
                        checkpoint['eigenvalues'] = eigenvalues.tolist()
                    torch.save(checkpoint, run_dir / "checkpoint.pt")
                    
                    # Save eigenvalues and losses
                    with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                        json.dump(eigenvalues_over_time, f, indent=2)
                    
                    with open(run_dir / "losses.json", "w") as f:
                        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
                    with open(pred_vs_true_path, "w") as f:
                        json.dump(pred_vs_true, f, indent=2)

        if epoch % pred_log_interval == 0 and epoch not in pred_vs_true:
            with torch.no_grad():
                rec = _pred_record(epoch, Y, last_output)
                pred_vs_true[epoch] = rec
                with open(pred_vs_true_path, "w") as f:
                    json.dump(pred_vs_true, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save config
    config = {
        "d": d, "P": P, "N": N, "eps": eps,
        "lr": float(lr), "temperature": float(temperature),
        "chi": float(chi), "effective_temperature": float(temperature / chi),
        "epochs": epochs, "ens": ens
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Final eigenvalues
    final_eigenvalues = None
    if epochs in eigenvalues_over_time:
        final_eigenvalues = np.array(eigenvalues_over_time[epochs])
    
    return final_eigenvalues, eigenvalues_over_time, run_dir


def plot_eigenvalues_over_time(run_dir):
    """Plot eigenvalue evolution over training."""
    eigenvalues_path = run_dir / "eigenvalues_over_time.json"
    if not eigenvalues_path.exists():
        print(f"No eigenvalues file found at {eigenvalues_path}")
        return
    
    with open(eigenvalues_path, "r") as f:
        eig_data = json.load(f)
    
    epochs = sorted([int(k) for k in eig_data.keys()])
    eigenvalues = np.array([eig_data[str(e)] for e in epochs])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each eigenvalue over time
    for i in range(eigenvalues.shape[1]):
        ax.plot(epochs, eigenvalues[:, i], alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"H Eigenvalues over Training\n{run_dir.name}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(run_dir / "eigenvalues_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved eigenvalue plot to {run_dir / 'eigenvalues_over_time.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train 2-layer erf network')
    parser.add_argument('--d', type=int, default=10, help='Input dimension')
    parser.add_argument('--P', type=int, default=30, help='Number of samples')
    parser.add_argument('--N', type=int, default=256, help='Hidden layer width')
    parser.add_argument('--epochs', type=int, default=10_000_000, help='Number of epochs')
    parser.add_argument('--log-interval', type=int, default=10_000, help='Logging interval')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Base temperature for Langevin')
    parser.add_argument('--chi', type=float, default=1.0, help='Scale factor; effective temperature = temperature/chi')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device')
    parser.add_argument('--dataset-seed', type=int, default=42, help='Random seed for dataset generation')
    parser.add_argument('--ens', type=int, default=10, help='Ensemble size')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon for cubic nonlinearity')
    args = parser.parse_args()
    
    print("="*60)
    print("Training 2-Layer ERF Network")
    print("="*60)
    
    # Setup TensorBoard
    tensorboard_dir = Path(__file__).parent / "runs" / f"d{args.d}_P{args.P}_N{args.N}_chi_{args.chi}_seed_{args.dataset_seed}_lr_{args.lr}_T_{args.temperature}_eps_{args.eps}"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Train
    final_eigs, eigs_over_time, run_dir = train_fcn2(
        d=args.d, P=args.P, N=args.N,eps=args.eps,
        epochs=args.epochs, log_interval=args.log_interval,
        device_str=args.device, base_lr=args.lr, temperature=args.temperature, chi=args.chi,
        writer=writer, dataset_seed=args.dataset_seed, ens=args.ens
    )
    
    writer.close()
    
    # Plot
    plot_eigenvalues_over_time(run_dir)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
