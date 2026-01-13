#!/usr/bin/env python3
"""
Training script for 2-layer erf data-averaged network with eigenvalue tracking.

Trains a 2-layer network with dataset-averaged ensembles:
Input(d) -> Hidden(n1) with erf -> Output(1), with weights indexed by
(num_datasets, num_ensembles). Tracks data-averaged H kernel eigenvalues.

Usage:
    python train_fcn2_data_averaged.py --d 50 --P 200 --N 200 \
        --datasets 20 --ensembles 3 --epochs 30000000 --lr 5e-6 --device cuda:0
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter

# Set default dtype
torch.set_default_dtype(torch.float32)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from DataAveragedNetworks import FCN2NetworkDataAveragedEnsemble


def _pred_record(epoch, targets, outputs):
    """Create a serializable record of predictions and alignment stats.
    Uses the mean over datasets and ensembles for prediction.
    """
    y_true = targets.squeeze(-1).detach().cpu().numpy()
    y_pred = outputs.mean(dim=(1, 2)).detach().cpu().numpy()
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


def train_fcn2_data_averaged(d, P, N, datasets, ensembles,
                              epochs=30_000_000, log_interval=10_000,
                              device_str="cuda:0", lr=5e-6, temperature=1.0, chi=None,
                              run_dir=None, writer=None):
    """Train 2-layer erf data-averaged network and track H eigenvalues.

    Args:
        d: Input dimension
        P: Number of training samples
        N: Hidden layer width
        datasets: Number of datasets (D)
        ensembles: Number of ensemble members (Q)
        epochs: Training iterations
        log_interval: Log every N epochs
        device_str: Device string
        lr: Learning rate
        temperature: Base temperature for weight decay and Langevin noise
        chi: Scale factor; effective temperature = temperature / chi. If None, uses N.
        run_dir: Directory to save outputs
        writer: TensorBoard writer

    Returns:
        (final_eigenvalues, eigenvalues_over_time, run_dir)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if chi is None:
        chi = float(N)
    effective_temperature = temperature / chi

    # Setup directory
    if run_dir is None:
        run_dir = Path(__file__).parent / (
            f"dataavg_d{d}_P{P}_N{N}_D{datasets}_Q{ensembles}_chi_{int(chi)}_lr_{lr}_T_{temperature}"
        )
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nTraining 2-layer erf data-averaged network:")
    print(f"  d={d}, P={P}, N={N}, D={datasets}, Q={ensembles}")
    print(f"  lr={lr:.6e}, T={temperature:.6f}, chi={chi:.6f}, T_eff={effective_temperature:.6f}")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")

    # Data: same X used across datasets; target uses first coordinate
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0].unsqueeze(-1)  # (P, 1)

    # Model
    model = FCN2NetworkDataAveragedEnsemble(
        d=d, n1=N, P=P,
        num_datasets=datasets, num_ensembles=ensembles,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    ).to(device)

    # Checkpoints and histories
    checkpoint_path = run_dir / "checkpoint.pt"
    model_checkpoint = run_dir / "model.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}
    pred_vs_true = {}
    pred_vs_true_path = run_dir / "pred_vs_true.json"

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
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
        print(f"Loading model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
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

    # Weight decay: lambda_W0 = d * T_eff, lambda_A = N * T_eff * chi
    wd_W0 = d * effective_temperature
    wd_A = N * effective_temperature * chi

    # Langevin noise scale
    noise_scale = np.sqrt(2.0 * lr * effective_temperature)

    # Log predictions every N epochs
    pred_log_interval = 10000

    # Large eval set for eigenvalues
    Xinf = torch.randn(3000, d, device=device)

    # Initial eigenvalues
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                model_cpu = model.cpu()
                Xinf_cpu = Xinf.cpu()
                eigenvalues = model_cpu.H_eig_data_averaged(Xinf_cpu, Xinf_cpu).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                model.to(device)
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues: {e}")
                model.to(device)

    # Initial loss and pred record
    with torch.no_grad():
        output = model(X)  # (P, D, Q)
        diff = output - Y[:, None, None]
        per_dq_loss = torch.sum(diff * diff, dim=0)  # (D, Q)
        loss_total = per_dq_loss.sum()
        loss_avg = loss_total.item() / (datasets * ensembles)
        loss_std = per_dq_loss.std().item()
        print(f"  loss={loss_avg:.6e}±{loss_std:.6e}")
        if writer is not None:
            writer.add_scalar('loss/sum_total', loss_total.item(), 0)
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
            torch.manual_seed(epoch)
            output = model(X)
            diff = output - Y[:, None, None]
            per_dq_loss = torch.sum(diff * diff, dim=0)
            loss = per_dq_loss.sum()
            loss_avg = loss.item() / (datasets * ensembles)
            loss_std = per_dq_loss.std().item()
            if writer is not None:
                writer.add_scalar('loss/sum_total', loss.item(), epoch)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if 'W0' == name:
                        wd = wd_W0
                    elif 'A' == name:
                        wd = wd_A
                    else:
                        wd = 0
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(noise)
                    last_output = output

        if epoch % log_interval == 0:
            with torch.no_grad():
                try:
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    eigenvalues = model_cpu.H_eig_data_averaged(Xinf_cpu, Xinf_cpu).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    model.to(device)
                    if writer is not None:
                        eig_dict = {f'eig_{i}': float(eigenvalues[i]) for i in range(len(eigenvalues))}
                        writer.add_scalars('eigenvalues/all', eig_dict, epoch)
                        writer.add_scalar('eigenvalues/mean', float(eigenvalues.mean()), epoch)
                        writer.add_scalar('eigenvalues/max', float(eigenvalues.max()), epoch)
                        writer.add_scalar('eigenvalues/min', float(eigenvalues.min()), epoch)
                        writer.add_scalar('eigenvalues/std', float(eigenvalues.std()), epoch)
                        writer.add_histogram('eigenvalues/distribution', eigenvalues, epoch)
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                    model.to(device)

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

                if epoch > 0:
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'd': d, 'P': P, 'N': N,
                            'datasets': datasets, 'ensembles': ensembles,
                            'lr': float(lr), 'temperature': float(temperature),
                            'chi': float(chi), 'effective_temperature': float(effective_temperature)
                        },
                        'loss': float(loss_avg) if epoch > 0 else None,
                        'loss_std': float(loss_std) if epoch > 0 else None,
                    }
                    if eigenvalues is not None:
                        checkpoint['eigenvalues'] = eigenvalues.tolist()
                    torch.save(checkpoint, run_dir / "checkpoint.pt")
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

    torch.save(model.state_dict(), run_dir / "model_final.pt")
    config = {
        "d": d, "P": P, "N": N,
        "datasets": datasets, "ensembles": ensembles,
        "lr": float(lr), "temperature": float(temperature),
        "chi": float(chi), "effective_temperature": float(temperature / chi),
        "epochs": epochs
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    final_eigenvalues = None
    if epochs in eigenvalues_over_time:
        final_eigenvalues = np.array(eigenvalues_over_time[epochs])

    return final_eigenvalues, eigenvalues_over_time, run_dir


def plot_eigenvalues_over_time(run_dir):
    import matplotlib.pyplot as plt
    eigenvalues_path = run_dir / "eigenvalues_over_time.json"
    if not eigenvalues_path.exists():
        print(f"No eigenvalues file found at {eigenvalues_path}")
        return
    with open(eigenvalues_path, "r") as f:
        eig_data = json.load(f)
    epochs = sorted([int(k) for k in eig_data.keys()])
    import numpy as np
    eigenvalues = np.array([eig_data[str(e)] for e in epochs])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(eigenvalues.shape[1]):
        ax.plot(epochs, eigenvalues[:, i], alpha=0.6, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Data-Avg H Eigenvalues over Training\n{run_dir.name}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "eigenvalues_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved eigenvalue plot to {run_dir / 'eigenvalues_over_time.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train 2-layer erf data-averaged network')
    parser.add_argument('--d', type=int, default=50, help='Input dimension')
    parser.add_argument('--P', type=int, default=200, help='Number of samples')
    parser.add_argument('--N', type=int, default=200, help='Hidden layer width')
    parser.add_argument('--datasets', type=int, default=20, help='Number of datasets')
    parser.add_argument('--ensembles', type=int, default=3, help='Number of ensembles')
    parser.add_argument('--epochs', type=int, default=30_000_000, help='Number of epochs')
    parser.add_argument('--log-interval', type=int, default=10_000, help='Logging interval')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Base temperature for Langevin')
    parser.add_argument('--chi', type=float, default=None, help='Scale factor; default N if None')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()

    print("="*60)
    print("Training 2-Layer ERF Data-Averaged Network")
    print("="*60)

    tensorboard_dir = Path(__file__).parent / "runs" / (
        f"dataavg_d{args.d}_P{args.P}_N{args.N}_D{args.datasets}_Q{args.ensembles}_chi_{args.chi or args.N}"
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logging to: {tensorboard_dir}")

    final_eigs, eigs_over_time, run_dir = train_fcn2_data_averaged(
        d=args.d, P=args.P, N=args.N,
        datasets=args.datasets, ensembles=args.ensembles,
        epochs=args.epochs, log_interval=args.log_interval,
        device_str=args.device, lr=args.lr, temperature=args.temperature, chi=args.chi,
        writer=writer
    )

    writer.close()
    plot_eigenvalues_over_time(run_dir)
    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
