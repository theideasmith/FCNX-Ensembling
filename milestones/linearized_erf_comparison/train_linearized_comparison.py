#!/usr/bin/env python3
"""
Training script for linearized erf comparison: erf vs linear networks.

Trains FCN2 data-averaged networks with:
- erf activation with sigma^2_w = 4/pi * sigma^2_w,lin
- linear activation with sigma^2_w = sigma^2_w,lin

For small weight variance, erf(x) ≈ x in the linearized regime.

Usage:
    python train_linearized_comparison.py \
        --d 2 --activation erf --chi 1.0 --device cuda:1
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


def train_linearized_network(d, activation, chi, datasets=50, ensembles=4,
                              epochs=4_000_000, log_interval=10_000,
                              device_str="cuda:1", lr=5e-6, kappa=0.1,
                              sigma_w_lin=1.0, sigma_A_lin=1.0, run_dir=None):
    """Train FCN2 data-averaged network in linearized regime.

    Args:
        d: Input dimension
        activation: "erf" or "linear"
        chi: Scale factor for MF/SSC (1.0 = SSC, N = MF)
        datasets: Number of datasets (D=50)
        ensembles: Number of ensemble members (Q=4)
        epochs: Training iterations
        log_interval: Log every N epochs
        device_str: Device string
        lr: Learning rate
        kappa: Effective temperature parameter (T_eff = kappa)
        sigma_w_lin: Base weight variance for linearized regime
        run_dir: Directory to save outputs

    Returns:
        run_dir
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Parameters
    P = 4 * d
    N = 200  # For chi scaling
    temperature = 2.0 * kappa

    # Weight initialization variance
    if activation == "erf":
        # In linearized regime: sigma^2_w = 4/pi * sigma^2_w,lin
        sigma_w_sq = (3 * np.pi / 4.0) * sigma_w_lin
        sigma_A_sq = 1.0
    else:  # linear
        sigma_w_sq = sigma_w_lin
        sigma_A_sq = sigma_A_lin

    # Setup directory
    if run_dir is None:
        scaling = "ssc" if chi == 1.0 else f"mf{int(chi)}"
        run_dir = Path(__file__).parent / "runs" / (
            f"{activation}_d{d}_P{P}_N{N}_D{datasets}_Q{ensembles}_{scaling}_lr_{lr:.0e}_kappa_{kappa}"
        )
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nTraining linearized {activation} network:")
    print(f"  d={d}, P={P}, N={N}, D={datasets}, Q={ensembles}")
    print(f"  activation={activation}, chi={chi}, kappa={kappa:.4f}")
    print(f"  sigma_w_lin={sigma_w_lin}, sigma_w_sq={sigma_w_sq:.6f}")
    print(f"  lr={lr:.6e}, T={temperature:.6f}, T={temperature:.6f}")
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
        activation=activation,
        weight_initialization_variance=(sigma_w_sq / d, sigma_A_sq / (chi * N)),
        device=device
    ).to(device)

    # Checkpoints and histories
    checkpoint_path = run_dir / "checkpoint.pt"
    start_epoch = 0
    eigenvalues_over_time = {}
    losses = {}
    loss_stds = {}

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

    model.train()

    # Weight decay: lambda_W0 = d * T_eff, lambda_A = N * T_eff * chi
    wd_W0 = d * temperature * sigma_w_sq
    wd_A = N * temperature * chi  * sigma_A_sq

    # Langevin noise scale
    noise_scale = np.sqrt(2.0 * lr * temperature)

    # Large eval set for eigenvalues
    Xinf = torch.randn(3000, d, device=device)

    # Initial eigenvalues
    if start_epoch == 0 and 0 not in eigenvalues_over_time:
        with torch.no_grad():
            try:
                model_cpu = model.cpu()
                Xinf_cpu = Xinf.cpu()
                model_cpu.device = torch.device("cpu")
                eigenvalues = model_cpu.H_eig_data_averaged(Xinf_cpu, Xinf_cpu).cpu().numpy()
                eigenvalues_over_time[0] = eigenvalues.tolist()
                print(f"  Epoch {0:7d} (init): max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                model.to(device)
            except Exception as e:
                print(f"  Warning: Could not compute initial eigenvalues: {e}")
                model.to(device)

    # Initial loss
    with torch.no_grad():
        output = model(X)  # (P, D, Q)
        diff = output - Y[:, None, None]
        per_dq_loss = torch.sum(diff * diff, dim=0)  # (D, Q)
        loss_total = per_dq_loss.sum()
        loss_avg = loss_total.item() / (datasets * ensembles)
        loss_std = per_dq_loss.std().item()
        print(f"  Initial loss={loss_avg:.6e}±{loss_std:.6e}")

    # Training loop
    for epoch in range(start_epoch + 1, epochs + 1):
        torch.manual_seed(epoch)
        output = model(X)
        diff = output - Y[:, None, None]
        per_dq_loss = torch.sum(diff * diff, dim=0)
        loss = per_dq_loss.sum()
        loss_avg = loss.item() / (datasets * ensembles)
        loss_std = per_dq_loss.std().item()

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

        if epoch % log_interval == 0:
            with torch.no_grad():
                try:
                    model_cpu = model.cpu()
                    Xinf_cpu = Xinf.cpu()
                    eigenvalues = model_cpu.H_eig_data_averaged(Xinf_cpu, Xinf_cpu).cpu().numpy()
                    eigenvalues_over_time[epoch] = eigenvalues.tolist()
                    model.to(device)
                except Exception as e:
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                    eigenvalues = None
                    model.to(device)

                losses[epoch] = float(loss_avg)
                loss_stds[epoch] = float(loss_std)

                if eigenvalues is not None:
                    print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}, "
                          f"max_eig={eigenvalues.max():.6f}, mean_eig={eigenvalues.mean():.6f}")
                else:
                    print(f"  Epoch {epoch:7d}: loss={loss_avg:.6e}±{loss_std:.6e}")

                # Save checkpoint
                torch.save(model.state_dict(), run_dir / "model.pt")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'd': d, 'P': P, 'N': N,
                        'datasets': datasets, 'ensembles': ensembles,
                        'activation': activation,
                        'chi': float(chi),
                        'kappa': float(kappa),
                        'lr': float(lr),
                        'temperature': float(temperature),
                        'sigma_w_lin': float(sigma_w_lin),
                        'sigma_w_sq': float(sigma_w_sq),
                    },
                    'loss': float(loss_avg),
                    'loss_std': float(loss_std),
                }
                if eigenvalues is not None:
                    checkpoint['eigenvalues'] = eigenvalues.tolist()
                torch.save(checkpoint, checkpoint_path)

                with open(run_dir / "eigenvalues_over_time.json", "w") as f:
                    json.dump(eigenvalues_over_time, f, indent=2)
                with open(run_dir / "losses.json", "w") as f:
                    json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)
                with open(run_dir / "config.json", "w") as f:
                    json.dump(checkpoint['config'], f, indent=2)
    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    print(f"\nTraining complete: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train linearized erf/linear comparison")
    parser.add_argument("--d", type=int, required=True, help="Input dimension")
    parser.add_argument("--activation", type=str, required=True, choices=["erf", "linear"],
                        help="Activation function")
    parser.add_argument("--chi", type=float, required=True, help="Scaling factor (1.0=SSC, N=MF)")
    parser.add_argument("--datasets", type=int, default=50, help="Number of datasets (D)")
    parser.add_argument("--ensembles", type=int, default=4, help="Number of ensemble members (Q)")
    parser.add_argument("--epochs", type=int, default=4_000_000, help="Training epochs")
    parser.add_argument("--log-interval", type=int, default=10_000, help="Log every N epochs")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--kappa", type=float, default=0.1, help="Effective temperature parameter")
    parser.add_argument("--sigma-w-lin", type=float, default=1.0, help="Base weight variance")
    args = parser.parse_args()

    train_linearized_network(
        d=args.d,
        activation=args.activation,
        chi=args.chi,
        datasets=args.datasets,
        ensembles=args.ensembles,
        epochs=args.epochs,
        log_interval=args.log_interval,
        device_str=args.device,
        lr=args.lr,
        kappa=args.kappa,
        sigma_w_lin=args.sigma_w_lin,
    )


if __name__ == "__main__":
    main()
