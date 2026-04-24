#!/usr/bin/env python3
"""
Train a single FCN2 (erf) network with MF scaling.

Usage:
    python d_sweep_fcn2_erf.py --P 1200 --d 10 --N 800 --chi 800 --kappa 0.1 --lr 1e-3 --epochs 3000000 --device cuda:0
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


torch.set_default_dtype(torch.float32)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lib"))

from FCN2Network import FCN2NetworkActivationGeneric


def make_target(X: torch.Tensor, eps: float) -> torch.Tensor:
    x0 = X[:, 0].unsqueeze(-1)
    return x0 + eps * (x0**3 - 3.0 * x0)


def save_checkpoint(seed_dir: Path, model: torch.nn.Module, cfg: dict, losses: dict, loss_stds: dict) -> None:
    torch.save(model.state_dict(), seed_dir / "model.pt")
    with open(seed_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(seed_dir / "losses.json", "w") as f:
        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)


def train_and_track(
    d: int,
    P: int,
    N: int,
    chi: float,
    kappa: float,
    lr0: float,
    epochs: int,
    device_str: str,
    eps: float = 0.03,
    seed: int = 42,
    ens: int = 10,
    log_interval: int = 5000,
    save_interval: int = 50000,
    to: str = "results_fcn2_erf",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    lr = lr0 / P
    temperature = 2.0 * kappa / chi

    base_name = f"d{d}_P{P}_N{N}_chi{chi}_kappa{kappa}_eps_{eps}"
    run_dir = Path(__file__).resolve().parent / to / base_name
    seed_dir = run_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir / f"seed{seed}"))

    print(f"\\nTraining FCN2-erf: d={d}, P={P}, N={N}, chi={chi}, kappa={kappa}, lr={lr:.3e}, ens={ens}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    print(f"TensorBoard: {tensorboard_dir / f'seed{seed}'}")

    torch.manual_seed(seed)
    X = torch.randn(P, d, device=device)
    Y = make_target(X, eps)

    torch.manual_seed(70)
    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=P,
        ens=ens,
        activation="erf",
        weight_initialization_variance=(1.0 / d, 1.0 / (N * chi)),
        device=device,
    ).to(device)

    model_checkpoint = seed_dir / "model.pt"
    if not model_checkpoint.exists():
        model_checkpoint = seed_dir / "model_final.pt"
    config_path = seed_dir / "config.json"

    start_epoch = 0
    losses = {}
    loss_stds = {}
    if model_checkpoint.exists() and config_path.exists():
        print(f"Loading existing model from {model_checkpoint}")
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        with open(config_path, "r") as f:
            cfg = json.load(f)
        start_epoch = int(cfg.get("current_epoch", 0))
        lr = float(cfg.get("lr", lr))
        print(f"Resuming from epoch {start_epoch}")

        losses_path = seed_dir / "losses.json"
        if losses_path.exists():
            with open(losses_path, "r") as f:
                loss_data = json.load(f)
            losses = {int(k): v for k, v in loss_data.get("losses", {}).items()}
            loss_stds = {int(k): v for k, v in loss_data.get("loss_stds", {}).items()}

    if losses:
        for logged_epoch in sorted(losses.keys()):
            writer.add_scalar("Loss/mean", float(losses[logged_epoch]), int(logged_epoch))
            if logged_epoch in loss_stds:
                writer.add_scalar("Loss/std", float(loss_stds[logged_epoch]), int(logged_epoch))

    model.train()
    Xinf = torch.randn(3000, d, device=device)

    wd_w0 = d * temperature
    wd_a = N * temperature * chi

    for epoch in range(start_epoch, epochs + 1):
        if epoch > 0:
            torch.manual_seed(7 + epoch)
            noise_scale = np.sqrt(2.0 * lr * temperature)

            output = model(X)
            diff = output - Y
            per_ensemble_loss = torch.sum(diff * diff, dim=0)
            loss = per_ensemble_loss.sum()
            loss_avg = loss.item() / model.ensembles
            loss_std = per_ensemble_loss.std().item()

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name == "W0":
                        wd = wd_w0
                    elif name == "A":
                        wd = wd_a
                    else:
                        wd = 0.0
                    param.add_(-lr * param.grad)
                    param.add_(-lr * wd * param.data)
                    param.add_(torch.randn_like(param) * noise_scale)

            if epoch % log_interval == 0:
                losses[epoch] = float(loss_avg)
                loss_stds[epoch] = float(loss_std)
                writer.add_scalar("Loss/mean", float(loss_avg), epoch)
                writer.add_scalar("Loss/std", float(loss_std), epoch)
                try:
                    eigenvalues = model.H_eig(Xinf, Xinf).cpu().numpy()
                    writer.add_scalar("Eigenvalues/max", eigenvalues.max(), epoch)
                    writer.add_scalar("Eigenvalues/mean", eigenvalues[1:].mean(), epoch)
                except Exception as e:
                    traceback.print_exc()
                    print(f"  Warning: Could not compute eigenvalues at epoch {epoch}: {e}")
                print(f"  Epoch {epoch:8d}: loss={loss_avg:.6e} +/- {loss_std:.6e}")

            if epoch % save_interval == 0:
                cfg = {
                    "model": "FCN2-erf",
                    "d": d,
                    "P": P,
                    "N": N,
                    "chi": chi,
                    "kappa": kappa,
                    "lr": lr,
                    "lr0": lr0,
                    "epochs": epochs,
                    "eps": eps,
                    "seed": seed,
                    "ens": ens,
                    "current_epoch": epoch,
                    "activation": "erf",
                }
                save_checkpoint(seed_dir, model, cfg, losses, loss_stds)

    torch.save(model.state_dict(), seed_dir / "model_final.pt")
    final_cfg = {
        "model": "FCN2-erf",
        "d": d,
        "P": P,
        "N": N,
        "chi": chi,
        "kappa": kappa,
        "lr": lr,
        "lr0": lr0,
        "epochs": epochs,
        "eps": eps,
        "seed": seed,
        "ens": ens,
        "current_epoch": epochs,
        "activation": "erf",
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(final_cfg, f, indent=2)
    with open(seed_dir / "losses.json", "w") as f:
        json.dump({"losses": losses, "loss_stds": loss_stds}, f, indent=2)

    writer.close()

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single FCN2-erf network with MF scaling")
    parser.add_argument("--P", type=int, required=True, help="Dataset size")
    parser.add_argument("--d", type=int, required=True, help="Input dimension")
    parser.add_argument("--N", type=int, required=True, help="Hidden width")
    parser.add_argument("--chi", type=float, required=True, help="MF chi parameter")
    parser.add_argument("--kappa", type=float, required=True, help="Kappa parameter")
    parser.add_argument("--lr", type=float, required=True, help="Base LR (effective LR is lr/P)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--device", type=str, required=True, help="Compute device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed")
    parser.add_argument("--ens", type=int, default=10, help="Ensemble size")
    parser.add_argument("--eps", type=float, default=0.03, help="Cubic target coefficient")
    parser.add_argument("--to", type=str, default="results_fcn2_erf", help="Output directory name")
    parser.add_argument("--dry-run", action="store_true", help="Run one epoch and remove outputs")
    args = parser.parse_args()

    epochs = 1 if args.dry_run else args.epochs
    run_dir = train_and_track(
        d=args.d,
        P=args.P,
        N=args.N,
        chi=args.chi,
        kappa=args.kappa,
        lr0=args.lr,
        epochs=epochs,
        device_str=args.device,
        seed=args.seed,
        ens=args.ens,
        eps=args.eps,
        to=args.to,
    )

    if args.dry_run:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
