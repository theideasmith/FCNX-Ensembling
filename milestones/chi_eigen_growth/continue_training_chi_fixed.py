"""
Continue training networks at fixed chi values, saving eigenvalues periodically.

This script:
1. Loads trained networks from chi_fixed_points.py
2. Continues training for 2x the original epochs
3. Saves eigenvalues every checkpoint_interval epochs
4. Stores results indexed by (chi, total_epoch)
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from FCN3Network import FCN3NetworkEnsembleLinear


def train_network_checkpoint(
    model,
    chi,
    X,
    y,
    device,
    num_checkpoints=20,
    lr=1e-5,
    temperature=2.0,
):
    """
    Continue training network, saving eigenvalues at checkpoints.
    
    Args:
        model: Pre-trained network (already partially trained)
        chi: Chi parameter
        X, y: Training data
        device: Device to train on
        num_checkpoints: Number of checkpoints to save
        lr: Learning rate
        temperature: Temperature parameter
    
    Returns:
        List of dicts with (epoch, lH, eigenvalues) at each checkpoint
    """
    model.train()
    loss_fn = nn.MSELoss(reduction='sum')
    temperature = temperature / chi
    noise_std = (2 * lr * temperature) ** 0.5
    
    checkpoint_results = []
    
    for checkpoint in range(num_checkpoints):
        for step in range(100_000):  # 100K steps per checkpoint
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            pred = model(X)
            ystack = y.squeeze().unsqueeze(-1).expand(-1, model.ensembles)
            data_loss = loss_fn(pred, ystack)
            total_loss = data_loss
            total_loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    grad_term = -lr * p.grad
                    if p is model.W0:
                        decay_term = -lr * temperature * p * model.d
                    elif p is model.W1:
                        decay_term = -lr * temperature * p * model.n1
                    elif p is model.A:
                        decay_term = -lr * temperature * p * model.n2 * chi
                    else:
                        decay_term = 0
                    noise = noise_std * torch.randn_like(p)
                    p.add_(grad_term + decay_term + noise)
        
        # Save checkpoint
        total_epoch = (checkpoint + 1) * 100_000
        lH, eigvals = compute_lH(model, model.d, device)
        
        checkpoint_results.append({
            "epoch": total_epoch,
            "lH": lH,
            "eigenvalues": eigvals,
        })
        
        print(f"  Checkpoint {checkpoint + 1}/{num_checkpoints} (epoch {total_epoch}): lH={lH:.6f}")
    
    return checkpoint_results


def compute_lH(model, d, device):
    """Compute H kernel eigenvalues on large 1000xd matrix."""
    model.eval()
    with torch.no_grad():
        X_large = torch.randn(1000, d, device=device)
        eigs = model.H_eig(X_large, X_large)
        lH = float(torch.max(eigs))
        return lH, eigs.cpu().tolist()


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # Fixed data: d=2, P=6
    d, P, N = 2, 6, 256
    X = torch.randn(P, d, device=device)
    y = X[:, 0].unsqueeze(-1).to(device)

    # Chi configs from chi_fixed_points.py
    chi_configs = [
        (1, 100_000),
        (64, 500_000),
        (128, 2_000_000),
        (192, 4_000_000),
        (256, 10_000_000),
    ]

    base_dir = Path(__file__).resolve().parent
    net_dir = base_dir / "data" / "networks_fixed"
    checkpoint_dir = base_dir / "data" / "checkpoints_fixed"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for chi, original_epochs in chi_configs:
        print(f"\nContinuing training chi={chi} (originally {original_epochs} epochs)...")
        
        # Load pre-trained network
        checkpoint_path = net_dir / f"chi_{chi}_epochs_{original_epochs}.pth"
        if not checkpoint_path.exists():
            print(f"  ERROR: Network file not found: {checkpoint_path}")
            continue
        
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
        model = FCN3NetworkEnsembleLinear(
            d=d, n1=N, n2=N, P=P, ensembles=50,
            weight_initialization_variance=weight_var,
            device=device
        )
        model.load_state_dict(checkpoint_data["state_dict"])
        
        # Continue training for 2x the original epochs
        # Save checkpoints every 100K epochs
        num_checkpoints = (original_epochs * 4) // 100_000
        
        results = train_network_checkpoint(
            model, chi, X, y, device,
            num_checkpoints=num_checkpoints,
            lr=1e-5,
            temperature=2.0
        )
        
        # Save checkpoint results
        out_path = checkpoint_dir / f"chi_{chi}_continued.json"
        with out_path.open("w") as f:
            json.dump({
                "chi": chi,
                "original_epochs": original_epochs,
                "checkpoints": results,
            }, f, indent=2)
        
        print(f"  Saved checkpoint results to {out_path}")


if __name__ == "__main__":
    main()
