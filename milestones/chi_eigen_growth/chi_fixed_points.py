"""
Train networks at fixed chi values with different training durations.
Chi values: 1, 64, 128, 192, 256
Epochs: 100K, 1M, 4M, 10M (but only for last; scale epochs linearly)
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from FCN3Network import FCN3NetworkEnsembleLinear


def train_network(
    model,
    chi,
    epochs,
    X,
    y,
    device,
    lr=1e-5,
    temperature=2.0,
    progress=5,
):
    """Train network with Langevin dynamics at fixed chi."""
    model.train()
    loss_fn = nn.MSELoss(reduction='sum')
    temperature = temperature / chi
    noise_std = (2 * lr * temperature) ** 0.5
    report_every = max(1, epochs // max(1, progress))
    last_loss = None

    for step in range(epochs):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        pred = model(X)
        # Stack y fifty times for ensemble
        ystack = y.squeeze().unsqueeze(-1).expand(-1, model.ensembles)
        data_loss = loss_fn(pred, ystack )

        total_loss = data_loss
        total_loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                grad_term = -lr * p.grad
                # Weight decay scaled by fan-in
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

        if (step + 1) % report_every == 0:
            last_loss = total_loss.item()
    if last_loss is None:
        last_loss = total_loss.item()
    return last_loss


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

    # Fixed chi values with corresponding epochs
    chi_configs = [
        (1, 100_000),
        (64, 500_000),
        (128, 2_000_000),
        (192, 4_000_000),  # Scale between last two
        (256, 10_000_000),
    ]

    base_dir = Path(__file__).resolve().parent
    net_dir = base_dir / "data" / "networks_fixed"
    res_dir = base_dir / "data" / "results_fixed"
    net_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for chi, epochs in chi_configs:
        print(f"\nTraining chi={chi} for {epochs} epochs...")

        # Initialize fresh network for each chi
        weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
        model = FCN3NetworkEnsembleLinear(
            d=d, n1=N, n2=N, P=P, ensembles=50,
            weight_initialization_variance=weight_var,
            device=device
        )

        loss_val = train_network(model, chi, epochs, X, y, device)
        lH, eigvals = compute_lH(model, d, device)

        torch.save(
            {"chi": chi, "epochs": epochs, "state_dict": model.state_dict()},
            net_dir / f"chi_{chi}_epochs_{epochs}.pth"
        )

        results.append({
            "chi": chi,
            "epochs": epochs,
            "loss": loss_val,
            "lH": lH,
            "eigenvalues": eigvals,
        })
        print(f"chi={chi:>3} | epochs={epochs:>10} | loss={loss_val:.6f} | lH={lH:.6f}")

    out_path = res_dir / "chi_fixed_eigenvalues.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
