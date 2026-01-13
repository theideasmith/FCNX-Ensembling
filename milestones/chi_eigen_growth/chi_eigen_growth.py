import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from FCN3Network import FCN3NetworkEnsembleLinear


def epoch_schedule(chi_values):
    """Linearly scale epochs from 10K at chi=1 to 4M at chi=256."""
    return {
        chi: int(round(np.interp(chi, [1, 256], [10_000, 4_000_000])))
        for chi in chi_values
    }


def train_for_chi(
    model,
    chi,
    epochs,
    X,
    y,
    device,
    lr=1e-4,
    temperature=2.0,
    progress=5,
):
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
        data_loss = loss_fn(pred, y)  # chi multiplies target

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
                    decay_term = -lr * temperature * p * model.n2 * chi # chi scaling for readout
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
    """Compute H kernel eigenvalues using FCN3NetworkEnsembleLinear.H_eig() on large 1000xd matrix."""
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

    # Chi schedule: start at 1, increment by 1 up to 256
    chi_values = list(range(1, 257))
    chi_epochs = epoch_schedule(chi_values)

    base_dir = Path(__file__).resolve().parent
    net_dir = base_dir / "data" / "networks"
    res_dir = base_dir / "data" / "results"
    net_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Initialize FCN3NetworkEnsembleLinear with 1/fan_in weight variance
    weight_var = (1.0 / d, 1.0 / N, 1.0 / N)
    model = FCN3NetworkEnsembleLinear(
        d=d, n1=N, n2=N, P=P, ensembles=1,
        weight_initialization_variance=weight_var,
        device=device
    )

    results = []
    for chi in chi_values:
        epochs = chi_epochs[chi]
        loss_val = train_for_chi(model, chi, epochs, X, y, device)
        lH, eigvals = compute_lH(model, d, device)

        torch.save({"chi": chi, "state_dict": model.state_dict()}, net_dir / f"chi_{chi}.pth")

        results.append({
            "chi": chi,
            "epochs": epochs,
            "loss": loss_val,
            "lH": lH,
            "eigenvalues": eigvals,
        })
        print(f"chi={chi:>3} | epochs={epochs:>8} | loss={loss_val:.6f} | lH={lH:.6f}")

    out_path = res_dir / "chi_eigenvalues.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
