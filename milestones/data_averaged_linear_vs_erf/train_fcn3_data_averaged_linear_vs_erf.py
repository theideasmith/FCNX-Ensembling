"""
Train FCN3 data-averaged ensembles in parallel for linear and nearly-linear erf activations
via Langevin dynamics, then compute data-averaged eigenvalues via random SVD.

Setup
- Dimensions: 10, 12, 14, 16
- Ensembles: 3 per dataset
- Datasets: 10 per dimension
- Samples: 256 per dataset
- Widths: n1 = n2 = 128
- Erf model uses tiny weight std so activations stay in the linear regime.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

import standard_hyperparams as hp
from lib.DataAveragedNetworks import FCN3NetworkDataAveragedEnsemble

# ------------------------------
# Configuration
# ------------------------------
CONFIG = {
    "dimensions": [10, 12, 14, 16],
    "num_datasets": 10,
    "num_ensembles": 3,
    "num_samples": 50,
    "n1": 128,
    "n2": 128,
    "langevin_steps": 50_000,
    "lr": 5e-3,
    "temperature": 1e-3,
    "weight_decay": 1e-4,
    "noise_std": 0.01,
    "teacher_weight_scale": 0.7,
    "linear_weight_var": (1e-2, 1e-2, 1e-2),
    "erf_linearized_weight_var": (1e-4, 1e-4, 1e-4),
    "eig_k": 64,
    "eig_p": 16,
    "seed": 42,
}

DEVICE = hp.DEVICE


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data(d: int, num_samples: int, num_datasets: int, weight_scale: float, noise_std: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Gaussian inputs and dataset-specific linear targets."""
    X = torch.randn(num_samples, d, device=device)
    teacher_w = weight_scale * torch.randn(num_datasets, d, device=device)
    y = X @ teacher_w.T  # (P, num_datasets)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return X, y


def train_model_langevin(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    temperature: float,
    weight_decay: float,
) -> None:
    """Langevin dynamics: θ <- θ - lr*(∇L + λθ) + sqrt(2 lr T) ξ."""
    model.train()
    mse = nn.MSELoss(reduction='mean')
    target = y.unsqueeze(-1)  # (P, D, 1), broadcasts over ensembles
    noise_std = (2 * lr * temperature) ** 0.5

    for _ in range(steps):
        for p in model.parameters():
            p.grad = None

        out = model(X)
        loss = mse(out, target)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                noise = noise_std * torch.randn_like(p)
                p.add_(-lr * (p.grad + weight_decay * p) + noise)


def compute_data_averaged_eigs(model: FCN3NetworkDataAveragedEnsemble, X: torch.Tensor, k: int, p: int):
    model.eval()
    return model.H_eig_random_svd_data_averaged(X, k=k, p=p).detach().cpu()


def run_for_dimension(d: int, cfg: Dict) -> Dict:
    X, y = generate_data(
        d=d,
        num_samples=cfg["num_samples"],
        num_datasets=cfg["num_datasets"],
        weight_scale=cfg["teacher_weight_scale"],
        noise_std=cfg["noise_std"],
        device=DEVICE,
    )

    common_kwargs = dict(
        d=d,
        n1=cfg["n1"],
        n2=cfg["n2"],
        P=cfg["num_samples"],
        num_datasets=cfg["num_datasets"],
        num_ensembles=cfg["num_ensembles"],
        device=DEVICE,
    )

    model_linear = FCN3NetworkDataAveragedEnsemble(
        activation="linear",
        weight_initialization_variance=cfg["linear_weight_var"],
        **common_kwargs,
    )
    model_erf_lin = FCN3NetworkDataAveragedEnsemble(
        activation="erf",
        weight_initialization_variance=cfg["erf_linearized_weight_var"],
        **common_kwargs,
    )

    train_model_langevin(
        model_linear,
        X,
        y,
        steps=cfg["langevin_steps"],
        lr=cfg["lr"],
        temperature=cfg["temperature"],
        weight_decay=cfg["weight_decay"],
    )
    train_model_langevin(
        model_erf_lin,
        X,
        y,
        steps=cfg["langevin_steps"],
        lr=cfg["lr"],
        temperature=cfg["temperature"],
        weight_decay=cfg["weight_decay"],
    )

    eig_linear = compute_data_averaged_eigs(model_linear, X, k=cfg["eig_k"], p=cfg["eig_p"])
    eig_erf_lin = compute_data_averaged_eigs(model_erf_lin, X, k=cfg["eig_k"], p=cfg["eig_p"])

    return {
        "linear": eig_linear.tolist(),
        "erf_nearly_linear": eig_erf_lin.tolist(),
    }


def main():
    set_seed(CONFIG["seed"])
    results = {}
    for d in CONFIG["dimensions"]:
        results[d] = run_for_dimension(d, CONFIG)
        print(f"Finished dimension {d}")

    out_dir = Path(__file__).parent
    out_file = out_dir / "data_averaged_linear_vs_erf_eigs.json"
    payload = {"config": CONFIG, "results": results}
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Saved eigenvalues to {out_file}")


if __name__ == "__main__":
    main()
