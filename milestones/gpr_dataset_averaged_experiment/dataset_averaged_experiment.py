#!/usr/bin/env python3
"""
GPR dataset-averaged experiment:
- Train 20 networks with N=256, ensemble size=15, kappa=5.
- Different dataset seeds for each network; fixed eval set for aggregation.
- Plot model vs GPR with error bars showing std across datasets and ensemble members.
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Defaults
torch.set_default_dtype(torch.float32)

# Library path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleLinear
from GPKit import gpr_dot_product_explicit

def custom_mse_loss(outputs, targets):
    diff = outputs - targets
    return torch.sum(diff * diff)

def train_one(seed, d, P, N, ens, kappa, epochs, lr, temperature, chi, X_eval):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0:1]  # simple linear target
    sigma0_sq = kappa

    model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=ens,
                                      weight_initialization_variance=(1/d, 1/N, 1/N)).to(device)
    model.train()

    wd_fc1 = d * temperature
    wd_fc2 = N * temperature
    wd_fc3 = N * temperature * chi
    noise_scale = np.sqrt(2.0 * lr * temperature)

    for epoch in range(epochs):
        output = model(X)
        diff = output - Y
        per_ens_loss = torch.sum(diff * diff, dim=0)  # (ens,)
        loss = per_ens_loss.sum()

        model.zero_grad()
        loss.backward()

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
                    wd = 0.0
                noise = torch.randn_like(param) * noise_scale
                param.add_(-lr * param.grad)
                param.add_(-lr * wd * param.data)
                param.add_(noise)

    model.eval()
    with torch.no_grad():
        gpr_pred = gpr_dot_product_explicit(X, Y, X_eval, sigma0_sq).cpu().numpy()  # (P_eval, 1)
        model_out_full = model(X_eval).detach().cpu().numpy()  # (P_eval, ens)
    return gpr_pred.squeeze(-1), model_out_full

def main():
    # Hyperparams
    d = 4
    P = 7
    N = 256
    ens = 15
    kappa = 5.0
    chi = 1.0
    num_seeds = 100
    epochs = 50_000
    lr = 1e-7 / P
    temperature = 2 * kappa

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Fixed eval set for aggregation
    torch.manual_seed(123)
    P_eval = 64
    X_eval = torch.randn(P_eval, d, device=device)

    gpr_list = []
    model_full_list = []

    for seed in range(num_seeds):
        print(f"Training seed {seed+1}/{num_seeds}")
        gpr_pred, model_out_full = train_one(seed, d, P, N, ens, kappa, epochs, lr, temperature, chi, X_eval)
        gpr_list.append(gpr_pred)  # shape (P_eval,)
        model_full_list.append(model_out_full)  # shape (P_eval, ens)

    gpr_array = np.stack(gpr_list, axis=0)  # (seeds, P_eval)
    model_array = np.stack(model_full_list, axis=0)  # (seeds, P_eval, ens)

    # Aggregate means/stds across seeds and ensembles
    gpr_mean = gpr_array.mean(axis=0)
    gpr_std = gpr_array.std(axis=0)
    model_mean = model_array.mean(axis=(0, 2))  # avg over seeds and ensembles
    model_std = model_array.std(axis=(0, 2))    # std over seeds and ensembles

    # Plot model vs GPR with error bars (std over seeds+ensemble on model; also show x-std as shade)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(gpr_mean, model_mean, yerr=model_std, fmt='o', markersize=6, alpha=0.7,
                elinewidth=1.3, capsize=3, label='model ± std (seeds+ens)')
    # Optional x-std band
    x_sorted_idx = np.argsort(gpr_mean)
    x_sorted = gpr_mean[x_sorted_idx]
    lower = (gpr_mean - gpr_std)[x_sorted_idx]
    upper = (gpr_mean + gpr_std)[x_sorted_idx]
    mn = min(lower.min(), (model_mean - model_std).min())
    mx = max(upper.max(), (model_mean + model_std).max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.4, label='y = x')
    slope, intercept = np.polyfit(gpr_mean, model_mean, 1)
    x_line = np.linspace(mn, mx, 200)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.4,
            label=f"best fit (slope={slope:.4f})")
    ax.fill_between(x_sorted, lower, upper,
                    color='gray', alpha=0.15, label='GPR ± std (seeds)')
    ax.set_xlabel('GPR prediction (mean over seeds)')
    ax.set_ylabel('Model output (mean over seeds+ens)')
    ax.set_title(f"Dataset-averaged GPR vs Model\nN={N}, ens={ens}, κ={kappa}, d={d}, P={P}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = Path(__file__).parent
    fig.savefig(str(out_dir / "gpr_model_dataset_averaged.png"), dpi=150)
    plt.close(fig)

    summary = {
        "d": d, "P": P, "N": N, "ens": ens, "kappa": kappa,
        "num_seeds": num_seeds, "epochs": epochs, "lr": lr,
        "model_mean": model_mean.tolist(),
        "model_std": model_std.tolist(),
        "gpr_mean": gpr_mean.tolist(),
        "gpr_std": gpr_std.tolist(),
        "slope": float(slope),
        "intercept": float(intercept)
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved plot and summary to {out_dir}")

if __name__ == "__main__":
    main()
