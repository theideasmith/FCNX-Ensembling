#!/usr/bin/env python3
"""Streaming He3 projection eigenvalues for Langevin d=100 N=3000 checkpoints.

Accumulates per-neuron
    proj = E_x[ h0(x) * He3(x_a) ]
over P_total Gaussian samples, then reports
    λ = mean_neuron[ proj^2 ]   (second moment)
for target axis a=0 and mean over several perp axes.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/akiva/FCNX-Ensembling/lib")
from FCN2Network import FCN2NetworkActivationGeneric

OUT = "/home/akiva/FCNX-Ensembling/journal/LearningCubic_models"
d, N, chi, eps, T, lr, seed, epochs = 100, 3000, 3000, 0.5, 0.001, 0.01, 42, 50_000
P_LIST = [50, 100, 500, 1000, 2000, 3000, 8000, 16000]
P_TOTAL = int(os.environ.get("P_TOTAL", 100_000_000))
BATCH = int(os.environ.get("BATCH", 50_000))
N_PERP = int(os.environ.get("N_PERP", 4))
INV_SQRT6 = 1.0 / math.sqrt(6.0)


def load_model(P, device):
    tag = (
        f"langevin_mf_d{d}_N{N}_P{P}_chi{int(chi)}"
        f"_lr{lr}_T{T}_eps{eps}_seed{seed}_ep{epochs}"
    )
    path = os.path.join(OUT, f"{tag}_model.pt")
    hist = os.path.join(OUT, f"{tag}_history.npz")
    he3 = float(np.load(hist)["he3_test"][-1]) if os.path.exists(hist) else float("nan")
    sigma_w0 = 1.0 / d
    sigma_a = 1.0 / (N * chi)
    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=P,
        ens=1,
        activation="erf",
        weight_initialization_variance=(sigma_w0, sigma_a),
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, he3, tag


@torch.no_grad()
def stream_proj_eigs(model, device, p_total=P_TOTAL, batch=BATCH, n_perp=N_PERP):
    ens, n1 = model.ens, model.n1
    n_axes = 1 + n_perp
    acc = torch.zeros(n_axes, ens, n1, dtype=torch.float64, device=device)
    n_seen = 0
    t0 = time.time()
    log_every = max(p_total // 10, batch)

    while n_seen < p_total:
        bs = min(batch, p_total - n_seen)
        x = torch.randn(bs, d, dtype=torch.float32, device=device)
        h0 = model.h0_activation(x)
        for a in range(n_axes):
            xa = x[:, a]
            phi = (xa * xa * xa - 3.0 * xa) * INV_SQRT6
            acc[a] += torch.einsum("pqn,p->qn", h0, phi).double()
        n_seen += bs
        if (n_seen % log_every) < batch or n_seen >= p_total:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-9)
            eta = (p_total - n_seen) / max(rate, 1e-9)
            print(
                f"    streamed {n_seen / 1e6:.1f}M/{p_total / 1e6:.0f}M  "
                f"{rate / 1e6:.2f}M/s  ETA {eta / 60:.1f} min",
                flush=True,
            )
        del x, h0

    proj = acc / n_seen
    sm = (proj * proj).mean(dim=(1, 2))
    var = proj.var(dim=(1, 2), unbiased=False)
    return {
        "lambda_T": float(sm[0].item()),
        "lambda_P": float(sm[1:].mean().item()),
        "lambda_P_std": float(sm[1:].std().item()) if n_perp > 1 else 0.0,
        "var_T": float(var[0].item()),
        "var_P": float(var[1:].mean().item()),
        "n_seen": n_seen,
        "seconds": time.time() - t0,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}  P_total={P_TOTAL:,}  batch={BATCH}  n_perp={N_PERP}",
        flush=True,
    )
    print(
        f"{'P':>6} {'He3':>8} {'λ3T':>12} {'λ3P':>12} {'T/P':>8} "
        f"{'varT':>12} {'varP':>12}",
        flush=True,
    )
    rows = []
    for P in P_LIST:
        print(f"\n=== P={P} ===", flush=True)
        model, he3, _tag = load_model(P, device)
        stats = stream_proj_eigs(model, device)
        ratio = (
            stats["lambda_T"] / stats["lambda_P"]
            if stats["lambda_P"] > 0
            else float("nan")
        )
        print(
            f"{P:6d} {he3:8.4f} {stats['lambda_T']:12.4e} {stats['lambda_P']:12.4e} "
            f"{ratio:8.2f} {stats['var_T']:12.4e} {stats['var_P']:12.4e}  "
            f"({stats['seconds'] / 60:.1f} min, λP±{stats['lambda_P_std']:.2e})",
            flush=True,
        )
        rows.append((P, he3, stats))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\n=== SUMMARY (streaming projections, "
        f"{P_TOTAL:.0e} samples, λ=2nd moment of E_x[h0·He3]) ===",
        flush=True,
    )
    print(f"{'P':>6} {'He3':>8} {'λ3T':>12} {'λ3P':>12} {'T/P':>8}", flush=True)
    for P, he3, s in rows:
        print(
            f"{P:6d} {he3:8.4f} {s['lambda_T']:12.4e} {s['lambda_P']:12.4e} "
            f"{s['lambda_T'] / s['lambda_P']:8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
