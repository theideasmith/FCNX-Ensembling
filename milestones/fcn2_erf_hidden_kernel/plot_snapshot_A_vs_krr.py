#!/usr/bin/env python3
"""Compare single-A vs snapshot-averaged A to feature KRR.

For the continued Langevin run with A_snapshots/, report whether averaging
late readout samples closes the gap to ridge on frozen Phi = erf(W0 x).

Compares, per ensemble member:
  - final:     f = Phi(W_final) · A_final
  - A_bar:     f = Phi(W_final) · mean_t(A_t)          (frozen features)
  - f_bar:     f = mean_t[ Phi(W_t) · A_t ]            (predictive mean)
  - krr:      ridge on Phi(W_final) with sigma^2 = kappa_eff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric  # noqa: E402
from kappa_eff_solver import compute_kappa_eff  # noqa: E402

from plot_model_vs_preact_krr import (  # noqa: E402
    make_training_dataset,
    krr_from_features,
    parse_config_from_dirname,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = (
    SCRIPT_DIR
    / "red_robin_alpha_beta_invariant_P0160_Pmax3674_betamax9"
    / "models"
    / "invariant_beta5.196_alpha110.05_d114_P1678_N377_sa00.6623_kappa1.3806_seed0_continue_Asnap"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "action_h0_activation_plots"
    / "AlphaBetaInvariant"
    / "model_vs_preact_krr"
    / "snapshot_A_avg_P1678"
)


def load_state(model_dir: Path, device: torch.device):
    for candidate in ("model_final.pt", "checkpoint.pt", "model.pt"):
        path = model_dir / candidate
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No model weights in {model_dir}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def load_snapshots(snap_dir: Path, device: torch.device):
    paths = sorted(snap_dir.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pt in {snap_dir}")
    As, W0s, epochs = [], [], []
    for p in paths:
        snap = torch.load(p, map_location="cpu", weights_only=False)
        As.append(snap["A"])
        W0s.append(snap["W0"])
        epochs.append(int(snap["epoch"]))
    A = torch.stack(As, dim=0).to(device)  # (T, ens, N)
    W0 = torch.stack(W0s, dim=0).to(device)  # (T, ens, N, d)
    return A, W0, epochs


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Snapshot-averaged A vs feature KRR")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sigma2", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    run_dir = args.run_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d, P, N, chi, seed, temperature, eps, s0 = parse_config_from_dirname(str(run_dir))
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    sa0 = float(cfg["sa0"])
    sigma_w0 = float(cfg.get("sigmaW0", 1.0 / d))
    sigma_a = float(cfg.get("sigmaA", sa0 / N))
    eps_val = float(eps if eps is not None else 0.0)

    state = load_state(run_dir, device)
    ens = int(state["A"].shape[0])
    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=P,
        ens=ens,
        activation="erf",
        weight_initialization_variance=(sigma_w0, sigma_a),
        device=device,
    )
    model.load_state_dict(state)
    model.eval()

    A_snaps, W0_snaps, epochs = load_snapshots(run_dir / "A_snapshots", device)
    T = A_snaps.shape[0]
    A_bar = A_snaps.mean(dim=0)  # (ens, N)

    X, y, he1, he3 = make_training_dataset(P, d, seed, eps_val, device)
    y = y.squeeze(-1)

    kappa_bare = float(temperature) / 2.0
    if args.sigma2 is not None:
        sigma2 = float(args.sigma2)
        sigma2_label = "override"
    else:
        try:
            sigma2 = float(
                compute_kappa_eff(
                    d=int(d),
                    P=int(P),
                    kappa_bare=float(kappa_bare),
                    n1=int(N),
                    chi=float(chi),
                    device=device,
                    verbose=False,
                )
            )
            sigma2_label = "kappa_eff"
        except Exception as exc:
            print(f"kappa_eff failed ({exc}); using kappa_bare={kappa_bare}")
            sigma2 = kappa_bare
            sigma2_label = "kappa_bare"

    print(f"run={run_dir.name}")
    print(f"device={device}  T_snaps={T}  epochs={epochs[0]}..{epochs[-1]}")
    print(f"sigma2={sigma2:.6g} ({sigma2_label})  kappa_bare={kappa_bare:.6g}")

    with torch.no_grad():
        phi_final = model.h0_activation(X)  # (P, ens, N)
        f_final = model.forward(X)  # (P, ens)
        f_Abar = (phi_final * A_bar).sum(-1)

        f_krr = torch.empty_like(f_final)
        A_ridge = torch.empty_like(A_bar)
        for q in range(ens):
            phi_q = phi_final[:, q, :]
            f_krr[:, q] = krr_from_features(phi_q, y, sigma2)
            gram = phi_q.T @ phi_q
            ridge = sigma2 * N
            eye = torch.eye(N, device=device, dtype=phi_q.dtype)
            A_ridge[q] = torch.linalg.solve(gram + ridge * eye, phi_q.T @ y)

        # Predictive mean over snapshots (recompute Phi each time)
        f_bar = torch.zeros_like(f_final)
        for t in range(T):
            model.W0.copy_(W0_snaps[t])
            model.A.copy_(A_snaps[t])
            f_bar += model.forward(X)
        f_bar /= float(T)

    # Restore final weights
    model.load_state_dict(state)

    y_np = y.detach().cpu().numpy()
    finals = {
        "final_A": f_final.detach().cpu().numpy(),
        "A_bar_frozen_Phi": f_Abar.detach().cpu().numpy(),
        "f_bar_snapshots": f_bar.detach().cpu().numpy(),
        "feature_krr": f_krr.detach().cpu().numpy(),
    }

    rows = []
    print("\n=== vs feature KRR (frozen Phi = erf(W_final x)) ===")
    print(f"{'predictor':<22} {'MSE_vs_krr':>12} {'corr_krr':>10} {'MSE_vs_y':>12}")
    for name, arr in finals.items():
        row = (
            name,
            mse(arr, finals["feature_krr"]),
            corr(arr, finals["feature_krr"]),
            mse(arr.mean(axis=1), y_np),
        )
        rows.append(row)
        print(f"{row[0]:<22} {row[1]:12.4e} {row[2]:10.4f} {row[3]:12.4e}")

    # Distance of A to ridge A
    A_final_np = state["A"].detach().cpu().numpy()
    A_bar_np = A_bar.detach().cpu().numpy()
    A_ridge_np = A_ridge.detach().cpu().numpy()
    print("\n=== readout A vs ridge A (same frozen Phi) ===")
    print(
        f"||A_final - A_ridge||_rms / ||A_ridge||_rms = "
        f"{np.sqrt(np.mean((A_final_np - A_ridge_np) ** 2)) / (np.sqrt(np.mean(A_ridge_np ** 2)) + 1e-12):.4f}"
    )
    print(
        f"||A_bar   - A_ridge||_rms / ||A_ridge||_rms = "
        f"{np.sqrt(np.mean((A_bar_np - A_ridge_np) ** 2)) / (np.sqrt(np.mean(A_ridge_np ** 2)) + 1e-12):.4f}"
    )
    print(f"corr(A_final, A_ridge) = {corr(A_final_np, A_ridge_np):.4f}")
    print(f"corr(A_bar,   A_ridge) = {corr(A_bar_np, A_ridge_np):.4f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True, sharey=True)
    krr_np = finals["feature_krr"]
    lo = min(krr_np.min(), finals["final_A"].min(), finals["A_bar_frozen_Phi"].min(), finals["f_bar_snapshots"].min())
    hi = max(krr_np.max(), finals["final_A"].max(), finals["A_bar_frozen_Phi"].max(), finals["f_bar_snapshots"].max())
    for ax, key, title in zip(
        axes,
        ["final_A", "A_bar_frozen_Phi", "f_bar_snapshots"],
        ["final A", r"$\bar A$ on frozen $\Phi$", r"$\overline{f_t}$ (snap mean)"],
    ):
        ax.scatter(krr_np.ravel(), finals[key].ravel(), s=3, alpha=0.2, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{title}\ncorr={corr(finals[key], krr_np):.3f}")
        ax.set_xlabel("feature KRR")
        ax.set_ylabel("predictor")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"P={P} N={N}  T_snaps={T}  sigma2={sigma2_label}={sigma2:.4g}\n{run_dir.name}",
        fontsize=9,
    )
    fig.tight_layout()
    scatter_path = out_dir / "predictors_vs_krr.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    x0 = X[:, 0].detach().cpu().numpy()
    order = np.argsort(x0)
    ax.scatter(x0, y_np, s=6, alpha=0.25, color="0.5", label="y", rasterized=True)
    ax.plot(x0[order], finals["feature_krr"].mean(1)[order], color="darkorange", lw=1.8, label="feature KRR")
    ax.plot(x0[order], finals["final_A"].mean(1)[order], color="royalblue", lw=1.4, label="final A")
    ax.plot(x0[order], finals["A_bar_frozen_Phi"].mean(1)[order], color="seagreen", lw=1.4, label=r"$\bar A$")
    ax.plot(x0[order], finals["f_bar_snapshots"].mean(1)[order], color="crimson", lw=1.2, ls="--", label=r"$\overline{f_t}$")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel("prediction (ens mean)")
    ax.legend(fontsize=8)
    ax.set_title(f"P={P}: snapshot-averaged readout vs KRR")
    fig.tight_layout()
    x0_path = out_dir / "predictors_vs_x0.png"
    fig.savefig(x0_path, dpi=150)
    plt.close(fig)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"snapshots: {T}  epochs {epochs[0]}..{epochs[-1]}\n")
        f.write(f"sigma2: {sigma2} ({sigma2_label})  kappa_bare={kappa_bare}\n\n")
        f.write(f"{'predictor':<22} {'MSE_vs_krr':>12} {'corr_krr':>10} {'MSE_vs_y':>12}\n")
        for row in rows:
            f.write(f"{row[0]:<22} {row[1]:12.4e} {row[2]:10.4f} {row[3]:12.4e}\n")
        f.write("\n")
        f.write(
            f"rel_rms(A_final,A_ridge)="
            f"{np.sqrt(np.mean((A_final_np - A_ridge_np) ** 2)) / (np.sqrt(np.mean(A_ridge_np ** 2)) + 1e-12):.6f}\n"
        )
        f.write(
            f"rel_rms(A_bar,A_ridge)="
            f"{np.sqrt(np.mean((A_bar_np - A_ridge_np) ** 2)) / (np.sqrt(np.mean(A_ridge_np ** 2)) + 1e-12):.6f}\n"
        )
        f.write(f"corr(A_final,A_ridge)={corr(A_final_np, A_ridge_np):.6f}\n")
        f.write(f"corr(A_bar,A_ridge)={corr(A_bar_np, A_ridge_np):.6f}\n")

    print(f"\nSaved {scatter_path}")
    print(f"Saved {x0_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
