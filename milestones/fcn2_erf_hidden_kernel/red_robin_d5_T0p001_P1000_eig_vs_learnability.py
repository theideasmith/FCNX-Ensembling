#!/usr/bin/env python3
"""Red-robin launcher: d=5, T=0.001, P=1000 — track eigs vs cubic learnability.

Motivation
----------
Journal Langevin runs at eps=0.074 showed cubic learnability is strongly
controlled by kappa ~ T (T=1 → negative He3 at P=1000; T=0.001 → He3≈0.40).
This run trains the milestone FCN2 Langevin pipeline long enough, with the
standard LR schedule, to watch *when* the H-spectrum and cubic learnability
converge relative to each other.

Specs
-----
  d=5, P=1000, N=400, chi=400
  T=0.001  (T_eff = T/chi = 2.5e-6; kappa_bare = T/2 = 5e-4)
  eps=0.074, ens=1, s0=1.0, seed=0
  base lr = 0.01  (step = lr/P; same convention as journal Langevin)
  --schedule: lr → lr0/3, lr0/8, lr0/9 on stretched wall time
  --epochs = 1_000_000 effective  (wall ≈ 3.6× under the schedule)
  log every 25_000 wall epochs (eigenvalues + cubic_L3 + test MSE)

Artifacts (under SWEEP_DIR)
---------------------------
  models/<run>/eigenvalues_over_time.json
  models/<run>/losses.json          # includes cubic_learnability
  models/<run>/eig_vs_learnability.png
  tensorboard/<run>/
  launcher_logs/<run>.log

Usage
-----
  python red_robin_d5_T0p001_P1000_eig_vs_learnability.py --dry-run
  python red_robin_d5_T0p001_P1000_eig_vs_learnability.py
  python red_robin_d5_T0p001_P1000_eig_vs_learnability.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_fcn2_erf.py"

D = 5
P = 1000
N = 400
CHI = 400.0
TEMPERATURE = 0.001
EPS = 0.074
ENS = 1
LR = 0.01
S0 = 1.0
SEED = 0
EPOCHS = 1_000_000  # effective under --schedule
LOG_INTERVAL = 25_000
DEVICE = "cuda:0"

SWEEP_DIR = SCRIPT_DIR / (
    f"red_robin_d{D}_T{TEMPERATURE}_P{P}_N{N}_chi{int(CHI)}"
    f"_eps{EPS}_lr{LR}_eig_vs_L3_schedule"
)
MODELS_DIR = SWEEP_DIR / "models"
TENSORBOARD_DIR = SWEEP_DIR / "tensorboard"
LOG_DIR = SWEEP_DIR / "launcher_logs"


def run_name() -> str:
    return (
        f"d{D}_P{P}_N{N}_chi_{CHI}_lr_{LR}_T_{TEMPERATURE}"
        f"_seed_{SEED}_eps_{EPS}_schedule"
    )


def run_dir() -> Path:
    return MODELS_DIR / run_name()


def make_train_cmd(device: str) -> list[str]:
    name = run_name()
    return [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--d",
        str(D),
        "--P",
        str(P),
        "--N",
        str(N),
        "--chi",
        str(CHI),
        "--eps",
        str(EPS),
        "--ens",
        str(ENS),
        "--lr",
        str(LR),
        "--temperature",
        str(TEMPERATURE),
        "--s0",
        str(S0),
        "--dataset-seed",
        str(SEED),
        "--epochs",
        str(EPOCHS),
        "--log-interval",
        str(LOG_INTERVAL),
        "--device",
        device,
        "--classic",
        "--schedule",
        "--output-dir",
        str(MODELS_DIR / name),
        "--tensorboard-dir",
        str(TENSORBOARD_DIR / name),
    ]


def _load_series(rd: Path):
    eigs_path = rd / "eigenvalues_over_time.json"
    losses_path = rd / "losses.json"
    if not eigs_path.exists():
        raise FileNotFoundError(f"missing {eigs_path}")
    if not losses_path.exists():
        raise FileNotFoundError(f"missing {losses_path}")

    with open(eigs_path) as f:
        eigs_raw = json.load(f)
    with open(losses_path) as f:
        losses = json.load(f)

    L3 = {int(k): float(v) for k, v in losses.get("cubic_learnability", {}).items()}
    mse = {int(k): float(v) for k, v in losses.get("test_mse", {}).items()}

    epochs = sorted(int(k) for k in eigs_raw.keys())
    max_eig, mean_perp, top5 = [], [], []
    for ep in epochs:
        arr = np.asarray(eigs_raw[str(ep)], dtype=float).ravel()
        # H_eig returns (ens, n_modes) or (n_modes,); take ensemble mean
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        # Sometimes stored nested; flatten safely
        arr = np.asarray(arr, dtype=float).ravel()
        max_eig.append(float(arr.max()) if arr.size else float("nan"))
        mean_perp.append(float(arr[1:].mean()) if arr.size > 1 else float("nan"))
        top5.append(arr[: min(5, arr.size)].copy())

    return {
        "epochs": np.asarray(epochs),
        "max_eig": np.asarray(max_eig),
        "mean_perp": np.asarray(mean_perp),
        "top5": top5,
        "L3": np.asarray([L3.get(ep, np.nan) for ep in epochs]),
        "mse": np.asarray([mse.get(ep, np.nan) for ep in epochs]),
    }


def plot_eig_vs_learnability(rd: Path | None = None, out_path: Path | None = None) -> Path:
    """Overlay H-spectrum stats and cubic learnability vs wall epoch."""
    rd = Path(rd) if rd is not None else run_dir()
    series = _load_series(rd)
    epochs = series["epochs"]
    out_path = Path(out_path) if out_path is not None else rd / "eig_vs_learnability.png"

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(epochs, series["L3"], "C0-o", ms=3, lw=1.5, label=r"cubic $L_3$")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_ylabel(r"cubic learnability $L_3$")
    ax.set_title(
        f"d={D}, P={P}, N=χ={N}, T={TEMPERATURE}, eps={EPS}\n"
        f"scheduled Langevin (lr0={LR}, 1M effective epochs)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(epochs, series["max_eig"], "C1-o", ms=3, lw=1.5, label=r"$\lambda_{\max}(H)$")
    ax.plot(epochs, series["mean_perp"], "C2--", lw=1.2, label=r"mean $\lambda_{\perp}$")
    for i in range(min(3, len(series["top5"][0]) if series["top5"] else 0)):
        ax.plot(
            epochs,
            [row[i] for row in series["top5"]],
            alpha=0.35,
            lw=1.0,
            label=rf"$\lambda_{{{i}}}$" if i < 3 else None,
        )
    ax.set_ylabel("H eigenvalues")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    ax.plot(epochs, series["mse"], "C3-o", ms=3, lw=1.5, label="test MSE")
    ax.set_ylabel("test MSE")
    ax.set_xlabel("wall epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    for ax in axes:
        ax.set_xscale("log")

    # Parametric inset: L3 vs λ_max
    if np.isfinite(series["L3"]).any() and np.isfinite(series["max_eig"]).any():
        ax_ins = axes[0].inset_axes([0.55, 0.15, 0.4, 0.45])
        sc = ax_ins.scatter(
            series["max_eig"],
            series["L3"],
            c=np.log10(np.maximum(epochs, 1)),
            s=12,
            cmap="viridis",
        )
        ax_ins.set_xlabel(r"$\lambda_{\max}$", fontsize=8)
        ax_ins.set_ylabel(r"$L_3$", fontsize=8)
        ax_ins.tick_params(labelsize=7)
        ax_ins.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax_ins, fraction=0.046, pad=0.04).set_label(
            r"$\log_{10}$ epoch", fontsize=7
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def _run(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    print(" ", " ".join(cmd))
    print(f"  log -> {log_path}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(SCRIPT_DIR)
        )
    print(f"  launched pid={proc.pid}")
    return int(proc.wait())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only rebuild eig_vs_learnability.png from an existing run dir.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch training with nohup and return immediately (log under launcher_logs/).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    name = run_name()
    print("=" * 60)
    print("Red-robin: d=5 T=0.001 P=1000 — eigs vs cubic learnability")
    print("=" * 60)
    print(f"  d={D}, P={P}, N={N}, chi={CHI}")
    print(f"  T={TEMPERATURE}  (T_eff={TEMPERATURE / CHI:g}, kappa_bare={TEMPERATURE / 2:g})")
    print(f"  eps={EPS}, ens={ENS}, lr0={LR}, s0={S0}, seed={SEED}")
    print(f"  epochs={EPOCHS} effective with --schedule (wall ≈ 3.6×)")
    print(f"  log_interval={LOG_INTERVAL}")
    print(f"  device={args.device}")
    print(f"  sweep: {SWEEP_DIR}")
    print()

    if args.plot_only:
        plot_eig_vs_learnability()
        return

    cmd = make_train_cmd(args.device)
    log_path = LOG_DIR / f"{name}.log"

    if args.background and not args.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(SCRIPT_DIR),
                start_new_session=True,
            )
        print(f"  background pid={proc.pid}")
        print(f"  log -> {log_path}")
        print(f"  models -> {run_dir()}")
        return

    print(f"{name}: scheduled train")
    rc = _run(cmd, log_path, args.dry_run)
    status = "OK" if rc == 0 else f"FAIL rc={rc}"
    print(f"  completed train [{status}]")

    if args.dry_run:
        print("\nDry run only; not launching.")
        return

    if rc == 0:
        try:
            plot_eig_vs_learnability()
        except Exception as exc:
            print(f"  warn: could not plot eig vs L3 yet: {exc}")

    print(f"\nDone. Models under {MODELS_DIR}")
    print(f"TensorBoard: {TENSORBOARD_DIR}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
