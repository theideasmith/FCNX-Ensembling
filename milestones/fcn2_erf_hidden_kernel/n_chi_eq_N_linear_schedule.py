#!/usr/bin/env python3
"""Linear-task N sweep with chi=N, LR schedule, then automatic Langevin snapshots.

Specs
-----
  d = 10, P = 100, eps = 0 (purely linear teacher y = x_0)
  N in {20, 100, 1000}, chi = N
  ens = 1
  base lr = 1e-3  (step size = lr/P; schedule divides further)
  --schedule: lr → lr0/3, lr0/8, lr0/9 on stretched wall time
  --epochs = 1_000_000 effective (wall ≈ 3.6e6 under the schedule)
  T = 1.0 (default Langevin temperature; T_eff = T/chi = T/N)
  s0 = 1.0, seed = 0

After each width finishes the scheduled train, automatically resumes with
  --extra-epochs (final lr0/9) and dumps A/W0 snapshots under A_snapshots/.

Uses train_fcn2_erf.py (chi=N convention + LR schedule + snapshot flags).

Usage
-----
  python n_chi_eq_N_linear_schedule.py --dry-run
  python n_chi_eq_N_linear_schedule.py
  python n_chi_eq_N_linear_schedule.py --no-snapshots   # train only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_fcn2_erf.py"

D = 10
P = 100
N_VALUES = [20, 100, 1000]
EPS = 0.0
ENS = 1
LR = 1e-3
TEMPERATURE = 1.0
S0 = 1.0
SEED = 0
EPOCHS = 1_000_000  # effective under --schedule
LOG_INTERVAL = 50_000
DEVICE = "cuda:0"

# Late Langevin sampling (after scheduled train)
SNAPSHOT_EXTRA_EPOCHS = 200_000
SNAPSHOT_INTERVAL = 2_000
SNAPSHOT_BURNIN = 2_000

SWEEP_DIR = SCRIPT_DIR / "n_chi_eq_N_linear_d10_P100_schedule"
MODELS_DIR = SWEEP_DIR / "models"
TENSORBOARD_DIR = SWEEP_DIR / "tensorboard"
LOG_DIR = SWEEP_DIR / "launcher_logs"


def run_name(N: int) -> str:
    return f"d{D}_P{P}_N{N}_chi_{N}_lr_{LR}_T_{TEMPERATURE}_seed_{SEED}_eps_{EPS}_schedule"


def make_train_cmd(N: int, device: str) -> list[str]:
    name = run_name(N)
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
        str(N),
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
        "--schedule",
        "--output-dir",
        str(MODELS_DIR / name),
        "--tensorboard-dir",
        str(TENSORBOARD_DIR / name),
    ]


def make_snapshot_cmd(N: int, device: str) -> list[str]:
    """Resume finished (or in-progress) run; sample at final LR into A_snapshots/."""
    cmd = make_train_cmd(N, device)
    cmd.extend(
        [
            "--extra-epochs",
            str(SNAPSHOT_EXTRA_EPOCHS),
            "--snapshot-A-interval",
            str(SNAPSHOT_INTERVAL),
            "--snapshot-A-burnin",
            str(SNAPSHOT_BURNIN),
        ]
    )
    return cmd


def _run(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    print(" ", " ".join(cmd))
    print(f"  log -> {log_path}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(SCRIPT_DIR))
    print(f"  launched pid={proc.pid}")
    return int(proc.wait())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=None,
        help=f"Subset of widths (default: {N_VALUES})",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Skip automatic late Langevin snapshot phase after each train.",
    )
    parser.add_argument(
        "--snapshots-only",
        action="store_true",
        help="Skip training; only run the snapshot resume phase (requires checkpoints).",
    )
    args = parser.parse_args()
    widths = args.N if args.N is not None else list(N_VALUES)
    do_train = not args.snapshots_only
    do_snapshots = not args.no_snapshots

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("N sweep, chi=N, linear task, LR schedule")
    print("=" * 60)
    print(f"  d={D}, P={P}, eps={EPS}, ens={ENS}")
    print(f"  N values: {widths}  (chi=N)")
    print(f"  lr0={LR}, T={TEMPERATURE}, s0={S0}, seed={SEED}")
    print(f"  epochs={EPOCHS} effective with --schedule (wall ≈ 3.6×epochs)")
    print(f"  train={do_train}  snapshots={do_snapshots}")
    if do_snapshots:
        print(
            f"  snapshot phase: extra_epochs={SNAPSHOT_EXTRA_EPOCHS}, "
            f"interval={SNAPSHOT_INTERVAL}, burnin={SNAPSHOT_BURNIN}"
        )
    print(f"  device={args.device}")
    print(f"  sweep: {SWEEP_DIR}")
    print()

    for N in widths:
        name = run_name(N)
        if do_train:
            print(f"N={N}, chi={N}: scheduled train")
            rc = _run(make_train_cmd(N, args.device), LOG_DIR / f"{name}.log", args.dry_run)
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  completed train N={N} [{status}]")
            if rc != 0 and not args.dry_run:
                print(f"  skipping snapshots for N={N}")
                continue

        if do_snapshots:
            print(f"N={N}, chi={N}: late Langevin snapshots")
            rc = _run(
                make_snapshot_cmd(N, args.device),
                LOG_DIR / f"{name}_snapshots.log",
                args.dry_run,
            )
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  completed snapshots N={N} [{status}]")
            print(f"  A_snapshots -> {MODELS_DIR / name / 'A_snapshots'}")

    if args.dry_run:
        print("\nDry run only; not launching.")
    else:
        print(f"\nDone. Models under {MODELS_DIR}")
        print(f"TensorBoard: {TENSORBOARD_DIR}")


if __name__ == "__main__":
    main()
