#!/usr/bin/env python3
"""Branch Langevin sampling for d50 weight histograms — does NOT stop the live runs.

Copies the current checkpoint into a new output dir, then continues Langevin at
the final schedule LR while writing A/W0 snapshots. plot_action_h0_activation
already pools A_snapshots/ for target-weight actions (n = N * T_snaps).

Default: ~200 snapshots × N=400 ≈ 80k weight samples per run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PYTHON = Path("/home/akiva/miniconda3/envs/ml_env/bin/python")

# (label, source model dir, base_lr, schedule epochs used by parent run)
RUNS = [
    {
        "label": "P3000",
        "source": ROOT
        / "red_robin_d50_T0.2_P3000_N400_chi400_eps0.5_lr0.01_ep50M_schedule_2_3_5"
        / "models"
        / "d50_P3000_N400_chi_400_lr_0.01_T_0.2_seed_42_eps_0.5_schedule_2_3_5",
        "branch_root": ROOT
        / "red_robin_d50_T0.2_P3000_N400_chi400_eps0.5_lr0.01_ep50M_weight_snaps",
        "P": 3000,
        "lr": 0.01,
        "schedule_epochs": 50_000_000,
    },
    {
        "label": "P16000",
        "source": ROOT
        / "red_robin_d50_T0.2_P16000_N400_chi400_eps0.5_lr0.053333_ep100M_schedule_2_3_5_resume"
        / "models"
        / "d50_P16000_N400_chi_400_lr_0.053333_T_0.2_seed_42_eps_0.5_schedule_2_3_5_from_lr0p01_ep29M",
        "branch_root": ROOT
        / "red_robin_d50_T0.2_P16000_N400_chi400_eps0.5_lr0.053333_ep100M_weight_snaps",
        "P": 16000,
        "lr": 0.053333,
        "schedule_epochs": 100_000_000,
    },
]


def peek_epoch(model_dir: Path) -> int:
    ckpt = model_dir / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return int(obj.get("epoch", 0))


def seed_branch_dir(source: Path, dest: Path) -> int:
    """Copy checkpoint artifacts into an empty branch dir; return resume epoch."""
    dest.mkdir(parents=True, exist_ok=True)
    epoch = peek_epoch(source)
    for name in (
        "checkpoint.pt",
        "model.pt",
        "config.json",
        "eigenvalues_over_time.json",
        "losses.json",
    ):
        src = source / name
        if src.exists() and not (dest / name).exists():
            shutil.copy2(src, dest / name)
    meta = {
        "branched_from": str(source),
        "branch_epoch": epoch,
        "purpose": "A/W0 Langevin snapshots for weight-action histograms",
    }
    (dest / "branch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return epoch


def build_cmd(run: dict, dest: Path, epoch: int, args: argparse.Namespace) -> list[str]:
    # Hold schedule budget at current epoch so --extra-epochs runs at final lr.
    schedule_epochs = min(int(run["schedule_epochs"]), epoch)
    return [
        str(PYTHON),
        "-u",
        str(ROOT / "train_fcn2_erf.py"),
        "--d",
        "50",
        "--P",
        str(run["P"]),
        "--N",
        "400",
        "--chi",
        "400",
        "--eps",
        "0.5",
        "--ens",
        "1",
        "--lr",
        str(run["lr"]),
        "--temperature",
        "0.2",
        "--s0",
        "1.0",
        "--dataset-seed",
        "42",
        "--epochs",
        str(schedule_epochs),
        "--extra-epochs",
        str(args.extra_epochs),
        "--log-interval",
        str(args.log_interval),
        "--schedule-divisors",
        "2,3,5",
        "--device",
        args.device,
        "--classic",
        "--output-dir",
        str(dest),
        "--tensorboard-dir",
        str(run["branch_root"] / "tensorboard" / dest.name),
        "--snapshot-A-interval",
        str(args.snapshot_interval),
        "--snapshot-A-burnin",
        str(args.snapshot_burnin),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--extra-epochs",
        type=int,
        default=2_000_000,
        help="Wall steps of continued Langevin at final schedule LR",
    )
    ap.add_argument(
        "--snapshot-interval",
        type=int,
        default=10_000,
        help="Save A/W0 every N wall epochs after burn-in",
    )
    ap.add_argument(
        "--snapshot-burnin",
        type=int,
        default=50_000,
        help="Mix steps after branch resume before first snapshot",
    )
    ap.add_argument("--log-interval", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        choices=["P3000", "P16000"],
        default=None,
        help="Launch a single branch (default: both)",
    )
    ap.add_argument(
        "--background",
        action="store_true",
        help="Launch each branch under nohup in branch launcher_logs/",
    )
    args = ap.parse_args()

    n_snaps = max(0, (args.extra_epochs - args.snapshot_burnin) // args.snapshot_interval)
    print(
        f"Plan: extra={args.extra_epochs}, burnin={args.snapshot_burnin}, "
        f"interval={args.snapshot_interval} → ~{n_snaps} snapshots "
        f"(~{n_snaps * 400} target weights each)"
    )

    selected = [r for r in RUNS if args.only is None or r["label"] == args.only]
    for run in selected:
        source = run["source"]
        if not (source / "checkpoint.pt").exists():
            print(f"SKIP {run['label']}: no checkpoint at {source}")
            continue
        dest = run["branch_root"] / "models" / f"{source.name}_weight_snaps"
        epoch = seed_branch_dir(source, dest)
        cmd = build_cmd(run, dest, epoch, args)
        log_dir = run["branch_root"] / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{dest.name}.log"
        print(f"\n=== {run['label']} branch @ epoch {epoch} ===")
        print(f"  source: {source}")
        print(f"  dest:   {dest}")
        print(f"  log:    {log_path}")
        print("  cmd:", " ".join(cmd))
        if args.dry_run:
            continue
        # Pin to nvidia-smi bus order so --device cuda:0 is the free 4080 when
        # CUDA_VISIBLE_DEVICES=0 (default CUDA order puts the 4090 first).
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        if args.background:
            with open(log_path, "wb", buffering=0) as logf:
                subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            print(f"  launched in background → {log_path}")
        else:
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
