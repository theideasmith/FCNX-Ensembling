#!/usr/bin/env python3
"""Expose journal Langevin cubic runs as milestone-style model directories.

The journal sweep (``journal/run_langevin_d20_P_sweep.py``) writes flat
``{tag}_model.pt`` / ``{tag}_history.npz`` pairs. The plotting code here expects
one directory per run holding ``config.json`` and ``model_final.pt``, so this
script creates those directories with ``model_final.pt`` symlinked back to the
journal checkpoint.

    python3 link_langevin_cubic_runs.py                 # ep50000 runs (N=400)
    python3 link_langevin_cubic_runs.py --epochs 1000000
"""
import argparse
import json
import re
from pathlib import Path

JOURNAL_DIR = Path("/home/akiva/FCNX-Ensembling/journal/LearningCubic_models")
MILESTONE_DIR = Path(__file__).parent

TAG_RE = re.compile(
    r"^langevin_mf_d(?P<d>\d+)_N(?P<N>\d+)_P(?P<P>\d+)_chi(?P<chi>\d+)"
    r"_lr(?P<lr>[\d.eE+-]+)_T(?P<T>[\d.]+)_eps(?P<eps>[\d.]+)"
    r"_seed(?P<seed>\d+)_ep(?P<epochs>\d+)_model\.pt$"
)


def group_dir_for(epochs):
    return MILESTONE_DIR / f"LangevinCubic_ep{epochs}"


def run_dir_name(cfg):
    return (
        f"d{cfg['d']}_P{cfg['P']}_N{cfg['N']}_chi_{cfg['chi']:.1f}"
        f"_lr_{cfg['lr']}_T_{cfg['temperature']}_seed_{cfg['dataset_seed']}"
        f"_eps_{cfg['eps']}"
    )


def materialize(epochs, s0=1.0):
    out_root = group_dir_for(epochs)
    made = []
    for ckpt in sorted(JOURNAL_DIR.glob("langevin_mf_*_model.pt")):
        m = TAG_RE.match(ckpt.name)
        if not m or int(m.group("epochs")) != epochs:
            continue
        hist = ckpt.with_name(ckpt.name.replace("_model.pt", "_history.npz"))
        cfg = {
            "d": int(m.group("d")),
            "P": int(m.group("P")),
            "N": int(m.group("N")),
            "chi": float(m.group("chi")),
            "temperature": float(m.group("T")),
            "eps": float(m.group("eps")),
            "s0": float(s0),
            "dataset_seed": int(m.group("seed")),
            "lr": float(m.group("lr")),
            "epochs": epochs,
            "activation": "erf",
            "source_checkpoint": str(ckpt),
            "source_history": str(hist) if hist.exists() else None,
        }
        run_dir = out_root / run_dir_name(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        link = run_dir / "model_final.pt"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ckpt)
        made.append(run_dir)
    return out_root, made


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--s0", type=float, default=1.0)
    args = parser.parse_args()
    out_root, made = materialize(args.epochs, s0=args.s0)
    print(f"{len(made)} run dirs under {out_root}")
    for run_dir in made:
        print(f"  {run_dir.name}")


if __name__ == "__main__":
    main()
