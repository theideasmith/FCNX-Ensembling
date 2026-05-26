#!/usr/bin/env python3

import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path


RESULTS_ROOT = Path(__file__).parent / "p_scan_erf_results"
TRAIN_SCRIPT = Path(__file__).parent / "d_sweep_seeds.py"

TARGET_D = 150
TARGET_CHI = 80.0
TARGET_KAPPA = 2.0
EXTRA_EPOCHS = 3_000_000
MAX_PARALLEL_JOBS = 1
DEFAULT_DEVICE = "cuda:0"


def load_json(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def matches_target(cfg):
    try:
        return (
            int(cfg.get("d", -1)) == TARGET_D
            and abs(float(cfg.get("chi", float("nan"))) - TARGET_CHI) < 1e-9
            and abs(float(cfg.get("kappa", float("nan"))) - TARGET_KAPPA) < 1e-9
        )
    except (TypeError, ValueError):
        return False


def resolve_seed(seed_dir: Path, cfg):
    if "base_seed" in cfg:
        try:
            return int(cfg["base_seed"])
        except (TypeError, ValueError):
            pass

    name = seed_dir.name
    if name.startswith("base_seed"):
        try:
            return int(name.removeprefix("base_seed"))
        except ValueError:
            return None

    return None


def build_cmd(cfg, seed, target_epochs):
    return [
        "python3",
        str(TRAIN_SCRIPT),
        "--d", str(int(cfg["d"])),
        "--P", str(int(cfg["P"])),
        "--N", str(int(cfg["N"])),
        "--chi", str(int(float(cfg["chi"]))),
        "--kappa", str(float(cfg["kappa"])),
        "--lr", str(float(cfg["lr"])),
        "--epochs", str(int(target_epochs)),
        "--device", str(cfg.get("device", DEFAULT_DEVICE)),
        "--num_seeds", str(int(cfg.get("num_seeds", 1))),
        "--base_seed", str(seed),
        "--ens", str(int(cfg.get("ens", 50))),
        "--to", "p_scan_erf_results",
        "--eps", str(float(cfg.get("eps", 0.03))),
        "--exact-epochs",
    ]


def discover_jobs():
    jobs = []
    for config_path in sorted(RESULTS_ROOT.glob("d*_P*_N*_chi*_kappa*/base_seed*/config.json")):
        seed_dir = config_path.parent
        checkpoint_exists = (seed_dir / "model_final.pt").exists() or (seed_dir / "model.pt").exists()
        if not checkpoint_exists:
            continue

        cfg = load_json(config_path)
        if not matches_target(cfg):
            continue

        seed = resolve_seed(seed_dir, cfg)
        if seed is None:
            continue
        
        P = int(cfg["P"])
    
        current_epoch = int(cfg.get("current_epoch", cfg.get("epochs", 0)))
        target_epochs = current_epoch + EXTRA_EPOCHS
        jobs.append({
            "seed_dir": seed_dir,
            "cfg": cfg,
            "seed": seed,
            "current_epoch": current_epoch,
            "target_epochs": target_epochs,
        })

    return jobs


def main():
    jobs = discover_jobs()
    if not jobs:
        print(f"No matching runs found under {RESULTS_ROOT}")
        return

    job_queue = deque(jobs)
    running_procs = []

    print(f"Found {len(jobs)} matching runs to continue by {EXTRA_EPOCHS} epochs each.")
    for job in jobs:
        cfg = job["cfg"]
        print(
            f"  d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}, kappa={cfg['kappa']}, "
            f"seed={job['seed']}, current_epoch={job['current_epoch']} -> target_epochs={job['target_epochs']}"
        )

    while job_queue or running_procs:
        for job in running_procs[:]:
            if job["proc"].poll() is not None:
                print(f"[Done] {job['seed_dir']}")
                running_procs.remove(job)

        while len(running_procs) < MAX_PARALLEL_JOBS and job_queue:
            next_job = job_queue.popleft()
            cmd = build_cmd(next_job["cfg"], next_job["seed"], next_job["target_epochs"])
            print(f"[Launching] {next_job['seed_dir']} -> epochs={next_job['target_epochs']}")
            proc = subprocess.Popen(cmd)
            next_job["proc"] = proc
            running_procs.append(next_job)
            time.sleep(1.0)

        time.sleep(5)

    print("All continuation jobs finished.")


if __name__ == "__main__":
    main()