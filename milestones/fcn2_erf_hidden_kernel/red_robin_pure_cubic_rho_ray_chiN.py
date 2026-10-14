#!/usr/bin/env python3
"""Pure-cubic χ=N ρ-ray train scan — matched to the multiwell FO theory.

Background
----------
LR-accelerated steepwell trains (d=5,9,16,28,50) used:

    red_robin_alpha_beta_invariant_steepwell_eps0p03_N30d.py
    (and parent red_robin_alpha_beta_invariant_steepwell.py, ε=0.5)

with χ=1, Y=z+ε He3(z), Finv≈4.6e4 invariant, base_lr=0.01,
--schedule-divisors 2,3,5 on a 60M wall, A snaps over the last 20M.

Those are *not* the pure-cubic χ=N FO theory we just benchmarked
(y=(0,ε), ρ=κ/(sa0 P), FO≈0.09, center catch-up≈0.076, L3→1 as ρ→0).

This launcher
-------------
Fixed theory anchor (from fo_rho_deep_l3 / FO-window scans):

    d=16, N=200, χ=N=200, P=1000, sa0=0.03, s0=1, ε=1, he1=0

Scan κ so ρ = κ/(sa0 P) covers pre-FO → outer FO → catch-up → deep L3:

    ρ ∈ {0.15, 0.12, 0.10, 0.095, 0.09, 0.085, 0.08, 0.076,
         0.07, 0.05, 0.02, 0.01, 0.005, 0.001}

Same LR recipe as the accelerated steepwell trains:
  base_lr=0.01, schedule 2,3,5, 60M wall, A snaps last 20M / 500k.

Optional follow-up (not launched here): a d∈{5,9,16,28,50} ray at *fixed*
ρ (e.g. 0.01 or 0.05) with χ=N, he1=0, ε=1, to check finite-d continuum
of the same pure-cubic learnable regime.

Usage (do not start until GPU is free / user asks)::

    python red_robin_pure_cubic_rho_ray_chiN.py            # dry-run table
    python red_robin_pure_cubic_rho_ray_chiN.py --launch   # actually train
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

# --- Theory-matched fixed point (χ=N pure cubic FO / deep L3) ---
D = 16
N = 200
CHI = float(N)  # classic χ=N
P = 1000
SA0 = 0.03
S0 = 1.0
TASK_EPS = 1.0
HE1 = 0.0  # pure cubic y=(0,ε)

# Four regimes from nonlearnable → L3≈0.3 (χ=N equilibrium theory):
#   ρ=0.15  L3≈0.007  pre-FO, center well, nonlearnable
#   ρ=0.08  L3≈0.040  outer FO deep, center still flat
#   ρ=0.03  L3≈0.17   post catch-up, wells deepening
#   ρ=0.015 L3≈0.30   cubic learnable target
RHOS = [
    0.15,
    0.08,
    0.03,
    0.015,
]

# --- Same LR acceleration as steepwell d=5…50 trains ---
BASE_LR = 0.01
SCHEDULE_DIVISORS = "2,3,5"
EPOCHS = 60_000_000
LOG_INTERVAL = 500_000
ENSEMBLE_SIZE = 1
SEEDS = [0]
SNAPSHOT_WINDOW = 20_000_000
SNAPSHOT_A_INTERVAL = 500_000
SNAPSHOT_A_BURNIN = EPOCHS - SNAPSHOT_WINDOW

# PyTorch: cuda:0 = 4090, cuda:1 = 4080 (nvidia-smi indices are flipped).
DEVICES = ["cuda:0", "cuda:1"]
MAX_PARALLEL_BY_DEVICE = {"cuda:0": 2, "cuda:1": 2}  # 2–2 split across lab GPUs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf_sigma_a.py")
# Prefer the machine ml_env (same as steepwell_invariant trains).
PYTHON = (
    "/home/akiva/miniconda3/envs/ml_env/bin/python"
    if os.path.isfile("/home/akiva/miniconda3/envs/ml_env/bin/python")
    else sys.executable
)

SWEEP_DIR = os.path.join(
    SCRIPT_DIR,
    (
        "red_robin_pure_cubic_rho_ray"
        f"_d{D}_N{N}_P{P}_chiN_sa0{SA0:g}_eps{TASK_EPS:g}_he1{HE1:g}"
        f"_lr{BASE_LR:g}_ep{EPOCHS // 1_000_000}M"
        f"_sched{SCHEDULE_DIVISORS.replace(',', '_')}"
    ),
)
MODELS_DIR = os.path.join(SWEEP_DIR, "models")
TENSORBOARD_DIR = os.path.join(SWEEP_DIR, "tensorboard")
LOG_BASE_DIR = os.path.join(SWEEP_DIR, "launcher_logs")


def kappa_of(rho: float) -> float:
    return rho * SA0 * P


def params_of(rho: float) -> dict:
    κ = kappa_of(rho)
    return {
        "rho": float(rho),
        "kappa": float(κ),
        "temperature": float(2.0 * κ),  # T = 2κ convention
        "d": D,
        "P": P,
        "N": N,
        "chi": CHI,
        "sa0": SA0,
    }


def run_dir_name(params: dict, seed: int) -> str:
    return (
        f"purecubic_rho{params['rho']:.4g}"
        f"_d{params['d']}_P{params['P']}_N{params['N']}"
        f"_chi{params['chi']:g}_sa0{params['sa0']:.4f}"
        f"_kappa{params['kappa']:.4f}_seed{seed}"
    )


def make_cmd(params: dict, seed: int, device: str) -> list[str]:
    run_name = run_dir_name(params, seed)
    return [
        PYTHON,
        "-u",
        TRAIN_SCRIPT,
        "--d",
        str(params["d"]),
        "--P",
        str(params["P"]),
        "--N",
        str(params["N"]),
        "--chi",
        str(params["chi"]),
        "--temperature",
        str(params["temperature"]),
        "--lr",
        str(BASE_LR),
        "--device",
        device,
        "--epochs",
        str(EPOCHS),
        "--log-interval",
        str(LOG_INTERVAL),
        "--dataset-seed",
        str(seed),
        "--ens",
        str(ENSEMBLE_SIZE),
        "--eps",
        str(TASK_EPS),
        "--he1",
        str(HE1),
        "--s0",
        str(S0),
        "--sa0",
        str(params["sa0"]),
        "--schedule-divisors",
        SCHEDULE_DIVISORS,
        "--snapshot-A-interval",
        str(SNAPSHOT_A_INTERVAL),
        "--snapshot-A-burnin",
        str(SNAPSHOT_A_BURNIN),
        "--output-dir",
        os.path.join(MODELS_DIR, run_name),
        "--tensorboard-dir",
        os.path.join(TENSORBOARD_DIR, run_name),
    ]


def print_table() -> list[dict]:
    grid = [params_of(r) for r in RHOS]
    print("Pure-cubic χ=N ρ-ray  (theory-matched FO → deep L3)")
    print(
        f"  fixed: d={D} N={N} χ={CHI:g} P={P} sa0={SA0} s0={S0} "
        f"ε={TASK_EPS} he1={HE1}  (Y = he1·z + ε·He3)"
    )
    print(
        f"  LR: base_lr={BASE_LR}, schedule={SCHEDULE_DIVISORS}, "
        f"epochs={EPOCHS}, Asnap last {SNAPSHOT_WINDOW // 1_000_000}M / "
        f"{SNAPSHOT_A_INTERVAL // 1000}k"
    )
    print(
        f"  {'ρ':>8} {'κ':>10} {'T=2κ':>10} {'note'}"
    )
    notes = {
        0.15: "nonlearnable  L3≈0.007  pre-FO",
        0.08: "outer-FO deep L3≈0.040  center flat",
        0.03: "deepening     L3≈0.17   post catch-up",
        0.015: "learnable     L3≈0.30   cubic target",
    }
    for p in grid:
        print(
            f"  {p['rho']:8.4g} {p['kappa']:10.4g} {p['temperature']:10.4g}  "
            f"{notes.get(p['rho'], '')}"
        )
    print(f"\n  sweep dir: {SWEEP_DIR}")
    return grid


def device_cap(device: str) -> int:
    return int(MAX_PARALLEL_BY_DEVICE.get(device, 1))


def running_on(running: list[dict], device: str) -> int:
    return sum(
        1 for j in running if j.get("device") == device and j["proc"].poll() is None
    )


def pick_device(running: list[dict]) -> str | None:
    """Prefer the GPU with the most free slots (2–2 capacity)."""
    best, best_free = None, -1
    for dev in DEVICES:
        free = device_cap(dev) - running_on(running, dev)
        if free > best_free:
            best, best_free = dev, free
    return best if best_free > 0 else None


def launch_all(grid: list[dict]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)
    # red-robin: alternate high/low ρ (FO window first + deep)
    pending = deque(sorted(grid, key=lambda x: -x["rho"]))
    order = []
    take_hi = True
    while pending:
        order.append(pending.popleft() if take_hi else pending.pop())
        take_hi = not take_hi

    print(
        f"  devices: {DEVICES}  caps={MAX_PARALLEL_BY_DEVICE}  "
        f"python={PYTHON}"
    )
    print(
        "  red-robin order: "
        + ", ".join(f"ρ={p['rho']:.4g}" for p in order)
    )

    running: list[dict] = []
    queue = deque((p, s) for p in order for s in SEEDS)
    while queue:
        running = [j for j in running if j["proc"].poll() is None]
        device = pick_device(running)
        if device is None:
            time.sleep(30)
            continue
        params, seed = queue.popleft()
        cmd = make_cmd(params, seed, device)
        log_path = os.path.join(LOG_BASE_DIR, run_dir_name(params, seed) + ".log")
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(
                f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"device={device} =====\n"
            )
            log_f.write(" ".join(cmd) + "\n")
            log_f.flush()
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append(
            {"proc": proc, "params": params, "seed": seed, "device": device}
        )
        print(
            f"Launched ρ={params['rho']:.4g} κ={params['kappa']:.4g} "
            f"device={device} seed={seed} -> {log_path}"
        )
        time.sleep(0.5)
    print("All jobs submitted; waiting for completion...")
    for j in running:
        j["proc"].wait()
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--launch",
        action="store_true",
        help="actually start trains (default: dry-run table only)",
    )
    args = ap.parse_args()
    grid = print_table()
    if args.launch:
        launch_all(grid)
    else:
        print(
            "\nDry-run only. Pass --launch to start trains "
            f"({MAX_PARALLEL_BY_DEVICE} on {DEVICES})."
        )


if __name__ == "__main__":
    main()
