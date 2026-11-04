#!/usr/bin/env python3
"""Eigenframe diag-Q theory P-ray → matching red-robin trains.

Background
----------
The mixture-saddle eigenframe solve
(`eigenframe_diag_q_action.jl`, plots in `action_h0_activation_plots/EigenframeDiagQ/`)
used

    d=10, N=χ=400, sa0=1, κ=0.5, ε=1, ỹ=(0,ε)

and scanned

    P ∈ {50, 100, 200, 500, 1000, 2000, 5000}   ⇔   α = log(P)/log(d)

with ρ' = κ/(sa0 P). Lab teacher is y = U ỹ ≈ (0.2–0.3, 0.95–0.98);
trains below use he1=0 (pure cubic Y=ε He3) as the fixed proxy for ỹ=(0,ε)
— same convention as `red_robin_pure_cubic_rho_ray_chiN.py`.

This launcher
-------------
Same red-robin / dual-GPU pattern as
`red_robin_pure_cubic_rho_ray_chiN.py` and the steepwell α–β invariants.

Fixed: d, N, χ=N, sa0, κ (⇒ T=2κ), ε, he1=0.
Scan: P along the theory list (α-ray at fixed d).

Same LR recipe as accelerated steepwell / pure-cubic ρ-ray:
  base_lr=0.01, schedule 2,3,5, 60M wall, A snaps last 20M / 500k.

Usage (do not start until GPU is free / user asks)::

    python red_robin_eigenframe_diag_q_p_ray.py            # dry-run table
    python red_robin_eigenframe_diag_q_p_ray.py --launch   # actually train
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque

# --- Theory-matched fixed point (EigenframeDiagQ mixture saddle) ---
D = 10
N = 400
CHI = float(N)  # χ=N
SA0 = 1.0
S0 = 1.0
KAPPA = 0.5
TASK_EPS = 1.0
HE1 = 0.0  # proxy for ỹ=(0,ε); lab eigenframe y = Uỹ has small He1

# Same P grid as eigenframe_diag_q_action.jl / learnability_vs_alpha
PS = [50, 100, 200, 500, 1000, 2000, 5000]

# Theory L̃₂ from summary_mix_d10_N400_chi400.csv (for the dry-run table)
THEORY_L2 = {
    50: 0.244,
    100: 0.467,
    200: 0.626,
    500: 0.816,
    1000: 0.901,
    2000: 0.947,
    5000: 0.978,
}
THEORY_PI_OUT = {
    50: 0.0,
    100: 0.0,
    200: 1.0,
    500: 1.0,
    1000: 1.0,
    2000: 0.0,
    5000: 0.0,
}

# --- Same LR acceleration as steepwell / pure-cubic ρ-ray ---
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
MAX_PARALLEL_BY_DEVICE = {"cuda:0": 2, "cuda:1": 2}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf_sigma_a.py")
PYTHON = (
    "/home/akiva/miniconda3/envs/ml_env/bin/python"
    if os.path.isfile("/home/akiva/miniconda3/envs/ml_env/bin/python")
    else sys.executable
)

SWEEP_DIR = os.path.join(
    SCRIPT_DIR,
    (
        "red_robin_eigenframe_diag_q_p_ray"
        f"_d{D}_N{N}_chiN_sa0{SA0:g}_kappa{KAPPA:g}"
        f"_eps{TASK_EPS:g}_he1{HE1:g}"
        f"_lr{BASE_LR:g}_ep{EPOCHS // 1_000_000}M"
        f"_sched{SCHEDULE_DIVISORS.replace(',', '_')}"
    ),
)
MODELS_DIR = os.path.join(SWEEP_DIR, "models")
TENSORBOARD_DIR = os.path.join(SWEEP_DIR, "tensorboard")
LOG_BASE_DIR = os.path.join(SWEEP_DIR, "launcher_logs")


def alpha_of(P: float) -> float:
    return float(math.log(P) / math.log(D))


def params_of(P: int) -> dict:
    ρp = KAPPA / (SA0 * P)
    return {
        "P": int(P),
        "alpha": alpha_of(P),
        "rho_p": float(ρp),
        "kappa": float(KAPPA),
        "temperature": float(2.0 * KAPPA),  # T = 2κ
        "d": D,
        "N": N,
        "chi": CHI,
        "sa0": SA0,
        "theory_L2": THEORY_L2.get(P, float("nan")),
        "theory_pi_out": THEORY_PI_OUT.get(P, float("nan")),
    }


def run_dir_name(params: dict, seed: int) -> str:
    return (
        f"eigdiag_P{params['P']}_a{params['alpha']:.3f}"
        f"_d{params['d']}_N{params['N']}"
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
    grid = [params_of(P) for P in PS]
    print("Eigenframe diag-Q P-ray  (matched to EigenframeDiagQ theory)")
    print(
        f"  fixed: d={D} N={N} χ={CHI:g} sa0={SA0} κ={KAPPA} "
        f"T={2*KAPPA:g} s0={S0} ε={TASK_EPS} he1={HE1}  "
        f"(Y = he1·z + ε·He3; theory ỹ=(0,ε))"
    )
    print(
        f"  LR: base_lr={BASE_LR}, schedule={SCHEDULE_DIVISORS}, "
        f"epochs={EPOCHS}, Asnap last {SNAPSHOT_WINDOW // 1_000_000}M / "
        f"{SNAPSHOT_A_INTERVAL // 1000}k"
    )
    print(
        f"  {'P':>6} {'α':>7} {'ρ′':>10} {'T=2κ':>8} "
        f"{'th L̃₂':>8} {'π_out':>6}  note"
    )
    for p in grid:
        note = "outer wells" if p["theory_pi_out"] > 0.5 else "inner ±μ"
        print(
            f"  {p['P']:6d} {p['alpha']:7.3f} {p['rho_p']:10.4g} "
            f"{p['temperature']:8.4g} {p['theory_L2']:8.3f} "
            f"{p['theory_pi_out']:6.1f}  {note}"
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
    # red-robin: alternate large/small P (learnable first + nonlearnable)
    pending = deque(sorted(grid, key=lambda x: -x["P"]))
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
        + ", ".join(f"P={p['P']}" for p in order)
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
            f"Launched P={params['P']} α={params['alpha']:.3f} "
            f"ρ′={params['rho_p']:.4g} device={device} seed={seed} -> {log_path}"
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
