"""Learnable-cubic P sweep at fixed d=45 (see LEARNABLE_CUBIC_PLAN.md).

d, N, sa0, ε and bare κ are fixed; only P grows (500 → 32000). κ_eff = 0.01 from
the plan would need a negative bare κ for P ≤ 2000, so a single bare κ is used
for every P and κ_eff(P) is printed at startup.

χ = N is the feature-learning readout (σ_A² = sa0/(N χ) = 1/N²). The earlier
χ = 1 sweep is kept in a separate directory (no `_chiN` tag).

ens=1. In-train A/W0 Langevin snapshots every 500k over the last 20M of the
60M wall budget. Red-robin launch order over P.

    python -u red_robin_learnable_cubic_d45.py [--dry-run]
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

D = 45
N = 1024
SA0 = 1.0
S0 = 1.0
CHI = float(N)  # feature learning; lazy was CHI = 1.0
TASK_EPS = 0.1
KAPPA_BARE = 0.005  # fixed for all P; T = 2*kappa
TEMPERATURE = 2.0 * KAPPA_BARE
P_VALUES = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 32000]

# Loss is a SUM over P; update = (base_lr/P)*∇sum = base_lr*∇mean. Not scaled with P.
BASE_LR = 0.01
SCHEDULE_DIVISORS = "2,3,5"
EPOCHS = 60_000_000
LOG_INTERVAL = 500_000
SNAPSHOT_WINDOW = 20_000_000
SNAPSHOT_A_INTERVAL = 500_000
SNAPSHOT_A_BURNIN = EPOCHS - SNAPSHOT_WINDOW
ENSEMBLE_SIZE = 1
SEEDS = [0]
DEVICE = "cuda:0"  # 4090
MAX_PARALLEL_JOBS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf_sigma_a.py")

SWEEP_DIR = os.path.join(
    SCRIPT_DIR,
    (
        f"red_robin_learnable_cubic_d{D}_N{N}_sa0{SA0:g}_chi{CHI:g}_kappa{KAPPA_BARE:g}"
        f"_eps{TASK_EPS:g}_ens{ENSEMBLE_SIZE}"
        f"_Asnap{SNAPSHOT_WINDOW // 1_000_000}M_{SNAPSHOT_A_INTERVAL // 1000}k"
        f"_lr{BASE_LR:g}_ep{EPOCHS // 1_000_000}M"
        f"_sched{SCHEDULE_DIVISORS.replace(',', '_')}"
    ),
)
MODELS_DIR = os.path.join(SWEEP_DIR, "models")
TENSORBOARD_DIR = os.path.join(SWEEP_DIR, "tensorboard")
LOG_BASE_DIR = os.path.join(SWEEP_DIR, "launcher_logs")


def run_dir_name(P: int, seed: int) -> str:
    return (
        f"d{D}_P{P}_N{N}_sa0{SA0:g}_chi{CHI:g}_kappa{KAPPA_BARE:.4f}"
        f"_eps{TASK_EPS:g}_seed{seed}"
    )


class PidProc:
    """Wait handle for a train process this launcher did not spawn."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        except PermissionError:
            return None
        return None


def find_train_pid(output_dir: str) -> int | None:
    needle = os.path.abspath(output_dir)
    try:
        pids = os.listdir("/proc")
    except OSError:
        return None
    for name in pids:
        if not name.isdigit():
            continue
        try:
            raw = open(os.path.join("/proc", name, "cmdline"), "rb").read()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        if "train_fcn2_erf_sigma_a.py" in cmd and needle in cmd:
            return int(name)
    return None


def kappa_eff_table(p_values: list[int]) -> dict[int, float]:
    """Forward κ_eff(P) for KAPPA_BARE: κ_eff = κ + Σ λ (κ_eff/P)/(λ + κ_eff/P)."""
    from scipy.optimize import brentq

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "lib"))
    from kappa_eff_solver import compute_arcsin_eigenvalues

    lam = np.clip(compute_arcsin_eigenvalues(d=D, num_samples=5000, device="cpu"), 0.0, None)
    lam = lam.astype(np.float64)
    out = {}
    for P in p_values:
        def f(k):
            r = k / P
            return k - KAPPA_BARE - np.sum(lam * r / (lam + r))
        out[P] = float(brentq(f, KAPPA_BARE, KAPPA_BARE + lam.sum() + 1.0))
    return out


def make_cmd(P: int, seed: int, device: str | None = None) -> list[str]:
    run_name = run_dir_name(P, seed)
    return [
        sys.executable,
        "-u",
        TRAIN_SCRIPT,
        "--d", str(D),
        "--P", str(P),
        "--N", str(N),
        "--chi", str(CHI),
        "--temperature", str(TEMPERATURE),
        "--lr", str(BASE_LR),
        "--device", device or DEVICE,
        "--epochs", str(EPOCHS),
        "--log-interval", str(LOG_INTERVAL),
        "--dataset-seed", str(seed),
        "--ens", str(ENSEMBLE_SIZE),
        "--eps", str(TASK_EPS),
        "--s0", str(S0),
        "--sa0", str(SA0),
        "--schedule-divisors", SCHEDULE_DIVISORS,
        "--output-dir", os.path.join(MODELS_DIR, run_name),
        "--tensorboard-dir", os.path.join(TENSORBOARD_DIR, run_name),
        "--snapshot-A-interval", str(SNAPSHOT_A_INTERVAL),
        "--snapshot-A-burnin", str(SNAPSHOT_A_BURNIN),
    ]


def build_red_robin_order(items: list[int]) -> list[int]:
    """Alternately take from the low and high end of the sorted list."""
    pending = deque(sorted(items))
    order = []
    take_left = True
    while pending:
        order.append(pending.popleft() if take_left else pending.pop())
        take_left = not take_left
    return order


def launch(P: int, seed: int, running: list) -> None:
    run_name = run_dir_name(P, seed)
    log_file = os.path.join(LOG_BASE_DIR, f"{run_name}.log")
    existing_pid = find_train_pid(os.path.join(MODELS_DIR, run_name))
    if existing_pid is not None:
        running.append({"proc": PidProc(existing_pid), "P": P, "seed": seed, "log": log_file})
        print(f"Attached pid={existing_pid} P={P}, seed={seed}")
        return
    with open(log_file, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} device={DEVICE} =====\n"
        )
        log_f.flush()
        proc = subprocess.Popen(make_cmd(P, seed), stdout=log_f, stderr=subprocess.STDOUT)
    running.append({"proc": proc, "P": P, "seed": seed, "log": log_file})
    print(f"Launched device={DEVICE} P={P}, seed={seed} -> {log_file}")
    time.sleep(0.5)


def try_launch_one(new_queue: deque, seed_queue: deque, running: list) -> bool:
    """Fill one free slot; a new P wins over the next seed of a started P."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False
    if new_queue:
        P = new_queue.popleft()
        launch(P, SEEDS[0], running)
        if len(SEEDS) > 1:
            seed_queue.append((P, 1))
        return True
    if seed_queue:
        P, next_idx = seed_queue.popleft()
        launch(P, SEEDS[next_idx], running)
        if next_idx + 1 < len(SEEDS):
            seed_queue.append((P, next_idx + 1))
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print settings and commands only")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    order = build_red_robin_order(P_VALUES)

    print("=" * 60)
    print(f"FCN2 ERF — learnable cubic P sweep, d={D}, chi={CHI:g} (FL, chi=N)")
    print("=" * 60)
    print(
        f"  d={D}, N={N}, sa0={SA0:g}, s0={S0:g}, chi={CHI:g}, eps={TASK_EPS:g}, "
        f"kappa_bare={KAPPA_BARE:g} (T={TEMPERATURE:g}), ens={ENSEMBLE_SIZE}"
    )
    try:
        keff = kappa_eff_table(P_VALUES)
    except Exception as exc:
        print(f"  kappa_eff table failed ({exc})")
        keff = {}
    for P in P_VALUES:
        ke = keff.get(P, float("nan"))
        print(
            f"    P={P:6d}  eta={np.log(P) / np.log(D):.3f}  kappa_eff={ke:.5f}  "
            f"rho'={ke / (SA0 * P):.3g}  step={BASE_LR / P:.3g}"
        )
    print(f"  red-robin order: {order}")
    print(
        f"  epochs={EPOCHS} with --schedule-divisors {SCHEDULE_DIVISORS}, "
        f"lr={BASE_LR:g}, log_interval={LOG_INTERVAL}, device={DEVICE}, "
        f"max_parallel={MAX_PARALLEL_JOBS}, seeds={SEEDS}"
    )
    print(
        f"  A/W0 snaps every {SNAPSHOT_A_INTERVAL} over last {SNAPSHOT_WINDOW} "
        f"(burnin={SNAPSHOT_A_BURNIN}; ~{1 + SNAPSHOT_WINDOW // SNAPSHOT_A_INTERVAL} snaps)"
    )
    print(f"  sweep directory: {SWEEP_DIR}")
    print()

    if args.dry_run:
        for P in order:
            print(" ".join(make_cmd(P, SEEDS[0])))
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    new_queue = deque(order)
    seed_queue: deque = deque()
    running: list = []
    completed: list = []

    while try_launch_one(new_queue, seed_queue, running):
        pass

    while new_queue or seed_queue or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is not None:
                running.remove(job)
                completed.append(job)
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"Completed P={job['P']}, seed={job['seed']} [{status}]")
        while try_launch_one(new_queue, seed_queue, running):
            pass
        time.sleep(2)

    print("=" * 60)
    print(f"All jobs completed. {len(completed)} jobs run.")
    print(f"Sweep outputs saved under: {SWEEP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
