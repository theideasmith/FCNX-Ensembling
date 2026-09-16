import os
import sys
import subprocess
import time
from collections import deque

import numpy as np


# Hyperparameters
D = 100
CHI = 700
N = 700
P_MIN = 100
P_MAX = 4000
NUM_P = 14  # Equidistributed from P=100 to P=4000 with step=300

# Initial parameters
PI = 250.0
PF = 1000.0
num_values = 4

# Step 1: Generate 4 equally spaced values for P between PI and PF
step_P = (PF - PI) / (num_values - 1)
P_list = [PI + i * step_P for i in range(num_values)]

# Step 2: Compute L for each P using P = 250 * sqrt(L) => L = (P / 250)^2
L_list = [(P / 250.0) ** 2 for P in P_list]

# Output parameters given in the problem
L_max = L_list[-1]  # 16.0
S0_List = [1.0 / L for L in L_list]  # 1.0 / 16.0 = 0.0625


TEMPERATURE = 0.1
LR = 5e-3
ENSEMBLE_SIZE = 10
EPOCHS = 10000000
EPS = 0.074
DEVICE = "cuda:1"

# Sweep over P values
# P_VALUES = [100, 400, 700] #4000]  # np.linspace(P_MIN, P_MAX, NUM_P, dtype=int).tolist()
P_VALUES = P_list
SEEDS = [0]

MAX_PARALLEL_JOBS = 6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf.py")
LOG_BASE_DIR = os.path.join(
    SCRIPT_DIR,
    f"launcher_logs_red_robin_sample_complexity_D{D}_chi{CHI}_eps{EPS}_T{TEMPERATURE}_lr{LR}_ens{ENSEMBLE_SIZE}",
)


def make_cmd(p_val, s0_val, seed):
    """Build the train_fcn2_erf.py command for a single P and seed."""
    return [
        sys.executable,
        TRAIN_SCRIPT,
        "--d",
        str(D),
        "--P",
        str(int(p_val)),
        "--chi",
        str(CHI),
        "--temperature",
        str(TEMPERATURE),
        "--N",
        str(N),
        "--lr",
        str(LR),
        "--device",
        DEVICE,
        "--epochs",
        str(EPOCHS),
        "--dataset-seed",
        str(seed),
        "--ens",
        str(ENSEMBLE_SIZE),
        "--eps",
        str(EPS),
        "--s0",
        str(s0_val)
    ]


def build_p_order(p_values):
    """Red-robin ordering of P values: alternately take from the low and high
    end of the sorted list, so the P-spread is covered as early as possible."""
    pending = deque(p_values)
    order = []
    take_left = True
    while pending:
        order.append(pending.popleft() if take_left else pending.pop())
        take_left = not take_left
    return order


def launch(p_val, s0_val, seed, running):
    cmd = make_cmd(p_val, s0_val, seed)
    log_file = os.path.join(
        LOG_BASE_DIR,
        f"d{D}_P{p_val}_N{N}_chi{CHI}_T{TEMPERATURE}_seed{seed}_eps{EPS}.log",
    )
    with open(log_file, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    running.append({"proc": proc, "P": p_val, "S0": s0_val, "chi": CHI, "N": N, "seed": seed, "log": log_file})
    print(f"Launched P={p_val}, S0={s0_val} (d={D}, chi={CHI}, N={N}), seed={seed} -> {log_file}")
    time.sleep(0.5)


def try_launch_one(new_p_queue, seed_queue, running):
    """Fill one free slot. A brand-new P always wins over advancing an
    already-started P to its next seed."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    if new_p_queue:
        p_val, s0_val = new_p_queue.popleft()
        launch(p_val, s0_val, SEEDS[0], running)
        if len(SEEDS) > 1:
            seed_queue.append((p_val, s0_val, 1))
        return True

    if seed_queue:
        p_val, s0_val, next_idx = seed_queue.popleft()
        launch(p_val, s0_val, SEEDS[next_idx], running)
        if next_idx + 1 < len(SEEDS):
            seed_queue.append((p_val, s0_val, next_idx + 1))
        return True

    return False


def main():
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    p_values = sorted(list(set(P_VALUES)))
    s0_by_p = {p_val: s0_val for p_val, s0_val in zip(P_list, S0_List)}

    print("=" * 42)
    print("Training FCN2 ERF Network - Sample Complexity Sweep (Cubic Task)")
    print("=" * 42)
    print("Hyperparameters:")
    print(f"  D={D}, chi={CHI}, N={N}")
    print(f"  P values ({len(p_values)} points): {p_values}")
    print(f"  S0 values by P: {[s0_by_p[p_val] for p_val in p_values]}")
    print(f"  T={TEMPERATURE}, lr={LR}, ens={ENSEMBLE_SIZE}")
    print(f"  epochs={EPOCHS}, eps={EPS}, device={DEVICE}")
    print(f"  seeds: {SEEDS} (new P always takes priority over next seed of an existing P)")
    print(f"  launcher logs: {LOG_BASE_DIR}")
    print()

    new_p_queue = deque((p_val, s0_by_p[p_val]) for p_val in build_p_order(p_values))
    seed_queue = deque()  # (p_val, s0_val, next_seed_idx) for P's that have started but have seeds left
    running = []
    completed = []

    # Fill as many slots as we can up front.
    while try_launch_one(new_p_queue, seed_queue, running):
        pass

    while new_p_queue or seed_queue or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is not None:
                running.remove(job)
                completed.append(job)
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(
                    f"Completed P={job['P']} (S0={job['S0']}, chi={job['chi']}, N={job['N']}), seed={job['seed']} [{status}]"
                )

        # Backfill freed slots, new-P first, then queued next-seed jobs.
        while try_launch_one(new_p_queue, seed_queue, running):
            pass

        time.sleep(2)

    print("=" * 42)
    print(f"All jobs completed. {len(completed)} jobs run.")
    print(f"Launcher logs saved to: {LOG_BASE_DIR}")
    print("=" * 42)


if __name__ == "__main__":
    main()