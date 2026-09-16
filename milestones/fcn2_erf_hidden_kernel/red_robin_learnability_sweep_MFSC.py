import os
import subprocess
import time
from collections import deque

import numpy as np


# Hyperparameters
D = 150
P = 600
TEMPERATURE = 0.1
LR = 5e-6
ENSEMBLE_SIZE = 10
EPOCHS = 20_000_000
EPS = 0.03
DEVICE = "cuda:0"

# Sweep variables (same as train_minigrokking.sh)
CHI_VALUES = [700, 1400, 3800]
SEEDS = [0, 42]

MAX_PARALLEL_JOBS = 4

SCRIPT_DIR = os.path.dirname(__file__)
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf.py")
LOG_BASE_DIR = os.path.join(
    SCRIPT_DIR,
    f"launcher_logs_red_robin_D{D}_P{P}_T{TEMPERATURE}_lr{LR}_ens{ENSEMBLE_SIZE}",
)


def make_cmd(chi, seed):
    """Build the train_fcn2_erf.py command for a single chi and seed."""
    n = chi
    return [
        "python3",
        TRAIN_SCRIPT,
        "--d",
        str(D),
        "--P",
        str(P),
        "--chi",
        str(chi),
        "--temperature",
        str(TEMPERATURE),
        "--N",
        str(n),
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
    ]


def launch_for_end(pending, end, running):
    """Launch one chi-batch (all seeds) for the requested queue end."""
    if not pending:
        return

    chi = pending.popleft() if end == "left" else pending.pop()
    n = chi
    for seed in SEEDS:
        cmd = make_cmd(chi, seed)
        log_file = os.path.join(
            LOG_BASE_DIR,
            f"d{D}_P{P}_N{n}_chi{chi}_T{TEMPERATURE}_seed{seed}.log",
        )
        with open(log_file, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append({"proc": proc, "chi": chi, "N": n, "end": end, "seed": seed, "log": log_file})
        print(f"Launched chi={chi} (N={n}), seed={seed}, end={end} -> {log_file}")
        time.sleep(0.5)


def main():
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    chi_values = np.unique(np.sort(CHI_VALUES)).tolist()

    print("=" * 42)
    print("Training FCN2 ERF Network - Red Robin Sweep")
    print("=" * 42)
    print("Hyperparameters:")
    print(f"  D={D}, P={P}")
    print(f"  T={TEMPERATURE}, lr={LR}, ens={ENSEMBLE_SIZE}")
    print(f"  epochs={EPOCHS}, eps={EPS}, device={DEVICE}")
    print(f"  chi/N values: {chi_values}")
    print(f"  seeds per chi: {SEEDS}")
    print(f"  launcher logs: {LOG_BASE_DIR}")
    print()

    pending = deque(chi_values)
    running = []
    completed = []

    # Launch jobs at both ends (red-robin over chi values).
    for end in ["left", "right"]:
        launch_for_end(pending, end, running)

    while pending or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is not None:
                running.remove(job)
                completed.append(job)
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(
                    f"Completed chi={job['chi']} (N={job['N']}), seed={job['seed']}, "
                    f"end={job['end']} [{status}]"
                )

                # Launch the next P from the same end whenever that end frees up.
                launch_for_end(pending, job["end"], running)

                # Soft cap to avoid accidental oversubscription if parameters change.
                while len(running) > MAX_PARALLEL_JOBS:
                    time.sleep(2)

        time.sleep(2)

    print("=" * 42)
    print(f"All jobs completed. {len(completed)} jobs run.")
    print(f"Launcher logs saved to: {LOG_BASE_DIR}")
    print("=" * 42)


if __name__ == "__main__":
    main()
