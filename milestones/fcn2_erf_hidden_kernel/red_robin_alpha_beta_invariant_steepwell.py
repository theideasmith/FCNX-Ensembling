"""Steepwell: red-robin alpha/beta sweep with invariant F and deep W0 wells.

Copy of red_robin_alpha_beta_invariant.py with SIGMA_A0=0.03 and
alpha = alpha0 * beta^gamma (gamma=1.25) so kappa stays O(1e-3) while P grows.

ens=1 throughout. After each scheduled train finishes, the launcher resumes the
same output dir with --extra-epochs + --snapshot-A-* and writes Langevin A/W0
samples under A_snapshots/ for posterior / ensemble averaging.

Scaling laws
------------
    d         = d0 * beta^{epsilon}
    N         = N0 * beta^{nu}
    sa0       = sigma_a0^2 * beta^{rho}
    P         = P0 * alpha^{lambda}
    kappa     = kappa0 * (alpha/beta)^{omega}      (T = 2 kappa)
    s0        = 1  (fixed)

    Sweep ray:  alpha = alpha0 * beta^{gamma}   (gamma >= 1 so kappa does not collapse)

Invariant expression
--------------------
    F = 2 P^2 sa0 / (pi kappa^2 d N)

Exponent constraints (F independent of alpha and beta)
------------------------------------------------------
    lambda = omega
    rho + 2 omega - epsilon - nu = 0

With rho = -omega (sa0 and kappa share beta scaling at fixed alpha):
    omega = epsilon + nu
    lambda = omega
    rho = -omega

Current choice
--------------
    epsilon = 1/2,  nu = 1/4
    =>  omega = lambda = 3/4,  rho = -3/4
    alpha0 = 1,  gamma = 5/4
    =>  kappa ~ beta^{omega(gamma-1)} gently rises
        P     ~ beta^{lambda * gamma} grows with alpha
        P/kappa ~ beta^{omega} independent of gamma
"""
import os
import sys
import subprocess
import time
from collections import deque

import numpy as np


# --- MF scaling exponents ---
EPSILON = 0.5   # d ~ beta^epsilon
NU = 0.25       # N ~ beta^nu
OMEGA = EPSILON + NU               # 3/4
RHO = -OMEGA                       # -3/4
LAMBDA = OMEGA                     # 3/4; lambda = omega for F flat in alpha

# --- Base parameters at (beta, alpha) = (1, 1) ---
D0 = 5
N0 = 1000
# Base readout prior; sa0 = SIGMA_A0 * beta^rho.
# SIGMA_A0=0.03 ⇒ deep double wells by d=50 on this ray.
SIGMA_A0 = 0.03
S0 = 1.0
KAPPA0 = 1.0 / N0
P0 = 160.0
D_MAX = 50
BETA_MAX = (D_MAX / D0) ** (1.0 / EPSILON)

# Alpha grows along the sweep so P grows and kappa does not collapse.
# gamma=1 ⇒ kappa flat; gamma>1 ⇒ kappa gently rises with beta.
ALPHA0 = 1.0
GAMMA = 1.25

NUM_SCALE_POINTS = 5

# --- Fixed training hyperparameters ---
# Hybrid: steepwell ray (small sa0) + d50-like cubic strength.
TASK_EPS = 0.5
# Loss is a SUM over P; update = (base_lr/P)*∇sum = base_lr*∇mean.
# Hold base_lr fixed across the ray (d5/d50 convention). Do NOT scale with P.
BASE_LR = 0.01
ENSEMBLE_SIZE = 1  # single Langevin chain; posterior avg via A_snapshots
# Equal-wall schedule like d50 (no stretch): wall = --epochs.
# Phases: lr0/2, lr0/3, lr0/5 on equal thirds.
SCHEDULE_DIVISORS = "2,3,5"
EPOCHS = 60_000_000  # wall budget; a bit above d50's 50M
LOG_INTERVAL = 500_000
DEVICE = "cuda:0"
SEEDS = [0]

_WALL_EPOCHS = EPOCHS  # equal-wall: no stretch

# After scheduled train finishes, resume with --extra-epochs at final lr and
# dump A/W0 under A_snapshots/ for ensemble/posterior averaging (ens=1).
SNAPSHOT_EXTRA_EPOCHS = 2_000_000
SNAPSHOT_A_INTERVAL = 20_000
SNAPSHOT_A_BURNIN = 50_000  # mix after resume before first snapshot

# Minibatch Langevin: gradients use (P/B) scaling; lr stays LR/P.
USE_MINIBATCH = False
BATCH_SIZE = 512

# Warm-start each scale point from the previous (beta, alpha) point with the
# same seed. Default OFF. When True, forces increasing-scale launch order
# (red-robin disabled).
USE_WARM_START = False

MAX_PARALLEL_JOBS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf_sigma_a.py")

SWEEP_DIR = os.path.join(
    SCRIPT_DIR,
    (
        "red_robin_alpha_beta_invariant_steepwell"
        f"_a0{ALPHA0:g}_g{GAMMA:g}_P0{int(P0)}"
        f"_betamax{int(BETA_MAX)}"
        f"_nu{NU:g}_N0{int(N0)}_sa0{SIGMA_A0:g}"
        f"_ens{ENSEMBLE_SIZE}_Asnap"
        f"_eps{TASK_EPS:g}"
        f"_lr{BASE_LR:g}_ep{EPOCHS // 1_000_000}M"
        f"_sched{SCHEDULE_DIVISORS.replace(',', '_')}"
    ),
)
MODELS_DIR = os.path.join(SWEEP_DIR, "models")
TENSORBOARD_DIR = os.path.join(SWEEP_DIR, "tensorboard")
LOG_BASE_DIR = os.path.join(SWEEP_DIR, "launcher_logs")


def run_dir_name(params: dict, seed: int) -> str:
    return (
        f"invariant_beta{params['beta']:.3f}_alpha{params['alpha']:.2f}"
        f"_d{params['d']}_P{params['P']}_N{params['N']}"
        f"_sa0{params['sa0']:.4f}_kappa{params['kappa']:.4f}"
        f"_seed{seed}"
    )


def run_model_dir(params: dict, seed: int) -> str:
    return os.path.join(MODELS_DIR, run_dir_name(params, seed))


def find_warm_start_path(params: dict, seed: int) -> str | None:
    """Prefer model_final.pt, then checkpoint.pt, then model.pt."""
    model_dir = run_model_dir(params, seed)
    for name in ("model_final.pt", "checkpoint.pt", "model.pt"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def scaled_params(beta: float, alpha: float | None = None) -> dict:
    """Compute scaled params along alpha = alpha0 * beta^gamma."""
    if alpha is None:
        alpha = ALPHA0 * (beta ** GAMMA)
    d = D0 * (beta ** EPSILON)
    n = N0 * (beta ** NU)
    sigma_a2 = SIGMA_A0 * (beta ** RHO)
    P = P0 * (alpha ** LAMBDA)
    kappa = KAPPA0 * ((alpha / beta) ** OMEGA)
    temperature = 2.0 * kappa
    chi = 1.0
    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "d": int(round(d)),
        "P": int(round(P)),
        "N": int(round(n)),
        "chi": float(chi),
        "sa0": float(sigma_a2),
        "kappa": float(kappa),
        "temperature": float(temperature),
    }


def undistributed_factor(params: dict) -> float:
    """2 P^2 sa0 / (pi kappa^2 d N), held invariant by the exponent constraints."""
    return (
        2.0 * params["P"] ** 2 * params["sa0"]
        / (np.pi * params["kappa"] ** 2 * params["d"] * params["N"])
    )


def build_scale_grid() -> list[dict]:
    """Beta sweep with alpha = alpha0 * beta^gamma."""
    betas = np.geomspace(1.0, BETA_MAX, NUM_SCALE_POINTS)
    return [scaled_params(float(b)) for b in betas]


def make_cmd(
    params: dict,
    seed: int,
    warm_start: str | None = None,
    *,
    snapshot_phase: bool = False,
) -> list[str]:
    """Build train_fcn2_erf_sigma_a.py command.

    Train phase: equal-wall schedule-divisors (no stretch).
    Snapshot phase: resume same output-dir with --extra-epochs + --snapshot-A-*.
    """
    run_name = run_dir_name(params, seed)
    cmd = [
        sys.executable,
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
        DEVICE,
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
        "--s0",
        str(S0),
        "--sa0",
        str(params["sa0"]),
        "--schedule-divisors",
        SCHEDULE_DIVISORS,
        "--output-dir",
        os.path.join(MODELS_DIR, run_name),
        "--tensorboard-dir",
        os.path.join(TENSORBOARD_DIR, run_name),
    ]
    if USE_MINIBATCH and BATCH_SIZE is not None:
        cmd.extend(["--batch-size", str(int(BATCH_SIZE))])
    if warm_start is not None:
        cmd.extend(["--warm-start", warm_start])
    if snapshot_phase:
        cmd.extend(
            [
                "--extra-epochs",
                str(SNAPSHOT_EXTRA_EPOCHS),
                "--snapshot-A-interval",
                str(SNAPSHOT_A_INTERVAL),
                "--snapshot-A-burnin",
                str(SNAPSHOT_A_BURNIN),
            ]
        )
    return cmd


def build_red_robin_order(items: list[dict]) -> list[dict]:
    """Alternately take from the low and high end of the sorted list."""
    pending = deque(sorted(items, key=lambda x: (x["beta"], x["alpha"])))
    order = []
    take_left = True
    while pending:
        order.append(pending.popleft() if take_left else pending.pop())
        take_left = not take_left
    return order


def launch(
    params: dict,
    seed: int,
    running: list,
    warm_start: str | None = None,
    *,
    snapshot_phase: bool = False,
) -> None:
    cmd = make_cmd(params, seed, warm_start=warm_start, snapshot_phase=snapshot_phase)
    run_name = run_dir_name(params, seed)
    suffix = "_snapshots" if snapshot_phase else ""
    log_file = os.path.join(LOG_BASE_DIR, f"{run_name}{suffix}.log")
    with open(log_file, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    running.append(
        {
            "proc": proc,
            "params": params,
            "seed": seed,
            "log": log_file,
            "warm_start": warm_start,
            "snapshot_phase": snapshot_phase,
        }
    )
    phase = "snapshots" if snapshot_phase else "train"
    ws = f", warm_start={warm_start}" if warm_start else ""
    print(
        f"Launched [{phase}] beta={params['beta']:.3f}, alpha={params['alpha']:.2f} "
        f"(d={params['d']}, P={params['P']}, N={params['N']}, sa0={params['sa0']:.4f}, "
        f"ens={ENSEMBLE_SIZE}), seed={seed}{ws} -> {log_file}"
    )
    time.sleep(0.5)


def try_launch_one(
    new_queue: deque,
    seed_queue: deque,
    snapshot_queue: deque,
    running: list,
) -> bool:
    """Prefer post-train snapshot jobs, then new trains / extra seeds."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    if snapshot_queue:
        params, seed = snapshot_queue.popleft()
        launch(params, seed, running, snapshot_phase=True)
        return True

    if new_queue:
        params = new_queue.popleft()
        launch(params, SEEDS[0], running, snapshot_phase=False)
        if len(SEEDS) > 1:
            seed_queue.append((params, 1))
        return True

    if seed_queue:
        params, next_idx = seed_queue.popleft()
        launch(params, SEEDS[next_idx], running, snapshot_phase=False)
        if next_idx + 1 < len(SEEDS):
            seed_queue.append((params, next_idx + 1))
        return True

    return False


def try_launch_warm_start_chain(
    scale_points: list[dict],
    next_idx_by_seed: dict,
    finished_train_keys: set,
    snapshot_queue: deque,
    running: list,
) -> bool:
    """Launch next train (warm-start) or pending snapshot if a slot is free."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    if snapshot_queue:
        params, seed = snapshot_queue.popleft()
        launch(params, seed, running, snapshot_phase=True)
        return True

    launched = False
    for seed in SEEDS:
        if len(running) >= MAX_PARALLEL_JOBS:
            break
        if any(job["seed"] == seed and not job.get("snapshot_phase") for job in running):
            continue
        idx = next_idx_by_seed[seed]
        if idx >= len(scale_points):
            continue

        params = scale_points[idx]
        warm_start = None
        if idx > 0:
            prev_key = (idx - 1, seed)
            if prev_key not in finished_train_keys:
                continue
            warm_start = find_warm_start_path(scale_points[idx - 1], seed)
            if warm_start is None:
                print(
                    f"No warm-start checkpoint for "
                    f"beta={scale_points[idx-1]['beta']:.3f}, seed={seed}; "
                    f"cold-starting beta={params['beta']:.3f}"
                )

        launch(params, seed, running, warm_start=warm_start, snapshot_phase=False)
        next_idx_by_seed[seed] = idx + 1
        launched = True
    return launched


def _n_expected_snapshots() -> int:
    return max(0, (SNAPSHOT_EXTRA_EPOCHS - SNAPSHOT_A_BURNIN) // SNAPSHOT_A_INTERVAL)


def main() -> None:
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    scale_points = build_scale_grid()

    print("=" * 60)
    print("FCN2 ERF — steepwell invariant alpha/beta MF scaling sweep")
    print("=" * 60)
    print("Scaling laws (factor 2 P^2 sa0 / (pi kappa^2 d N) held invariant):")
    print(f"  d     = {D0} * beta^{EPSILON}")
    print(f"  N     = {N0} * beta^{NU}")
    print(f"  P     = {P0} * alpha^{LAMBDA}")
    print(f"  alpha = {ALPHA0} * beta^{GAMMA}")
    print(
        f"  kappa = {KAPPA0} * (alpha/beta)^{OMEGA},  T = 2*kappa,  chi = 1"
    )
    print(f"  sa0   = {SIGMA_A0} * beta^{RHO}")
    print(f"  beta_max={BETA_MAX} (d_max={D0 * BETA_MAX ** EPSILON:.0f})")
    print()
    print(f"  {NUM_SCALE_POINTS} points on alpha∝beta^{GAMMA}:")
    for p in scale_points:
        print(
            f"    beta={p['beta']:6.3f}  alpha={p['alpha']:7.2f}  "
            f"d={p['d']:4d}  P={p['P']:5d}  N={p['N']:4d}  "
            f"sa0={p['sa0']:.4f}  kappa={p['kappa']:.6f}  T={p['temperature']:.6f}  "
            f"base_lr={BASE_LR:g}  step={BASE_LR/p['P']:.3g}  "
            f"P/kappa={p['P']/p['kappa']:.1f}  factor={undistributed_factor(p):.4f}"
        )
    print()
    print(f"  s0={S0} (fixed), task_eps={TASK_EPS}, ens={ENSEMBLE_SIZE}")
    print(
        f"  base_lr={BASE_LR:g} fixed (update ~ base_lr·∇mean; phase-1 uses "
        f"/{SCHEDULE_DIVISORS.split(',')[0]} → effective {BASE_LR / float(SCHEDULE_DIVISORS.split(',')[0]):g})"
    )
    print(
        f"  epochs={EPOCHS} wall with --schedule-divisors {SCHEDULE_DIVISORS} "
        f"(equal thirds, no stretch; like d50), log_interval={LOG_INTERVAL}, "
        f"device={DEVICE}, seeds={SEEDS}"
    )
    print(
        f"  post-train Langevin samples: ens={ENSEMBLE_SIZE}, then --extra-epochs="
        f"{SNAPSHOT_EXTRA_EPOCHS} with --snapshot-A-interval={SNAPSHOT_A_INTERVAL}, "
        f"--snapshot-A-burnin={SNAPSHOT_A_BURNIN} "
        f"(~{_n_expected_snapshots()} A/W0 snapshots under A_snapshots/)"
    )
    if USE_MINIBATCH and BATCH_SIZE is not None:
        print(f"  minibatch: batch_size={BATCH_SIZE} with P/B gradient scaling")
    else:
        print("  minibatch: OFF (full batch)")
    if USE_WARM_START:
        print("  warm-start: ON (increasing scale order; each point inits from previous)")
    else:
        print("  warm-start: OFF (red-robin launch order)")
    print(f"  sweep directory: {SWEEP_DIR}")
    print(f"  models:          {MODELS_DIR}")
    print(f"  tensorboard:     {TENSORBOARD_DIR}")
    print(f"  launcher logs:   {LOG_BASE_DIR}")
    print()

    running = []
    completed = []
    snapshot_queue: deque = deque()

    if USE_WARM_START:
        next_idx_by_seed = {seed: 0 for seed in SEEDS}
        finished_train_keys: set = set()
        for seed in SEEDS:
            for idx, params in enumerate(scale_points):
                if find_warm_start_path(params, seed) is None:
                    break
                finished_train_keys.add((idx, seed))
                next_idx_by_seed[seed] = idx + 1
            if next_idx_by_seed[seed] > 0:
                print(
                    f"Seed {seed}: resuming warm-start chain at scale index "
                    f"{next_idx_by_seed[seed]} / {len(scale_points)}"
                )

        while try_launch_warm_start_chain(
            scale_points, next_idx_by_seed, finished_train_keys, snapshot_queue, running
        ):
            pass

        while (
            any(idx < len(scale_points) for idx in next_idx_by_seed.values())
            or snapshot_queue
            or running
        ):
            for job in running[:]:
                ret = job["proc"].poll()
                if ret is not None:
                    running.remove(job)
                    completed.append(job)
                    p = job["params"]
                    phase = "snapshots" if job.get("snapshot_phase") else "train"
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    print(
                        f"Completed [{phase}] beta={p['beta']:.3f}, alpha={p['alpha']:.2f} "
                        f"(d={p['d']}, P={p['P']}, N={p['N']}), seed={job['seed']} [{status}]"
                    )
                    if not job.get("snapshot_phase"):
                        scale_idx = next(
                            i for i, sp in enumerate(scale_points)
                            if sp["beta"] == p["beta"] and sp["alpha"] == p["alpha"]
                        )
                        finished_train_keys.add((scale_idx, job["seed"]))
                        if ret == 0:
                            snapshot_queue.append((p, job["seed"]))

            while try_launch_warm_start_chain(
                scale_points, next_idx_by_seed, finished_train_keys, snapshot_queue, running
            ):
                pass

            time.sleep(2)
    else:
        red_robin_order = build_red_robin_order(scale_points)
        new_queue = deque(red_robin_order)
        seed_queue = deque()

        while try_launch_one(new_queue, seed_queue, snapshot_queue, running):
            pass

        while new_queue or seed_queue or snapshot_queue or running:
            for job in running[:]:
                ret = job["proc"].poll()
                if ret is not None:
                    running.remove(job)
                    completed.append(job)
                    p = job["params"]
                    phase = "snapshots" if job.get("snapshot_phase") else "train"
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    print(
                        f"Completed [{phase}] beta={p['beta']:.3f}, alpha={p['alpha']:.2f} "
                        f"(d={p['d']}, P={p['P']}, N={p['N']}), seed={job['seed']} [{status}]"
                    )
                    if not job.get("snapshot_phase") and ret == 0:
                        snapshot_queue.append((p, job["seed"]))

            while try_launch_one(new_queue, seed_queue, snapshot_queue, running):
                pass

            time.sleep(2)

    print("=" * 60)
    print(f"All jobs completed. {len(completed)} jobs run.")
    print(f"Sweep outputs saved under: {SWEEP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
