"""Red-robin alpha/beta sweep with an invariant quadratic-overlap prefactor.

Scaling laws
------------
    d         = d0 * beta^{epsilon}
    N         = N0 * beta^{nu}
    sigma_a^2 = sa0 = sigma_a0^2 * beta^{rho}
    P         = P0 * alpha^{lambda}
    kappa     = kappa0 * (alpha/beta)^{omega}      (sigma^2 = kappa; T = 2 kappa)
    s0        = 1  (fixed)

Invariant expression
--------------------
    F = 2 P^2 sa0 / (pi kappa^2 d N)

Derivation of exponent constraints
----------------------------------
    P^2     ~ alpha^{2 lambda}
    sa0     ~ beta^{rho}
    kappa^2 ~ (alpha/beta)^{2 omega} = alpha^{2 omega} * beta^{-2 omega}
    d       ~ beta^{epsilon}
    N       ~ beta^{nu}

    F ~ alpha^{2 lambda - 2 omega} * beta^{rho + 2 omega - epsilon - nu}

F independent of alpha and beta iff:

    2 lambda - 2 omega = 0          =>  lambda = omega
    rho + 2 omega - epsilon - nu = 0

(Note: the beta condition uses -nu because N is in the denominator of F.
 A written "+ nu" would not keep F flat.)

Imposed matching of kappa and sa0 in beta (fixed alpha):
    kappa ~ beta^{-omega},  sa0 ~ beta^{rho}  =>  rho = -omega
(i.e. omega = -rho).

Then
    -omega + 2 omega - epsilon - nu = 0
    omega = epsilon + nu
    lambda = omega
    rho = -omega

Current choice
--------------
    epsilon = 1/2,  nu = 1/4
    =>  omega = 3/4,  lambda = 3/4,  rho = -3/4
    =>  d ~ beta^{1/2},  N ~ beta^{1/4},  sa0 ~ beta^{-3/4},
        P ~ alpha^{3/4},  kappa ~ (alpha/beta)^{3/4}
    At fixed alpha: kappa and sa0 both ~ beta^{-3/4}.

Sweep mode
----------
    Alpha is held fixed by choosing the physical sample size P = P_FIXED.
        alpha = (P_FIXED / P0)^{1/lambda}
    Only beta is swept (geomspace from 1 to beta_max). Along this line,
    P is constant, while d, N, sa0, kappa (and T) vary with beta.
"""
import os
import sys
import subprocess
import time
from collections import deque

import numpy as np


# --- MF scaling exponents ---
# F = 2 P^2 sa0 / (pi kappa^2 d N) invariant under:
#   lambda = omega,
#   rho + 2 omega - epsilon - nu = 0.
# With rho = -omega (kappa and sa0 share beta scaling at fixed alpha):
#   omega = epsilon + nu.
EPSILON = 0.5   # d ~ beta^epsilon
NU = 0.25       # N ~ beta^nu
OMEGA = EPSILON + NU               # 3/4
RHO = -OMEGA                       # -3/4; omega = -rho
LAMBDA = OMEGA                     # 3/4; lambda = omega

# --- Base parameters at (beta, alpha) = (1, 1) ---
D0 = 5
N0 = 1000
SIGMA_A0 = 1.0
S0 = 1.0
KAPPA0 = 1.0 / N0
D_MAX = 50
P0 = 160.0
P_FIXED = 1500  # physical P; determines alpha and is held fixed in the sweep
BETA_MAX = (D_MAX / D0) ** (1.0 / EPSILON)
ALPHA_FIXED = (P_FIXED / P0) ** (1.0 / LAMBDA)

NUM_SCALE_POINTS = 5

# --- Fixed training hyperparameters ---
TASK_EPS = 0.03
LR = 5e-3
ENSEMBLE_SIZE = 1
EPOCHS = 1_000_000  # effective budget under --schedule (wall ≈ 3.6×)
USE_SCHEDULE = True
LOG_INTERVAL = 100_000
DEVICE = "cuda:0"
SEEDS = [0]

# Late Langevin A/W0 snapshots (posterior averaging with ens=1).
# Collect ~50 snapshots over the final 1M *wall* epochs.
SNAPSHOT_A_INTERVAL = 20_000
_WALL_EPOCHS = int(np.ceil(3.6 * EPOCHS)) if USE_SCHEDULE else EPOCHS
SNAPSHOT_A_BURNIN = max(0, _WALL_EPOCHS - 1_000_000)

# Minibatch Langevin: gradients use (P/B) scaling; lr stays LR/P.
USE_MINIBATCH = False
BATCH_SIZE = 512

# Warm-start each scale point from the previous (beta, alpha) point with the
# same seed. Default OFF. When True, forces increasing-scale launch order
# (red-robin disabled).
USE_WARM_START = False

MAX_PARALLEL_JOBS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_fcn2_erf_sigma_a.py")

SWEEP_DIR = os.path.join(
    SCRIPT_DIR,
    (
        "red_robin_alpha_beta_invariant"
        f"_Pfixed{int(P_FIXED)}_betamax{int(BETA_MAX)}"
        f"_nu{NU:g}_N0{int(N0)}_lam_eq_omega_alpha_fixed"
        f"_ens{ENSEMBLE_SIZE}_Asnap"
        f"_eps{TASK_EPS:g}"
        f"{'_schedule' if USE_SCHEDULE else ''}"
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


def scaled_params(beta: float, alpha: float) -> dict:
    """Compute all scaled network/training parameters for a (beta, alpha) pair."""
    d = D0 * (beta ** EPSILON)
    n = N0 * (beta ** NU)
    sigma_a2 = SIGMA_A0 * (beta ** RHO)
    p_val = P0 * (alpha ** LAMBDA)
    kappa = KAPPA0 * ((alpha / beta) ** OMEGA)
    temperature = 2.0 * kappa
    chi = 1.0
    return {
        "beta": beta,
        "alpha": alpha,
        "d": int(round(d)),
        "P": int(round(p_val)),
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
    """Beta-only sweep at fixed alpha (from P_FIXED)."""
    betas = np.geomspace(1.0, BETA_MAX, NUM_SCALE_POINTS)
    return [scaled_params(float(b), float(ALPHA_FIXED)) for b in betas]


def make_cmd(params: dict, seed: int, warm_start: str | None = None) -> list[str]:
    """Build the train_fcn2_erf_sigma_a.py command for one scaled run."""
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
        str(LR),
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
        "--output-dir",
        os.path.join(MODELS_DIR, run_name),
        "--tensorboard-dir",
        os.path.join(TENSORBOARD_DIR, run_name),
        "--snapshot-A-interval",
        str(SNAPSHOT_A_INTERVAL),
        "--snapshot-A-burnin",
        str(SNAPSHOT_A_BURNIN),
    ]
    if USE_SCHEDULE:
        cmd.append("--schedule")
    if USE_MINIBATCH and BATCH_SIZE is not None:
        cmd.extend(["--batch-size", str(int(BATCH_SIZE))])
    if warm_start is not None:
        cmd.extend(["--warm-start", warm_start])
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


def launch(params: dict, seed: int, running: list, warm_start: str | None = None) -> None:
    cmd = make_cmd(params, seed, warm_start=warm_start)
    run_name = run_dir_name(params, seed)
    log_file = os.path.join(LOG_BASE_DIR, f"{run_name}.log")
    with open(log_file, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    running.append(
        {
            "proc": proc,
            "params": params,
            "seed": seed,
            "log": log_file,
            "warm_start": warm_start,
        }
    )
    ws = f", warm_start={warm_start}" if warm_start else ""
    print(
        f"Launched beta={params['beta']:.3f}, alpha={params['alpha']:.2f} "
        f"(d={params['d']}, P={params['P']}, N={params['N']}, sa0={params['sa0']:.4f}), "
        f"seed={seed}{ws} -> {log_file}"
    )
    time.sleep(0.5)


def try_launch_one(new_queue: deque, seed_queue: deque, running: list) -> bool:
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    if new_queue:
        params = new_queue.popleft()
        launch(params, SEEDS[0], running)
        if len(SEEDS) > 1:
            seed_queue.append((params, 1))
        return True

    if seed_queue:
        params, next_idx = seed_queue.popleft()
        launch(params, SEEDS[next_idx], running)
        if next_idx + 1 < len(SEEDS):
            seed_queue.append((params, next_idx + 1))
        return True

    return False


def try_launch_warm_start_chain(
    scale_points: list[dict],
    next_idx_by_seed: dict,
    finished_keys: set,
    running: list,
) -> bool:
    """Launch the next scale point for any seed whose previous point finished."""
    if len(running) >= MAX_PARALLEL_JOBS:
        return False

    launched = False
    for seed in SEEDS:
        if len(running) >= MAX_PARALLEL_JOBS:
            break
        if any(job["seed"] == seed for job in running):
            continue
        idx = next_idx_by_seed[seed]
        if idx >= len(scale_points):
            continue

        params = scale_points[idx]
        warm_start = None
        if idx > 0:
            prev_key = (idx - 1, seed)
            if prev_key not in finished_keys:
                continue
            warm_start = find_warm_start_path(scale_points[idx - 1], seed)
            if warm_start is None:
                print(
                    f"No warm-start checkpoint for "
                    f"beta={scale_points[idx-1]['beta']:.3f}, seed={seed}; "
                    f"cold-starting beta={params['beta']:.3f}"
                )

        launch(params, seed, running, warm_start=warm_start)
        next_idx_by_seed[seed] = idx + 1
        launched = True
    return launched


def main() -> None:
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    scale_points = build_scale_grid()

    print("=" * 60)
    print("FCN2 ERF — invariant alpha/beta MF scaling sweep")
    print("=" * 60)
    print("Scaling laws (factor 2 P^2 sa0 / (pi kappa^2 d N) held invariant):")
    print(f"  d     = {D0} * beta^{EPSILON}")
    print(f"  N     = {N0} * beta^{NU}")
    print(f"  sa0   = {SIGMA_A0} * beta^{RHO}   (sigma_a^2 readout weight decay)")
    print(
        f"  P     = {P_FIXED} fixed  "
        f"(alpha = (P/P0)^{{1/lambda}} = {ALPHA_FIXED:.4f}, P0={P0:g}, lambda={LAMBDA})"
    )
    print(f"  kappa = {KAPPA0} * (alpha/beta)^{OMEGA},  T = 2*kappa,  chi = {1.0}")
    print(f"  omega={OMEGA}, lambda={LAMBDA}, rho={RHO}")
    print(f"    (lambda=omega, rho=-omega, omega=eps+nu; F uses rho+2omega-eps-nu=0)")
    print(f"  beta_max={BETA_MAX} (d_max={D0 * BETA_MAX ** EPSILON:.0f})")
    print()
    print(f"  {NUM_SCALE_POINTS} beta points at fixed alpha={ALPHA_FIXED:.4f}:")
    for p in scale_points:
        print(
            f"    beta={p['beta']:6.3f}  alpha={p['alpha']:7.2f}  "
            f"d={p['d']:4d}  P={p['P']:5d}  N={p['N']:4d}  "
            f"sa0={p['sa0']:.4f}  kappa={p['kappa']:.4f}  T={p['temperature']:.4f}  "
            f"P/kappa={p['P']/p['kappa']:.1f}  factor={undistributed_factor(p):.4f}"
        )
    print()
    print(f"  s0={S0} (fixed), task_eps={TASK_EPS}, lr={LR}, ens={ENSEMBLE_SIZE}")
    print(
        f"  epochs={EPOCHS}{' effective with --schedule' if USE_SCHEDULE else ''}, "
        f"log_interval={LOG_INTERVAL}, device={DEVICE}, seeds={SEEDS}"
    )
    if USE_SCHEDULE:
        print(f"  schedule=ON (wall ≈ {_WALL_EPOCHS} steps; lr → lr0/3,/8,/9)")
    print(
        f"  A snapshots: interval={SNAPSHOT_A_INTERVAL}, burnin={SNAPSHOT_A_BURNIN} "
        f"(~{max(0, (_WALL_EPOCHS - SNAPSHOT_A_BURNIN) // SNAPSHOT_A_INTERVAL)} late "
        f"wall samples under A_snapshots/)"
    )
    if USE_MINIBATCH and BATCH_SIZE is not None:
        print(f"  minibatch: batch_size={BATCH_SIZE} with P/B gradient scaling (lr=LR/P)")
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

    if USE_WARM_START:
        next_idx_by_seed = {seed: 0 for seed in SEEDS}
        finished_keys: set = set()
        for seed in SEEDS:
            for idx, params in enumerate(scale_points):
                if find_warm_start_path(params, seed) is None:
                    break
                finished_keys.add((idx, seed))
                next_idx_by_seed[seed] = idx + 1
            if next_idx_by_seed[seed] > 0:
                print(
                    f"Seed {seed}: resuming warm-start chain at scale index "
                    f"{next_idx_by_seed[seed]} / {len(scale_points)}"
                )

        while try_launch_warm_start_chain(scale_points, next_idx_by_seed, finished_keys, running):
            pass

        while any(idx < len(scale_points) for idx in next_idx_by_seed.values()) or running:
            for job in running[:]:
                ret = job["proc"].poll()
                if ret is not None:
                    running.remove(job)
                    completed.append(job)
                    p = job["params"]
                    scale_idx = next(
                        i for i, sp in enumerate(scale_points)
                        if sp["beta"] == p["beta"] and sp["alpha"] == p["alpha"]
                    )
                    finished_keys.add((scale_idx, job["seed"]))
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    print(
                        f"Completed beta={p['beta']:.3f}, alpha={p['alpha']:.2f} "
                        f"(d={p['d']}, P={p['P']}, N={p['N']}), seed={job['seed']} [{status}]"
                    )

            while try_launch_warm_start_chain(scale_points, next_idx_by_seed, finished_keys, running):
                pass

            time.sleep(2)
    else:
        red_robin_order = build_red_robin_order(scale_points)
        new_queue = deque(red_robin_order)
        seed_queue = deque()

        while try_launch_one(new_queue, seed_queue, running):
            pass

        while new_queue or seed_queue or running:
            for job in running[:]:
                ret = job["proc"].poll()
                if ret is not None:
                    running.remove(job)
                    completed.append(job)
                    p = job["params"]
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    print(
                        f"Completed beta={p['beta']:.3f}, alpha={p['alpha']:.2f} "
                        f"(d={p['d']}, P={p['P']}, N={p['N']}), seed={job['seed']} [{status}]"
                    )

            while try_launch_one(new_queue, seed_queue, running):
                pass

            time.sleep(2)

    print("=" * 60)
    print(f"All jobs completed. {len(completed)} jobs run.")
    print(f"Sweep outputs saved under: {SWEEP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
