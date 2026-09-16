"""N-width ablation at a fixed invariant-sweep operating point.

Purpose
-------
Hold (d, P, kappa/T, sa0, eps, chi, s0, seed, lr schedule) fixed and vary only
hidden width N, to test whether larger N improves A–W alignment / learnability.

Baseline source (the run being ablated)
---------------------------------------
This ablation is anchored to one completed AlphaBetaInvariant scale point from:

  red_robin_alpha_beta_invariant_P0160_Pmax3674_betamax9/

Baseline model directory (absolute path resolved from this file):

  .../models/invariant_beta5.196_alpha110.05_d114_P1678_N377_sa00.6623_kappa1.3806_seed0

How to identify / recover that run
----------------------------------
From the directory name:
  beta   = 5.196
  alpha  = 110.05
  d      = 114
  P      = 1678
  N      = 377          <-- baseline width; ablation varies this only
  sa0    = 0.6623
  kappa  = 1.3806
  seed   = 0

From that run's config.json (copied into BASELINE below):
  temperature = 2*kappa, chi=1, eps=0, s0=1, ens=10,
  base lr = 5e-3  (train script uses lr/P; config stores the effective lr).

Parent sweep exponents at the time of that run (nu=1/4 invariant family):
  epsilon=1/2, nu=1/4, rho=-1/4, omega=1/4, lambda=1/2

Usage
-----
  python n_ablation_invariant_P1678.py --dry-run
  python n_ablation_invariant_P1678.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_fcn2_erf_sigma_a.py"

# ---------------------------------------------------------------------------
# Explicit baseline identity
# ---------------------------------------------------------------------------
BASELINE_SWEEP_DIR = SCRIPT_DIR / "red_robin_alpha_beta_invariant_P0160_Pmax3674_betamax9"
BASELINE_RUN_NAME = (
    "invariant_beta5.196_alpha110.05_d114_P1678_N377_sa00.6623_kappa1.3806_seed0"
)
BASELINE_RUN_DIR = BASELINE_SWEEP_DIR / "models" / BASELINE_RUN_NAME
BASELINE_CONFIG_PATH = BASELINE_RUN_DIR / "config.json"

# Parsed from the baseline run name / config (single source of truth for the
# ablation hyperparameters other than N).
BASELINE = {
    "beta": 5.196,
    "alpha": 110.05,
    "d": 114,
    "P": 1678,
    "N": 377,
    "sa0": 0.662337782140526,
    "kappa": 1.3805924367330555,
    "temperature": 2.761184873466111,  # = 2 * kappa
    "chi": 1.0,
    "eps": 0.0,
    "s0": 1.0,
    "seed": 0,
    "ens": 10,
    "base_lr": 5e-3,  # same as invariant launcher; train script divides by P
    "epochs": 10_000_000,
    "log_interval": 100_000,
}

# Widths to train. Baseline N is listed for documentation; by default we skip
# retraining it and reuse BASELINE_RUN_DIR.
N_VALUES = [377, 1500, 3000]
SKIP_BASELINE_N = True

DEVICE = "cuda:0"
MAX_PARALLEL_JOBS = 2

ABLATION_DIR = SCRIPT_DIR / "n_ablation_from_invariant_P1678"
MODELS_DIR = ABLATION_DIR / "models"
TENSORBOARD_DIR = ABLATION_DIR / "tensorboard"
LOG_DIR = ABLATION_DIR / "launcher_logs"
MANIFEST_PATH = ABLATION_DIR / "baseline_manifest.json"


def run_dir_name(n: int) -> str:
    b = BASELINE
    return (
        f"ablateN{n}"
        f"_from_beta{b['beta']:.3f}_alpha{b['alpha']:.2f}"
        f"_d{b['d']}_P{b['P']}_N{n}"
        f"_sa0{b['sa0']:.4f}_kappa{b['kappa']:.4f}"
        f"_seed{b['seed']}"
    )


def make_cmd(n: int) -> list[str]:
    b = BASELINE
    run_name = run_dir_name(n)
    return [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--d",
        str(b["d"]),
        "--P",
        str(b["P"]),
        "--N",
        str(n),
        "--chi",
        str(b["chi"]),
        "--temperature",
        str(b["temperature"]),
        "--lr",
        str(b["base_lr"]),
        "--device",
        DEVICE,
        "--epochs",
        str(b["epochs"]),
        "--log-interval",
        str(b["log_interval"]),
        "--dataset-seed",
        str(b["seed"]),
        "--ens",
        str(b["ens"]),
        "--eps",
        str(b["eps"]),
        "--s0",
        str(b["s0"]),
        "--sa0",
        str(b["sa0"]),
        "--output-dir",
        str(MODELS_DIR / run_name),
        "--tensorboard-dir",
        str(TENSORBOARD_DIR / run_name),
    ]


def load_baseline_config() -> dict | None:
    if not BASELINE_CONFIG_PATH.exists():
        return None
    with open(BASELINE_CONFIG_PATH) as f:
        return json.load(f)


def write_manifest(cfg: dict | None) -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": (
            "N-width ablation at fixed (d,P,kappa,sa0,T,eps,chi,s0,seed) "
            "taken from one AlphaBetaInvariant scale point."
        ),
        "baseline_sweep_dir": str(BASELINE_SWEEP_DIR),
        "baseline_run_name": BASELINE_RUN_NAME,
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "baseline_config_path": str(BASELINE_CONFIG_PATH),
        "baseline_exists": BASELINE_RUN_DIR.is_dir(),
        "baseline_params_from_script": BASELINE,
        "baseline_config_json": cfg,
        "held_fixed": [
            "d",
            "P",
            "sa0",
            "kappa",
            "temperature",
            "chi",
            "eps",
            "s0",
            "seed",
            "ens",
            "base_lr",
            "epochs",
        ],
        "varied": ["N"],
        "n_values": N_VALUES,
        "skip_baseline_n": SKIP_BASELINE_N,
        "note": (
            "F = 2 P^2 sa0 / (pi kappa^2 d N) is NOT held invariant here: "
            "N changes while sa0/kappa/d/P stay fixed. That is intentional."
        ),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def planned_widths() -> list[int]:
    widths = list(N_VALUES)
    if SKIP_BASELINE_N:
        widths = [n for n in widths if n != BASELINE["N"]]
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N ablation at the invariant P=1678 operating point"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write manifest; do not launch training",
    )
    parser.add_argument(
        "--include-baseline-n",
        action="store_true",
        help=f"Also train N={BASELINE['N']} instead of only referencing the sourced run",
    )
    args = parser.parse_args()

    global SKIP_BASELINE_N
    if args.include_baseline_n:
        SKIP_BASELINE_N = False

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    cfg = load_baseline_config()
    write_manifest(cfg)

    print("=" * 60)
    print("N ablation — sourced from AlphaBetaInvariant P=1678 point")
    print("=" * 60)
    print(f"Baseline run dir: {BASELINE_RUN_DIR}")
    print(f"Baseline exists:  {BASELINE_RUN_DIR.is_dir()}")
    if cfg is None:
        print("WARNING: baseline config.json not found; using hardcoded BASELINE dict.")
    else:
        print(
            "Baseline config.json: "
            f"d={cfg.get('d')}, P={cfg.get('P')}, N={cfg.get('N')}, "
            f"sa0={cfg.get('sa0')}, T={cfg.get('temperature')}, eps={cfg.get('eps')}"
        )
        # Soft checks against the sourced config.
        for key, script_key in (("d", "d"), ("P", "P"), ("N", "N"), ("chi", "chi"), ("eps", "eps")):
            if key in cfg and abs(float(cfg[key]) - float(BASELINE[script_key])) > 1e-8:
                print(
                    f"WARNING: config {key}={cfg[key]} differs from BASELINE[{script_key}]="
                    f"{BASELINE[script_key]}"
                )
    print(f"Manifest:         {MANIFEST_PATH}")
    print(
        "Held fixed: "
        f"d={BASELINE['d']}, P={BASELINE['P']}, kappa={BASELINE['kappa']}, "
        f"sa0={BASELINE['sa0']:.4f}, T={BASELINE['temperature']:.4f}, "
        f"eps={BASELINE['eps']}, seed={BASELINE['seed']}"
    )
    print(f"N values:         {N_VALUES}")
    print(f"Will train N:     {planned_widths()}")
    print(f"Output models:    {MODELS_DIR}")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for n in planned_widths():
        cmd = make_cmd(n)
        run_name = run_dir_name(n)
        log_path = LOG_DIR / f"{run_name}.log"
        print(f"N={n}:")
        print("  " + " ".join(cmd))
        print(f"  log -> {log_path}")
        jobs.append((n, cmd, log_path, run_name))

    if args.dry_run:
        print("\nDry run only; not launching.")
        return

    running = []
    completed = []
    pending = list(jobs)

    def try_launch() -> bool:
        if not pending or len(running) >= MAX_PARALLEL_JOBS:
            return False
        n, cmd, log_path, run_name = pending.pop(0)
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append({"proc": proc, "n": n, "log": log_path, "run_name": run_name})
        print(f"Launched N={n} -> {log_path}")
        return True

    while try_launch():
        pass

    while pending or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is None:
                continue
            running.remove(job)
            completed.append(job)
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"Completed N={job['n']} [{status}] log={job['log']}")
            try_launch()
        if pending or running:
            time.sleep(5)

    print(f"\nDone. Trained {len(completed)} runs under {MODELS_DIR}")
    print(f"Compare against baseline: {BASELINE_RUN_DIR}")


if __name__ == "__main__":
    main()
