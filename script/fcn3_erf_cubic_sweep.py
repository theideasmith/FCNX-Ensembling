#!/usr/bin/env python3
"""
Parallel sweep runner for `ensembling_fcn3_erf_cubic` jobs.

This script launches multiple independent processes (in parallel) that run
the `ensembling_fcn3_erf_cubic` job (assumed available as
`script/ensembling_fcn3_erf_cubic.py`) with different values of `P`.

Configuration (hard-coded per your request):
- N = 250
- d = 40
- chi = 50
- epochs = 20000000
- lrA = 1e-6
- P values: [d//2, d, 5*d, 10*d, 1000]

Features:
- Uses `concurrent.futures` to run jobs in parallel
- Launches subprocesses using the same Python interpreter (`sys.executable`)
- Writes per-job stdout/stderr to `logs/fcn3_erf_cubic_sweep_P_<P>.log`
- Supports `--workers` to limit concurrent jobs and `--dry-run` to just print commands

Behavior:
- Starts each job with `--device cuda:1`.
- If the job exits with non-zero return code, the runner will retry once with `--device cuda:0`.
- Jobs are launched in descending order of `P` (largest first).

NOTE: This file only writes the commands and spawns processes when executed.
Do not run it automatically from here; the file is created for you to run later.
"""

import argparse
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import centralized experiment collections
try:
    from script.experiment_collections import erf_cubic_P_SWEEP
except Exception:
    # Fallback: if import fails, define an empty list so script still runs
    erf_cubic_P_SWEEP = []

# === User-specified constants ===
N = 250
d = 40
chi = 50
epochs = 20_000_000
lrA = 1e-6
# Default P sweep (kept for backward compatibility) but we will prefer
# experiments listed in `erf_cubic_P_SWEEP` from `script/experiment_collections.py`.
DEFAULT_P_VALUES = [d // 2, d, 5 * d, 10 * d, 1000]

# Base directory to prepend to experiment folder names from the collection
BASE_EXPERIMENT_DIR = Path("/home/akiva/exp/fcn3erf")

# If the centralized collection provides experiments, use those; otherwise
# fall back to launching a job per P value (as before).
if erf_cubic_P_SWEEP:
    # Build absolute modeldesc paths from the collection
    MODELDESCS = [str((BASE_EXPERIMENT_DIR / m).resolve()) for m in erf_cubic_P_SWEEP]
else:
    # Fallback behavior: create pseudo modeldesc strings for each P
    P_values = sorted(DEFAULT_P_VALUES, reverse=True)
    MODELDESCS = [f"P_{P}" for P in P_values]

# Device preferences for sweep: start with this, fallback to fallback_device on failure
START_DEVICE = 'cuda:1'
FALLBACK_DEVICE = 'cuda:0'

# Path to the ensembling script relative to repo root (adjust if different)
ENSEMBLING_SCRIPT = Path(__file__).resolve().parent / "ensembling_fcn3_erf_cubic.py"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_command(modeldesc, device, extra_args=None):
    """Build the command (list) to call the ensembling script for given modeldesc and device.

    We pass the absolute `--modeldesc` path so the target script resumes the
    existing experiment directory. We still pass the other hyperparameters to
    ensure the run uses the desired settings.
    """
    cmd = [sys.executable, str(ENSEMBLING_SCRIPT)]

    cmd += [
        "--modeldesc", str(modeldesc),
        "--N", str(N),
        "--d", str(d),
        "--chi", str(chi),
        "--epochs", str(epochs),
        "--lrA", str(lrA),
        "--ens", str(7),
        "--device", str(device),
    ]

    if extra_args:
        cmd += extra_args

    return cmd


def run_job(modeldesc, device, dry_run=False, extra_args=None):
    """Run a single job for the given `modeldesc` using `device`.

    Behavior:
    - Attempt to run with `device`.
    - If the process exits with non-zero returncode and the initial device
      was a CUDA device, retry once with `FALLBACK_DEVICE`.

    Returns (modeldesc, final_returncode, log_path, attempted_cmds).
    When dry_run is True, returns the constructed command(s) without launching.
    """
    initial_cmd = build_command(modeldesc, device, extra_args=extra_args)
    # Use a safe filename for the log by taking the basename of the modeldesc
    safe_name = Path(str(modeldesc)).name
    log_path = LOG_DIR / f"fcn3_erf_cubic_sweep_{safe_name}.log"
    attempted = []

    if dry_run:
        return (modeldesc, None, log_path, " && ".join([" ".join(map(str, initial_cmd))]))

    # Helper to run a command and return rc
    def _run_cmd(cmd, logfile_path):
        with open(logfile_path, "ab") as logf:
            logf.write(f"\n--- Running: {' '.join(map(str, cmd))} ---\n".encode())
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            return proc.wait()

    # First attempt
    attempted.append(initial_cmd)
    rc = _run_cmd(initial_cmd, log_path)
    final_rc = rc

    # If failed and initial device was a cuda device different from fallback, retry
    if rc != 0 and str(device).startswith('cuda') and FALLBACK_DEVICE and str(device) != FALLBACK_DEVICE:
        fallback_cmd = build_command(modeldesc, FALLBACK_DEVICE, extra_args=extra_args)
        attempted.append(fallback_cmd)
        final_rc = _run_cmd(fallback_cmd, log_path)

    return (modeldesc, final_rc, log_path, attempted)


def main():
    parser = argparse.ArgumentParser(description="Parallel sweep runner for ensembling jobs")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Maximum concurrent worker processes")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't run them")
    parser.add_argument("--extra-args", nargs="*", default=None,
                        help="Extra arguments to append to each ensembling command")
    args = parser.parse_args()

    workers = args.workers
    dry_run = args.dry_run
    extra_args = args.extra_args

    print(f"Running sweep with workers={workers}, dry_run={dry_run}")
    print(f"Ensembling script: {ENSEMBLING_SCRIPT}")
    print(f"Logs will be written to: {LOG_DIR}")

    # Use ThreadPoolExecutor just to start subprocesses concurrently; each subprocess
    # is independent and does the heavy work. Multiprocessing executor could also be used,
    # but subprocess launching (IO-bound) is fine with threads here.
    futures = []
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # submit jobs for each modeldesc collected from experiment_collections
        for modeldesc in MODELDESCS:
            futures.append(ex.submit(run_job, modeldesc, START_DEVICE, dry_run, extra_args))

        for fut in as_completed(futures):
            modeldesc, rc, log_path, attempted = fut.result()
            safe_name = Path(str(modeldesc)).name
            if dry_run:
                print(f"model={safe_name}: {attempted}")
            else:
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"model={safe_name}: {status}; log: {log_path}")
                results.append((modeldesc, rc, log_path, attempted))

    if not dry_run:
        print("\nSweep finished. Summary:")
        for modeldesc, rc, log_path, attempted in results:
            print(f"- model={Path(str(modeldesc)).name}: rc={rc}, log={log_path}, attempts={len(attempted)}")


if __name__ == "__main__":
    main()
