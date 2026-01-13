#!/usr/bin/env python3
"""
Parallel d-sweep runner for `ensembling_fcn3_linear` jobs.

This runner sweeps over input dimension `d`, sets P = 5 * d, n = 4 * d, and chi = n.
Launches jobs in parallel with a maximum of 5 concurrent subprocesses on cuda:1.
All job logs are written to script/logs/ directory.

Features:
- Parallel subprocess launching with a pool of up to 5 concurrent jobs.
- All jobs run on cuda:1 only.
- Proper SIGINT handling to kill child processes when the main script is interrupted.
- Writes per-job logs to `script/logs/d_sweep_linear_d_<d>.log`.
"""

import argparse
import os
import sys
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

_script_dir = Path(__file__).resolve().parent
_sys_python = sys.executable

# Path to the ensembling script
ENSEMBLING_SCRIPT = _script_dir / "ensembling_fcn3_linear.py"
LOG_DIR = _script_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Defaults / constants ===
n_factor = 4
epochs = 20_000_000
lrA_VALUES = [1e-6]  # Sweep over two learning rates
NUM_DATASETS = 1 # Train on 7 different datasets
device = "cuda:1"  # Fixed to cuda:1 only
MAX_WORKERS = 5

# Sweep defaults
DEFAULT_START = 20
DEFAULT_STOP = 80
DEFAULT_POINTS = 6

# Global to track child processes for cleanup
child_processes = []

def signal_handler(sig, frame):
    """Handle SIGINT by terminating all child processes."""
    print("\nKeyboardInterrupt received — terminating child processes...")
    for proc in child_processes:
        try:
            proc.terminate()
        except Exception:
            pass
    # Give them a moment to die, then forcefully kill
    import time
    time.sleep(0.5)
    for proc in child_processes:
        try:
            proc.kill()
        except Exception:
            pass
    sys.exit(130)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def build_command(d, lr, dataset_num, extra_args=None, headless=False):
    """Build command for linear ensembling job."""
    P = 5 * d
    n = n_factor * d
    chi = n
    cmd = [
        _sys_python,
        str(ENSEMBLING_SCRIPT),
        "--d", str(d),
        "--P", str(P),
        "--chi", str(chi),
        "--N", str(n),
        "--kappa", str(1.0),
        "--epochs", str(epochs),
        '--headless',
        "--lrA", str(lr),
        "--device", str(device),
        "--ens", str(5),
        "--experiment_dirname", "d_sweep_linear",
    ]
    if headless:
        cmd += ["--headless"]
    if extra_args:
        cmd += extra_args
    return cmd

def _run_cmd(cmd, logfile_path):
    """Run command and log output to file."""
    with open(logfile_path, "ab") as logf:
        logf.write(("\n--- Running: %s\n" % (" ".join(map(str, cmd)))).encode())
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        child_processes.append(proc)
        rc = proc.wait()
        try:
            child_processes.remove(proc)
        except ValueError:
            pass
        return rc

def run_job(d, lr, dataset_num, dry_run=False, extra_args=None, headless=False):
    """Run a single d-value, learning rate, and dataset job."""
    cmd = build_command(d, lr, dataset_num, extra_args=extra_args, headless=headless)
    log_path = LOG_DIR / f"d_sweep_linear_d_{d}_lr_{lr:.0e}.log"
    attempted = [cmd]
    
    if dry_run:
        return ((d, lr, dataset_num), None, log_path, " ".join(map(str, cmd)))

    env = os.environ.copy()
    rc = _run_cmd(cmd, log_path)
    return ((d, lr, dataset_num), rc, log_path, attempted)

def main():
    parser = argparse.ArgumentParser(description="Parallel d-sweep runner for FCN3 linear ensembling jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to ensembling script")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="start d (inclusive)")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="stop d (inclusive)")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="number of log-spaced d values")
    parser.add_argument("--extra-args", nargs="*", default=None)
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    points = args.points
    # Generate log-spaced d values
    ds = np.unique(np.round(np.logspace(np.log10(start), np.log10(stop), num=points))).astype(int)[::-1]
    d_values = ds.tolist()
    
    dry_run = args.dry_run
    extra_args = args.extra_args 
    headless = args.headless

    print(f"Linear d-sweep: P=5*d, n={n_factor}*d, chi=n; d_values={d_values}")
    print(f"Learning rates: {lrA_VALUES}")
    print(f"Datasets per d: {NUM_DATASETS}")
    print(f"ensembling script: {ENSEMBLING_SCRIPT}")
    print(f"device: {device}")
    print(f"max parallel jobs: {MAX_WORKERS}")
    print(f"logs -> {LOG_DIR}")
    
    run_results = []
    
    # Use ThreadPoolExecutor with MAX_WORKERS threads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        try:
            # Submit all jobs to the executor (all combinations of d, lr, and dataset_num)
            for d in d_values:
                for lr in lrA_VALUES:
                    for dataset_num in range(NUM_DATASETS):
                        print(f"Submitting d={d}, lr={lr}, dataset={dataset_num}...")
                        fut = executor.submit(run_job, d, lr, dataset_num, dry_run, extra_args, headless)
                        futures[fut] = (d, lr, dataset_num)

            # Collect results as they finish
            for fut in as_completed(list(futures.keys())):
                (d_submitted, lr_submitted, ds_submitted) = futures.pop(fut)
                try:
                    (d, lr, dataset_num), rc, log_path, attempted = fut.result()
                except Exception as e:
                    d, lr, dataset_num = d_submitted, lr_submitted, ds_submitted
                    rc = -1
                    log_path = LOG_DIR / f'd_sweep_linear_d_{d}_lr_{lr:.0e}.log'
                    attempted = []
                    print(f"Job for d={d}, lr={lr}, dataset={dataset_num} raised exception: {e}")

                if dry_run:
                    print(f"d={d}, lr={lr}, dataset={dataset_num}: {attempted}")
                else:
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"d={d}, lr={lr}, dataset={dataset_num}: {status}; log: {log_path}")
                    run_results.append(((d, lr, dataset_num), rc, log_path, attempted))
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received — cancelling remaining jobs...")
            executor.shutdown(wait=False)
            sys.exit(130)
    
    if not dry_run:
        print("\nSweep finished. Summary:")
        for item in sorted(run_results):
            (d, lr, dataset_num) = item[0]
            rc = item[1]
            log_path = item[2]
            attempted = item[3]
            print(f"- d={d}, lr={lr}, dataset={dataset_num}: rc={rc}, log={log_path}, attempts={len(attempted) if isinstance(attempted, list) else 1}")

if __name__ == "__main__":
    main()
