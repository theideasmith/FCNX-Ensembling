#!/usr/bin/env python3
"""
Parallel d-sweep runner for `ensembling_fcn3_erf_cubic` jobs.

This runner sweeps over input dimension `d` (default 10..200 step 10), sets
P = 4 * d, n = n_factor * d (default n_factor=10), and chi = n. For each d
it launches the ensembling job `script/ensembling_fcn3_erf_cubic.py` with the
corresponding hyperparameters. Behavior mirrors `fcn3_erf_cubic_sweep.py`.

Usage:
  python script/d_sweep_fcn3_P4d.py [--workers N] [--dry-run] [--start D] [--stop D]

By default it will sweep d in 10,20,...,200.

Features:
- Parallel subprocess launching using `concurrent.futures.ThreadPoolExecutor`.
- Writes per-job logs to `script/logs/d_sweep_d_<d>.log`.
- Retries on device failure (tries START_DEVICE then FALLBACK_DEVICE).
"""

import argparse
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import shutil
import uuid
import numpy as np

_script_dir = Path(__file__).resolve().parent
_sys_python = sys.executable

# Path to the ensembling script (adjust if needed)
ENSEMBLING_SCRIPT = _script_dir / "ensembling_fcn3_erf_cubic.py"
LOG_DIR = _script_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Optional: run using a list of pre-existing experiment folders under EXPERIMENT_DIR.
# If all entries in EXPERIMENT_FILES exist under EXPERIMENT_DIR, the runner will
# launch one `ensembling_fcn3_erf_cubic.py --modeldesc <name>` process per file
# in parallel instead of performing the numeric d-sweep.
EXPERIMENT_DIR = Path('/home/akiva/exp/fcn3erf')
EXPERIMENT_FILES = [
    # 'erf_cubic_eps_0.03_P_225_D_45_N_180_epochs_20000000_lrA_4.44e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_475_D_95_N_380_epochs_20000000_lrA_2.11e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_100_D_20_N_80_epochs_20000000_lrA_1.00e-08_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_350_D_70_N_280_epochs_20000000_lrA_2.86e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_975_D_195_N_780_epochs_20000000_lrA_1.03e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_850_D_170_N_680_epochs_20000000_lrA_1.18e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_725_D_145_N_580_epochs_20000000_lrA_1.38e-09_time_20251201_220935',
    # 'erf_cubic_eps_0.03_P_600_D_120_N_480_epochs_20000000_lrA_1.67e-09_time_20251201_220935',
]

# === Defaults / constants ===

n_factor = 4
epochs = 20_000_000
lrA = 1.5e-5
# devices
START_DEVICE = "cuda:0"
FALLBACK_DEVICE = "cuda:1"
# Sweep defaults
DEFAULT_START = 20
DEFAULT_STOP = 65
DEFAULT_POINTS = 10

def build_command(d, device, extra_args=None, headless=False):
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
        '--kappa', str(1.0),
        "--epochs", str(epochs),
        "--lrA", str(lrA),
        "--device", str(device),
        "--eps", str(0.03),
        "--ens", str(5),
        '--experiment_dirname', store_location,
    ]
    if headless:
        cmd += ["--headless"]
    if extra_args:
        cmd += extra_args
    return cmd


def _run_cmd(cmd, logfile_path):
    with open(logfile_path, "ab") as logf:
        logf.write(("\n--- Running: %s\n" % (" ".join(map(str, cmd)))).encode())
        # cmd may include environment overrides in a tuple: (cmd_list, env)
        if isinstance(cmd, tuple) and len(cmd) == 2:
            cmd_list, env = cmd
            proc = subprocess.Popen(cmd_list, stdout=logf, stderr=subprocess.STDOUT, env=env)
        else:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        return proc.wait()


def build_command_for_modeldesc(modeldesc, device, extra_args=None, headless=False):
    """Build a command-line invocation of the ensembling script using
    `--modeldesc <modeldesc>`. When possible, reuse the same flags that the
    numeric d-sweep would have produced by parsing `D` out of `modeldesc` and
    calling `build_command(d, ...)`. Then append `--modeldesc <modeldesc>` so
    the child process receives the same runtime flags as a numeric run plus
    the `modeldesc` hint.

    If `D` cannot be parsed, fall back to a minimal invocation that includes
    `--modeldesc` and the selected device.
    """
    import re
    d = None
    m = re.search(r"_D_(\d+)_", modeldesc)
    if not m:
        m = re.search(r"_D_(\d+)$", modeldesc)
    if m:
        try:
            d = int(m.group(1))
        except Exception:
            d = None

    if d is not None:
        # Reuse the same flags the numeric run would set for this d
        cmd = build_command(d, device, extra_args=None, headless=headless)
    else:
        # Minimal fallback: include device and headless if requested
        cmd = [_sys_python, str(ENSEMBLING_SCRIPT), "--device", str(device)]
        if headless:
            cmd += ["--headless"]

    # Append the modeldesc flag so the ensembling script resumes from that folder
    # Pass the full path under EXPERIMENT_DIR because the experiments live there
    cmd += ["--modeldesc", str(EXPERIMENT_DIR / modeldesc)]
    if extra_args:
        cmd += extra_args
    return cmd


def run_modeldesc_job(modeldesc, device, dry_run=False, extra_args=None, headless=False):
    """Run a single ensembling job for an existing `modeldesc` folder.
    Returns (modeldesc, rc, log_path, attempted_cmds)"""
    cmd = build_command_for_modeldesc(modeldesc, device, extra_args=extra_args, headless=headless)
    safe_name = modeldesc.replace('/', '_')
    log_path = LOG_DIR / f"d_sweep_modeldesc_{safe_name}.log"
    if dry_run:
        return (modeldesc, None, log_path, " ".join(map(str, cmd)))

    env = os.environ.copy()
    attempted = [cmd]
    rc = _run_cmd((cmd, env), log_path)
    final_rc = rc
    # no device fallback here: modeldesc jobs are treated as user-specified
    return (modeldesc, final_rc, log_path, attempted)


def run_job(d, device, dry_run=False, extra_args=None, headless=False):
    cmd = build_command(d, device, extra_args=extra_args, headless=headless)
    log_path = LOG_DIR / f"d_sweep_d_{d}.log"
    attempted = [cmd]
    if dry_run:
        return (d, None, log_path, " ".join(map(str, cmd)))

    env = os.environ.copy()

    # Run the job with the adjusted environment. Wrap cmd together with env
    rc = _run_cmd((cmd, env), log_path)
    final_rc = rc

    # On device failure attempt fallback device using same per-job depot
    if rc != 0 and str(device).startswith('cuda') and FALLBACK_DEVICE and str(device) != FALLBACK_DEVICE:
        fallback_cmd = build_command(d, FALLBACK_DEVICE, extra_args=extra_args)
        attempted.append(fallback_cmd)
        final_rc = _run_cmd((fallback_cmd, env), log_path)

    return (d, final_rc, log_path, attempted)


def main():
    parser = argparse.ArgumentParser(description="Parallel d-sweep runner for FCN3 ensembling jobs")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to ensembling script to reduce plotting/logging (prints every ~10k epochs)")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="start d (inclusive)")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="stop d (inclusive)")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="number of log-spaced d values (default 10)")
    parser.add_argument("--step", type=int, default=None, help="(deprecated) d step — if provided it will override logspacing")
    parser.add_argument("--store-location",type=str, default='d_sweep_fcn3_P5d_N4d', help="Experiment directory name to use for storing results (default: d_sweep_fcn3_P5d_N4d)")
    parser.add_argument("--extra-args", nargs="*", default=None)
    args = parser.parse_args()

    global store_location
    store_location = args.store_location
    # log-spaced integer d values
    ds = np.unique(np.round(np.logspace(np.log10(args.start), np.log10(args.stop), num=args.points))).astype(int)[::-1]
    d_values = ds.tolist()
    
    workers = args.workers
    dry_run = args.dry_run
    extra_args = args.extra_args 
    headless = args.headless

    print(f"d-sweep: P=4*d, n={n_factor}*d, chi=n; d_values={d_values}")
    print(f"ensembling script: {ENSEMBLING_SCRIPT}")
    print(f"logs -> {LOG_DIR}")
    # If a curated list of EXPERIMENT_FILES is provided and all of those
    # folders exist under EXPERIMENT_DIR, run the ensembling script in
    # parallel for each `--modeldesc` instead of the numeric d-sweep.
    run_results = []
    try:
        if EXPERIMENT_FILES and all((EXPERIMENT_DIR / f).exists() for f in EXPERIMENT_FILES):
            print("All EXPERIMENT_FILES found under", EXPERIMENT_DIR)
            # Launch one process per modeldesc but schedule them round-robin across
            # available GPUs so that only one network runs per GPU at a time.
            devices = [START_DEVICE]
            if FALLBACK_DEVICE and FALLBACK_DEVICE != START_DEVICE:
                devices.append(FALLBACK_DEVICE)

            executors = {dev: ThreadPoolExecutor(max_workers=1) for dev in devices}
            futures = {}
            try:
                # Submit modeldesc jobs round-robin across devices
                for i, modeldesc in enumerate(EXPERIMENT_FILES):
                    dev = devices[i % len(devices)]
                    fut = executors[dev].submit(run_modeldesc_job, modeldesc, dev, dry_run, extra_args, headless)
                    futures[fut] = (modeldesc, dev)

                # Collect results as they finish
                for fut in as_completed(list(futures.keys())):
                    modeldesc_submitted, dev_used = futures.pop(fut)
                    try:
                        modeldesc, rc, log_path, attempted = fut.result()
                    except Exception as e:
                        modeldesc = modeldesc_submitted
                        rc = -1
                        safe_name = modeldesc.replace('/', '_')
                        log_path = LOG_DIR / f"d_sweep_modeldesc_{safe_name}.log"
                        attempted = []
                        print(f"Job for modeldesc={modeldesc} on {dev_used} raised exception: {e}")

                    if dry_run:
                        print(f"modeldesc={modeldesc}: {attempted}")
                    else:
                        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                        print(f"modeldesc={modeldesc}: {status}; log: {log_path} (device={dev_used})")
                        run_results.append((modeldesc, rc, log_path, attempted))
            finally:
                for ex in executors.values():
                    ex.shutdown(wait=True)
        else:
            if EXPERIMENT_FILES:
                missing = [f for f in EXPERIMENT_FILES if not (EXPERIMENT_DIR / f).exists()]
                if missing:
                    print("Some EXPERIMENT_FILES were not found under EXPERIMENT_DIR; falling back to numeric d-sweep. Missing:", missing)
        # Numeric d-sweep: existing behavior
            # Numeric d-sweep: run one job per GPU concurrently by creating a
            # single-worker ThreadPoolExecutor for each GPU and submitting jobs
            # in round-robin fashion. This ensures only one network trains on a
            # GPU at a time and utilizes both GPUs (START_DEVICE and FALLBACK_DEVICE).
            devices = [START_DEVICE]
            if FALLBACK_DEVICE and FALLBACK_DEVICE != START_DEVICE:
                devices.append(FALLBACK_DEVICE)

            executors = {dev: ThreadPoolExecutor(max_workers=1) for dev in devices}
            futures = {}
            try:
                # Submit jobs round-robin across devices
                for i, d in enumerate(d_values):
                    dev = devices[i % len(devices)]
                    print(f"Submitting d={d} to device {dev}...")
                    fut = executors[dev].submit(run_job, d, dev, dry_run, extra_args, headless)
                    futures[fut] = (d, dev)

                # Wait for jobs to finish and collect results
                for fut in as_completed(list(futures.keys())):
                    d_submitted, dev_used = futures.pop(fut)
                    try:
                        d, rc, log_path, attempted = fut.result()
                    except Exception as e:
                        # Executor-level exception
                        d = d_submitted
                        rc = -1
                        log_path = LOG_DIR / f'd_sweep_d_{d}.log'
                        attempted = []
                        print(f"Job for d={d} on {dev_used} raised exception: {e}")

                    if dry_run:
                        print(f"d={d}: {attempted}")
                    else:
                        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                        print(f"d={d}: {status}; log: {log_path} (device={dev_used})")
                        run_results.append((d, rc, log_path, attempted))
            finally:
                # Shutdown per-device executors
                for ex in executors.values():
                    ex.shutdown(wait=True)
    except KeyboardInterrupt:
        # Runner interrupted by user — try to cancel outstanding work and
        # remove any per-d log files created by this sweep to avoid partial logs.
        print("\nKeyboardInterrupt received — cancelling remaining jobs and cleaning up logs...")
        try:
            # remove d_sweep log files matching pattern
            for p in LOG_DIR.glob('d_sweep_d_*.log'):
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Could not remove log file {p}: {e}")
        except Exception as _e:
            print(f"Error cleaning up logs: {_e}")
        # exit with 130 (standard for SIGINT)
        sys.exit(130)
     
    if not dry_run:
        print("\nSweep finished. Summary:")
        for item in sorted(run_results):
            ident = item[0]
            rc = item[1]
            log_path = item[2]
            attempted = item[3]
            print(f"- {ident}: rc={rc}, log={log_path}, attempts={len(attempted) if isinstance(attempted, list) else 1}")


if __name__ == "__main__":
    main()
