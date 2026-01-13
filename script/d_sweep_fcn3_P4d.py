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

_script_dir = Path(__file__).resolve().parent
_sys_python = sys.executable

# Path to the ensembling script (adjust if needed)
ENSEMBLING_SCRIPT = _script_dir / "ensembling_fcn3_erf_cubic.py"
LOG_DIR = _script_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Defaults / constants ===

n_factor = 4
epochs = 20_000_000
lrA = 1e-6
# devices
START_DEVICE = "cuda:0"
FALLBACK_DEVICE = "cuda:1"
# Sweep defaults
DEFAULT_START = 20
DEFAULT_STOP = 80
DEFAULT_STEP = 20


def build_command(d, device, extra_args=None, headless=False):
    P = 4 * d
    n = n_factor * d
    chi = n
    cmd = [
        _sys_python,
        str(ENSEMBLING_SCRIPT),
        "--d", str(d),
        "--P", str(P),
        "--chi", str(chi),
        "--N", str(n),
        "--epochs", str(epochs),
        "--lrA", str(lrA),
        "--device", str(device),
        "--eps", str(0.03),
        "--ens", str(5)
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
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to ensembling script to reduce plotting/logging (prints every ~10k epochs)")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="start d (inclusive)")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP, help="stop d (inclusive)")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="d step")
    parser.add_argument("--extra-args", nargs="*", default=None)
    parser.add_argument("--precompile", action='store_true', help="Create per-d JULIA_DEPOT_PATH and precompile sequentially before launching jobs")
    parser.add_argument("--keep-depots", action='store_true', help="Don't remove temporary Julia depots after successful runs (for debugging)")
    args = parser.parse_args()

    d_values = list(range(args.start, args.stop + 1, args.step))
    workers = args.workers
    dry_run = args.dry_run
    extra_args = args.extra_args
    headless = args.headless

    print(f"d-sweep: P=4*d, n={n_factor}*d, chi=n; d_values={d_values}")
    print(f"ensembling script: {ENSEMBLING_SCRIPT}")
    print(f"logs -> {LOG_DIR}")

   

    results = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_job, d, START_DEVICE, dry_run, extra_args, headless) for d in d_values]
            for fut in as_completed(futures):
                d, rc, log_path, attempted = fut.result()
                if dry_run:
                    print(f"d={d}: {attempted}")
                else:
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"d={d}: {status}; log: {log_path}")
                    results.append((d, rc, log_path, attempted))
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
        for d, rc, log_path, attempted in sorted(results):
            print(f"- d={d}: rc={rc}, log={log_path}, attempts={len(attempted)}")


if __name__ == "__main__":
    main()
