#!/usr/bin/env python3
"""Red-robin launcher for FCN2-erf P-sweep experiments."""

import argparse
import os
import subprocess
import time
from collections import deque

import numpy as np


def build_p_values(d: int, p_max: int, num_points: int) -> np.ndarray:
    values = np.geomspace(max(1, d), p_max, num=num_points)
    p_values = np.unique(values.astype(int))
    if p_values[0] != d:
        p_values = np.unique(np.append(p_values, d))
    if p_values[-1] != p_max:
        p_values = np.unique(np.append(p_values, p_max))
    return p_values


def make_cmd(train_script: str, d: int, P: int, N: int, chi: float, kappa: float, lr: float, epochs: int, device: str, ens: int, seed: int, out_dir: str, eps: float):
    return [
        "python3",
        train_script,
        "--d",
        str(d),
        "--P",
        str(P),
        "--N",
        str(N),
        "--chi",
        str(chi),
        "--kappa",
        str(kappa),
        "--lr",
        str(lr),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--ens",
        str(ens),
        "--to",
        out_dir,
        "--eps",
        str(eps),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Red-robin launcher for FCN2-erf MF P-sweep")
    parser.add_argument("--d", type=int, default=10, help="Input dimension")
    parser.add_argument("--p-max", type=int, default=4000, help="Maximum P in sweep")
    parser.add_argument("--num-p", type=int, default=4, help="Number of P values")
    parser.add_argument("--seeds", type=int, default=1, help="Number of dataset seeds per P")
    parser.add_argument("--N", type=int, default=800, help="Hidden width")
    parser.add_argument("--chi", type=float, default=80, help="Chi parameter (default: N)")
    parser.add_argument("--kappa", type=float, default=0.1, help="Kappa")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base LR")
    parser.add_argument("--epochs", type=int, default=12_000_000, help="Training epochs")
    parser.add_argument("--ens", type=int, default=10, help="Ensemble size")
    parser.add_argument("--eps", type=float, default=0.4, help="Target cubic coefficient")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--max-parallel-jobs", type=int, default=2, help="Max concurrent processes")
    parser.add_argument("--stagger-seconds", type=float, default=0.5, help="Delay between launches")
    parser.add_argument("--out-dir", type=str, default="results_fcn2_erf", help="Output subdirectory")
    args = parser.parse_args()

    chi = float(args.N if args.chi is None else args.chi)
    p_values = build_p_values(args.d, args.p_max, args.num_p)
    train_script = os.path.join(os.path.dirname(__file__), "d_sweep_fcn2_erf.py")

    pending = deque(p_values)
    running = []
    completed = []

    print("Sweep settings:")
    print(f"  d={args.d}, N={args.N}, chi={chi}, kappa={args.kappa}, ens={args.ens}")
    print(f"  P values={p_values.tolist()}")
    print(f"  seeds={args.seeds}, max_parallel_jobs={args.max_parallel_jobs}")

    # Prime the queue from both ends.
    for end in ["left", "right"]:
        if not pending:
            break
        P = pending.popleft() if end == "left" else pending.pop()
        for seed in range(args.seeds):
            if len(running) >= args.max_parallel_jobs:
                break
            cmd = make_cmd(
                train_script=train_script,
                d=args.d,
                P=int(P),
                N=args.N,
                chi=chi,
                kappa=args.kappa,
                lr=args.lr,
                epochs=args.epochs,
                device=args.device,
                ens=args.ens,
                seed=seed,
                out_dir=args.out_dir,
                eps=args.eps,
            )
            proc = subprocess.Popen(cmd)
            running.append({"proc": proc, "P": int(P), "seed": seed, "end": end})
            time.sleep(args.stagger_seconds)

    while pending or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is None:
                continue
            running.remove(job)
            completed.append({**job, "returncode": ret})

            if pending and len(running) < args.max_parallel_jobs:
                next_P = pending.popleft() if job["end"] == "left" else pending.pop()
                for seed in range(args.seeds):
                    if len(running) >= args.max_parallel_jobs:
                        break
                    cmd = make_cmd(
                        train_script=train_script,
                        d=args.d,
                        P=int(next_P),
                        N=args.N,
                        chi=chi,
                        kappa=args.kappa,
                        lr=args.lr,
                        epochs=args.epochs,
                        device=args.device,
                        ens=args.ens,
                        seed=seed,
                        out_dir=args.out_dir,
                        eps=args.eps,
                    )
                    proc = subprocess.Popen(cmd)
                    running.append({"proc": proc, "P": int(next_P), "seed": seed, "end": job["end"]})
                    time.sleep(args.stagger_seconds)

        time.sleep(2.0)

    failed = [job for job in completed if job["returncode"] != 0]
    print(f"All jobs finished. total={len(completed)}, failed={len(failed)}")
    if failed:
        print("Failed jobs:")
        for job in failed:
            print(f"  P={job['P']} seed={job['seed']} rc={job['returncode']}")


if __name__ == "__main__":
    main()
