#!/usr/bin/env python3
"""Red-robin launcher for FCN2/FCN3-erf epsilon-sweep experiments."""

import argparse
import os
import subprocess
import time
from collections import deque

import numpy as np


def build_epsilon_values(eps_min: float, eps_max: float, num_points: int) -> np.ndarray:
    """Build logarithmically spaced epsilon values."""
    values = np.geomspace(eps_min, eps_max, num=num_points)
    return np.unique(np.round(values, 4))


def make_cmd(
    train_script: str,
    d: int,
    P: int,
    N: int,
    chi: float,
    kappa: float,
    lr: float,
    epochs: int,
    device: str,
    ens: int,
    seed: int,
    out_dir: str,
    eps: float,
):
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
    parser = argparse.ArgumentParser(
        description="Red-robin launcher for FCN2/FCN3-erf MF epsilon-sweep"
    )
    parser.add_argument("--d", type=int, default=10, help="Input dimension")
    parser.add_argument("--eps-min", type=float, default=0.03, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=3.0, help="Maximum epsilon")
    parser.add_argument("--num-eps", type=int, default=8, help="Number of epsilon values")
    parser.add_argument(
        "--eps-values",
        type=float,
        nargs="+",
        help="Explicit epsilon values; overrides the geometric sweep",
    )
    parser.add_argument(
        "--nets",
        type=str,
        nargs="+",
        choices=["fcn2", "fcn3"],
        default=["fcn2", "fcn3"],
        help="Network types to launch",
    )
    parser.add_argument("--P", type=int, default=100, help="Number of samples")
    parser.add_argument("--seeds", type=int, default=1, help="Number of dataset seeds per epsilon")
    parser.add_argument("--N", type=int, default=800, help="Hidden width")
    parser.add_argument("--chi", type=float, default=40, help="Chi parameter (default: N)")
    parser.add_argument("--kappa", type=float, default=0.1, help="Kappa")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base LR")
    parser.add_argument("--epochs", type=int, default=2_000_000, help="Training epochs")
    parser.add_argument("--ens", type=int, default=5, help="Ensemble size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--max-parallel-jobs", type=int, default=2, help="Max concurrent processes")
    parser.add_argument("--stagger-seconds", type=float, default=0.5, help="Delay between launches")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_epsilon_sweep",
        help="Output root; jobs are written to <out-dir>/fcn2 and <out-dir>/fcn3",
    )
    args = parser.parse_args()

    chi = float(args.N if args.chi is None else args.chi)
    eps_values = (
        np.array(args.eps_values, dtype=float)
        if args.eps_values is not None
        else build_epsilon_values(args.eps_min, args.eps_max, args.num_eps)
    )
    script_dir = os.path.dirname(__file__)
    train_script_fcn2 = os.path.join(script_dir, "d_sweep_fcn2_erf.py")
    train_script_fcn3 = os.path.join(script_dir, "d_sweep_fcn3_erf.py")

    # Create interleaved queue of (epsilon, network_type) pairs.
    pending = deque()
    for eps in eps_values:
        for net_type in args.nets:
            pending.append((float(eps), net_type))

    running = []
    completed = []

    print("Sweep settings:")
    print(f"  d={args.d}, P={args.P}, N={args.N}, chi={chi}, kappa={args.kappa}, ens={args.ens}")
    print(f"  nets={args.nets}")
    print(f"  Epsilon values={eps_values.tolist()}")
    print(f"  seeds={args.seeds}, max_parallel_jobs={args.max_parallel_jobs}")

    # Prime the queue from both ends.
    for end in ["left", "right"]:
        if not pending:
            break
        eps, net_type = pending.popleft() if end == "left" else pending.pop()
        train_script = train_script_fcn2 if net_type == "fcn2" else train_script_fcn3
        out_dir = os.path.join(args.out_dir, f"{net_type}")
        for seed in range(args.seeds):
            if len(running) >= args.max_parallel_jobs:
                break
            cmd = make_cmd(
                train_script=train_script,
                d=args.d,
                P=args.P,
                N=args.N,
                chi=chi,
                kappa=args.kappa,
                lr=args.lr,
                epochs=args.epochs,
                device=args.device,
                ens=args.ens,
                seed=seed,
                out_dir=out_dir,
                eps=eps,
            )
            proc = subprocess.Popen(cmd)
            running.append(
                {"proc": proc, "eps": eps, "net_type": net_type, "seed": seed, "end": end}
            )
            time.sleep(args.stagger_seconds)

    while pending or running:
        for job in running[:]:
            ret = job["proc"].poll()
            if ret is None:
                continue
            running.remove(job)
            completed.append({**job, "returncode": ret})

            if pending and len(running) < args.max_parallel_jobs:
                next_eps, next_net_type = (
                    pending.popleft() if job["end"] == "left" else pending.pop()
                )
                train_script = (
                    train_script_fcn2 if next_net_type == "fcn2" else train_script_fcn3
                )
                out_dir = os.path.join(args.out_dir, f"{next_net_type}")
                for seed in range(args.seeds):
                    if len(running) >= args.max_parallel_jobs:
                        break
                    cmd = make_cmd(
                        train_script=train_script,
                        d=args.d,
                        P=args.P,
                        N=args.N,
                        chi=chi,
                        kappa=args.kappa,
                        lr=args.lr,
                        epochs=args.epochs,
                        device=args.device,
                        ens=args.ens,
                        seed=seed,
                        out_dir=out_dir,
                        eps=next_eps,
                    )
                    proc = subprocess.Popen(cmd)
                    running.append(
                        {
                            "proc": proc,
                            "eps": next_eps,
                            "net_type": next_net_type,
                            "seed": seed,
                            "end": job["end"],
                        }
                    )
                    time.sleep(args.stagger_seconds)

        time.sleep(2.0)

    failed = [job for job in completed if job["returncode"] != 0]
    print(f"All jobs finished. total={len(completed)}, failed={len(failed)}")
    if failed:
        print("Failed jobs:")
        for job in failed:
            print(
                f"  eps={job['eps']} net_type={job['net_type']} seed={job['seed']} rc={job['returncode']}"
            )


if __name__ == "__main__":
    main()
