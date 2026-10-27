#!/usr/bin/env python3
"""
Convergence-driven collapsed-HMC runs along the κ0 = 0.015 invariant ray (see launch_ray_k0p015.sh).

For each ray point b (processed in order):
  1. three chains: prior start, small start, and a start drawn from the theory's p(w0)
     (w_perp from the prior), so the equilibrium is approached from both sides;
  2. Gibbs sweeps in mode 'both' (w0 sweep, then whole-neuron sweep that resamples w_perp too);
  3. after each stage, split-chain R̂ over the second half of every chain on each --keys observable;
     if any R̂ ≥ --rhat, all chains are resumed for --extend more iterations, up to --max-iters;
  4. the point is marked converged or not in runs/logs/<prefix>_driver.log, then the next b starts.

Run it inside tmux so it survives the terminal:
    tmux new -d -s ray_both "python ray_driver.py --bs 1 1.41421356 2 2.82842712 4"
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from readin_action_theory import Kernel, reduced_params, solve_loop
from runs import RUNS_ROOT

HERE = Path(__file__).resolve().parent
LOGS = RUNS_ROOT / "logs"
INITS = RUNS_ROOT / "inits"
S0, EPS = 0.5, -0.3 * 6 ** 0.5


def ray_point(b):
    al = b ** 1.25
    return dict(d=round(150 * b ** 0.5), N=round(1500 * b ** 0.25), P=round(3000 * al ** 0.75),
                T=2 * 0.015 * (al / b) ** 0.75, sa0=(8 / 1500) * b ** -0.75)


def theory_init(p, path, seed=7):
    """W0 (N, d): w0 from the theory's p(w0) with random signs, w_perp from the prior."""
    T, beta, lam, Chat = reduced_params(p["N"], p["d"], S0, p["sa0"], p["P"], p["T"] / 2)
    K = Kernel(T, beta)
    D, _ = solve_loop(Chat, lam, EPS, K)
    prob = K.stats(D)[1]
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(prob) / prob.sum()
    w0 = np.interp(rng.random(p["N"]), cdf, K.w) * rng.choice([-1, 1], p["N"])
    W = np.concatenate([w0[:, None], rng.normal(0, np.sqrt(S0 / p["d"]), (p["N"], p["d"] - 1))], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W0": torch.tensor(W)}, path)


def split_rhat(series):
    """Split-chain R̂ over the second half of each chain (list of 1-d arrays)."""
    n = min(len(v) for v in series) // 2
    n -= n % 2
    if n < 4:
        return float("inf")
    halves = [h for v in series for h in (v[-n:][: n // 2], v[-n:][n // 2:])]
    L = n // 2
    means = np.array([h.mean() for h in halves])
    W = np.mean([h.var(ddof=1) for h in halves])
    B = L * means.var(ddof=1)
    return float(np.sqrt(((L - 1) / L * W + B / L) / W)) if W > 0 else float("inf")


def obs_series(run_dir, key):
    import csv
    with open(run_dir / "obs.csv") as f:
        return np.array([float(r[key]) for r in csv.DictReader(f)])


def find_run(group, init):
    for cp in sorted(RUNS_ROOT.glob("*/config.json")):
        a = json.loads(cp.read_text())["args"]
        if a.get("group") == group and a.get("init") == init:
            return cp.parent
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bs", nargs="+", type=float, required=True)
    ap.add_argument("--prefix", default="ray_k0p015_both")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:0"])
    ap.add_argument("--iters0", type=int, default=600)
    ap.add_argument("--burn", type=int, default=150)
    ap.add_argument("--extend", type=int, default=600)
    ap.add_argument("--max-iters", type=int, default=2400)
    ap.add_argument("--rhat", type=float, default=1.1)
    ap.add_argument("--keys", nargs="+", default=["frac_aligned", "w0_sq"])
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    dlog = open(LOGS / f"{args.prefix}_driver.log", "a", buffering=1)

    def say(msg):
        line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        dlog.write(line + "\n")

    for b in args.bs:
        p = ray_point(b)
        btag = f"{b:g}".replace(".", "p")
        group = f"{args.prefix}_b{btag}"
        init_file = INITS / f"{group}_theory_init.pt"
        if not init_file.exists():
            theory_init(p, init_file)
        specs = [("prior", 1, []), ("small", 1, []), ("file", 1, ["--init-file", str(init_file)])]
        common = ["--d", str(p["d"]), "--P", str(p["P"]), "--N", str(p["N"]), "--T", repr(p["T"]),
                  "--s0", str(S0), "--sa0", repr(p["sa0"]), "--eps", repr(EPS), "--seed", "0",
                  "--dtype", args.dtype, "--leapfrog", "50", "--step0", "0.03", "--target-acc", "0.8",
                  "--burn", str(args.burn), "--obs-every", "5", "--save-every", "4", "--chains", "1",
                  "--gibbs-every", "3", "--gibbs-mode", "both", "--group", group, "--tag", f"rayboth_b{btag}"]
        say(f"b={b:g} {group}: d={p['d']} N={p['N']} P={p['P']} κ={p['T'] / 2:.5f} sa0={p['sa0']:.4e}")
        iters = args.iters0
        stage = 0
        while True:
            procs = []
            for (init, seed, extra), dev in zip(specs, args.devices):
                log = open(LOGS / f"{group}_{init}_r{seed}.log", "a")
                run = find_run(group, init)
                if stage == 0 and run is None:
                    cmd = [sys.executable, "train_collapsed_hmc.py", *common, "--iters", str(iters),
                           "--device", dev, "--init", init, "--rng-seed", str(seed), *extra]
                else:
                    cmd = [sys.executable, "train_collapsed_hmc.py", "--resume", str(run), "--iters", str(iters),
                           # required physics args are re-read from the run's config on resume
                           "--d", "0", "--P", "0", "--N", "0", "--T", "0", "--sa0", "0", "--device", dev]
                procs.append((init, subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)))
                time.sleep(3)
            codes = {init: pr.wait() for init, pr in procs}
            if any(codes.values()):
                say(f"  stage {stage}: FAILED exit codes {codes}; stopping")
                return
            runs = {init: find_run(group, init) for init, _, _ in specs}
            rh = {k: split_rhat([obs_series(r, k) for r in runs.values()]) for k in args.keys}
            means = {k: [obs_series(r, k)[-len(obs_series(r, k)) // 2:].mean() for r in runs.values()] for k in args.keys}
            say(f"  stage {stage} (iters={iters}): " + "  ".join(
                f"{k}: R̂={rh[k]:.3f} [{', '.join(f'{m:.5f}' for m in means[k])}]" for k in args.keys))
            if all(v < args.rhat for v in rh.values()):
                say(f"  b={b:g} CONVERGED at {iters} iterations")
                break
            if iters >= args.max_iters:
                say(f"  b={b:g} NOT converged by {iters} iterations; moving on")
                break
            iters = min(iters + args.extend, args.max_iters)
            stage += 1
    say("driver finished")


if __name__ == "__main__":
    main()
