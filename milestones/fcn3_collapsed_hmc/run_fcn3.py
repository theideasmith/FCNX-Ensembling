#!/usr/bin/env python3
"""
Runner for the exact FCN3 posterior sampler (fcn3_sampler.py): collapsed HMC on (W, B) + exact a draws + exact
per-neuron jump sweeps. One chain per process.

Each run gets runs/<stamp>_<tag>_d<d>_P<P>_N<N1>x<N2>_<init>_r<seed>/ with config.json, trace.csv, obs.csv,
checkpoint.pt (for --resume) and summary.json (post-burn means with the central-branch theory from
../fcn3_matrix_theory for comparison). Lifecycle events go to runs/registry.jsonl.

Thesis point (notes Table 1), chain from the prior:
    python run_fcn3.py --d 150 --P 3000 --N1 1600 --N2 1600 --chi 10 --kappa 0.1 --eps 0.074 \
        --iters 400 --burn 100 --init prior --tag thesis
"""
import argparse
import csv
import json
import math
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fcn3_sampler import FCN3, X0Quadrature, data_fingerprint, make_data

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("physics")
    g.add_argument("--d", type=int, default=150)
    g.add_argument("--P", type=int, default=3000)
    g.add_argument("--N1", type=int, default=1600)
    g.add_argument("--N2", type=int, default=1600)
    g.add_argument("--chi", type=float, default=10.0)
    g.add_argument("--kappa", type=float, default=0.1)
    g.add_argument("--eps", type=float, default=0.074, help="target He1 + eps He3/sqrt(6)")
    g.add_argument("--sigA", type=float, default=1.0)
    g.add_argument("--sigB", type=float, default=1.0)
    g.add_argument("--sigw", type=float, default=1.0)
    g.add_argument("--nmax", type=int, default=3, help="Hermite tower used to parametrise middle-neuron jumps")
    g.add_argument("--seed", type=int, default=0, help="data seed")
    s = ap.add_argument_group("sampler")
    s.add_argument("--init", choices=["prior", "aligned"], default="prior")
    s.add_argument("--init-frac", type=float, default=0.3, help="aligned init: fraction of read-in neurons moved out")
    s.add_argument("--init-w0", type=float, default=2.4)
    s.add_argument("--rng-seed", type=int, default=1)
    s.add_argument("--iters", type=int, default=400)
    s.add_argument("--burn", type=int, default=100)
    s.add_argument("--leapfrog", type=int, default=24)
    s.add_argument("--step0", type=float, default=0.05)
    s.add_argument("--target-acc", type=float, default=0.8)
    s.add_argument("--sweep-every", type=int, default=2, help="jump sweep every k iterations (0: none)")
    s.add_argument("--sweep-layers", default="middle,readin")
    s.add_argument("--n-cand", type=int, default=16)
    s.add_argument("--redraw-frac", type=float, default=0.25, help="fraction of jumps that also redraw the perp direction")
    s.add_argument("--cg-iters", type=int, default=4)
    s.add_argument("--energy", choices=["direct", "series"], default="direct",
                   help="read-in candidate energies; series is faster only for many candidates (~40+)")
    s.add_argument("--chunk", type=int, default=None, help="rows of X per chunk (default: ~2^26 / max(N, d))")
    s.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    s.add_argument("--device", default="cuda")
    b = ap.add_argument_group("bookkeeping")
    b.add_argument("--obs-every", type=int, default=5)
    b.add_argument("--n-perp", type=int, default=4000, help="x_perp samples in the f-bar quadrature")
    b.add_argument("--checkpoint-every", type=int, default=20)
    b.add_argument("--tag", default="fcn3")
    b.add_argument("--group", default=None)
    b.add_argument("--runs-root", default=str(HERE / "runs"))
    b.add_argument("--resume", default=None)
    b.add_argument("--hot", type=float, default=80.0, help="pause between iterations at this GPU temperature (°C)")
    b.add_argument("--cool", type=float, default=70.0)
    b.add_argument("--smi-index", default="1", help="nvidia-smi index of the GPU in use (the 4090 is 1)")
    return ap.parse_args()


def gpu_temp(idx):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(idx), "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out.splitlines()[-1])
    except Exception:
        return None


def thermal_guard(args, log):
    t = gpu_temp(args.smi_index)
    if t is None or t < args.hot:
        return 0.0
    t0 = time.time()
    log(f"  GPU at {t:.0f} °C >= {args.hot}: pausing until {args.cool} °C")
    while True:
        time.sleep(2)
        t = gpu_temp(args.smi_index)
        if t is None or t <= args.cool:
            break
    return time.time() - t0


def registry(root, run_id, status, **kw):
    Path(root).mkdir(parents=True, exist_ok=True)
    with open(Path(root) / "registry.jsonl", "a") as fh:
        fh.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                             "status": status, **kw}) + "\n")


def theory_central(args):
    try:
        sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
        from fcn3_matrix_solver import FCN3Solver, Params
        p = Params(d=args.d, P=args.P, N1=args.N1, N2=args.N2, chi=args.chi, kappa=args.kappa, eps=args.eps,
                   sigA=args.sigA, sigB=args.sigB, sigw=args.sigw, nmax=max(args.nmax, 7))
        r = FCN3Solver(p, verbose=False).solve()
        H, J = np.array(r["H"]), np.array(r["J"])
        return dict(d_w0sq=r["d_w0sq"], L1=r["L1"], L3=r["L3"], qh_over_qbar=r["qh_over_qbar"],
                    q_over_q0=r["qperp_over_prior"], H11=H[0, 0], H33=H[1, 1], H13=H[0, 1], J11=J[0, 0],
                    J33=J[1, 1], J13=J[0, 1], fbar5=r["fbar"][2], fbar7=r["fbar"][3], converged=r["converged"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    args = parse()
    if args.resume:
        run_dir = Path(args.resume)
        cfg = json.loads((run_dir / "config.json").read_text())
        iters = args.iters
        for k, v in cfg["args"].items():
            setattr(args, k, v)
        args.iters, args.resume = max(iters, cfg["args"]["iters"]), str(run_dir)
    else:
        run_id = (f"{datetime.now():%Y%m%d-%H%M%S}_{args.tag}_d{args.d}_P{args.P}_N{args.N1}x{args.N2}"
                  f"_{args.init}_r{args.rng_seed}")
        run_dir = Path(args.runs_root) / run_id
        run_dir.mkdir(parents=True)
        cfg = dict(run_id=run_id, created=datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
                   command=" ".join([sys.executable] + sys.argv), args=vars(args))
    run_id = cfg["run_id"]
    logf = open(run_dir / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    dev = torch.device(args.device)
    dt = torch.float64 if args.dtype == "float64" else torch.float32
    X, Y, _, _ = make_data(args.d, args.P, args.eps, 1.0, args.seed, dev)
    X, Y = X.to(dt), Y.to(dt)
    fp = data_fingerprint(X)
    if cfg.setdefault("data_fingerprint", fp) != fp:
        raise RuntimeError(f"training data differ from the run's: {fp} vs {cfg['data_fingerprint']}")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    ks = tuple(range(1, args.nmax + 1, 2))
    m = FCN3(X, Y, args.N1, args.N2, args.chi, args.kappa, args.sigA, args.sigB, args.sigw, ks=ks, chunk=args.chunk,
             n_cand=args.n_cand, cg_iters=args.cg_iters, redraw_frac=args.redraw_frac, seed=args.rng_seed,
             energy=args.energy)
    xq = X0Quadrature(args.d, args.n_perp, device=dev, dtype=dt)
    layers = tuple(args.sweep_layers.split(",")) if args.sweep_every else ()

    ck_path = run_dir / "checkpoint.pt"
    if args.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ZW, ZB, step, start = ck["ZW"].to(dev, dt), ck["ZB"].to(dev, dt), ck["step"], ck["iter"] + 1
        m.gen.set_state(ck["gen"]); m.np_rng.bit_generator.state = ck["np_rng"]
    else:
        g = torch.Generator(device=dev).manual_seed(args.rng_seed)
        ZW = torch.randn(args.N1, args.d, device=dev, dtype=dt, generator=g)
        ZB = torch.randn(args.N2, args.N1, device=dev, dtype=dt, generator=g)
        if args.init == "aligned":
            n_al = int(args.init_frac * args.N1)
            sgn = (torch.randint(0, 2, (n_al,), device=dev, generator=g) * 2 - 1).to(dt)
            ZW[:n_al, 0] = sgn * args.init_w0 / m.sw
            ZW[:n_al, 1:] *= math.sqrt(0.5)
        step, start = args.step0, 0
    log(f"[{run_id}] d={args.d} P={args.P} N1={args.N1} N2={args.N2} chi={args.chi} kappa={args.kappa} "
        f"eps={args.eps} S={ks} {args.dtype} chunk={m.chunk} init={args.init} start={start}")
    registry(args.runs_root, run_id, "resumed" if args.resume else "running", group=args.group, tag=args.tag,
             d=args.d, P=args.P, N1=args.N1, N2=args.N2, chi=args.chi, kappa=args.kappa, init=args.init,
             rng_seed=args.rng_seed)

    mode = "a" if start > 0 else "w"
    trf = open(run_dir / "trace.csv", mode, newline=""); tr = csv.writer(trf)
    obf = open(run_dir / "obs.csv", mode, newline=""); ob = None
    if start == 0:
        tr.writerow(["iter", "U", "acc", "step", "L", "mid_changed", "mid_crossed", "in_changed", "in_crossed",
                     "t_hmc", "t_sweep", "wall"])
    u, gW, gB = m.U(ZW, ZB)
    t_start, paused = time.time(), 0.0
    try:
        for it in range(start, args.iters):
            paused += thermal_guard(args, log)
            ZW, ZB = m.flip_signs(ZW, ZB)
            u, gW, gB = m.U(ZW, ZB)
            t0 = time.time()
            ZW, ZB, u, gW, gB, acc, L = m.hmc_step(ZW, ZB, u, gW, gB, step, args.leapfrog)
            torch.cuda.synchronize(dev) if dev.type == "cuda" else None
            t_hmc = time.time() - t0
            if it < args.burn:
                step = math.exp(math.log(step) + (acc - args.target_acc) * 2.0 / math.sqrt(it + 10))
            W, B = m.sw * ZW, m.sb * ZB
            a = m.draw_a(W, B)
            st, t_sw = {}, 0.0
            if layers and it % args.sweep_every == 0:
                t0 = time.time()
                W, B, a, st = m.sweep(W, B, a, layers)
                torch.cuda.synchronize(dev) if dev.type == "cuda" else None
                t_sw = time.time() - t0
                ZW, ZB = W / m.sw, B / m.sb
                u, gW, gB = m.U(ZW, ZB)
            tr.writerow([it, f"{u:.8e}", f"{acc:.4f}", f"{step:.5e}", L] +
                        [f"{st[k]:.4f}" if k in st else "" for k in ("mid_changed", "mid_crossed", "in_changed",
                                                                      "in_crossed")]
                        + [f"{t_hmc:.3f}", f"{t_sw:.3f}", f"{time.time() - t_start - paused:.1f}"])
            if it % args.obs_every == 0:
                o = m.observables(W, B, a, xq)
                if ob is None:
                    ob = csv.writer(obf)
                    if start == 0:
                        ob.writerow(["iter", "burn"] + list(o))
                    keys = list(o)
                ob.writerow([it, int(it < args.burn)] + [f"{o[k]:.8g}" for k in keys])
                log(f"  it {it:5d} U={u:.6e} acc={acc:.2f} step={step:.3e} L={L} hmc {t_hmc:.2f}s sweep {t_sw:.2f}s "
                    + " ".join(f"{k}={v:.3f}" for k, v in st.items())
                    + f" | d<w0^2>={o['d_w0sq']:.3f} in_out={o['frac_in_out']:.3f} mid_out={o['frac_mid_out']:.3f}"
                      f" H11={o['H11']:.4e} H33={o['H33']:.3e} L3={o['fbar3'] / args.eps:.3f} q/q0={o['q_over_q0']:.3f}")
            if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
                trf.flush(); obf.flush()
                torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step, iter=it, gen=m.gen.get_state(),
                                np_rng=m.np_rng.bit_generator.state), ck_path)
    except BaseException as e:
        trf.close(); obf.close()
        registry(args.runs_root, run_id, "failed", error=f"{type(e).__name__}: {e}")
        raise
    trf.close(); obf.close()
    # summary: post-burn means and stderr (batch means over 10 batches)
    rows = [r for r in csv.DictReader(open(run_dir / "obs.csv")) if r["burn"] == "0"]
    summ = {}
    for k in rows[0]:
        if k in ("iter", "burn"):
            continue
        x = np.array([float(r[k]) for r in rows])
        nb = min(10, len(x))
        bm = np.array([b.mean() for b in np.array_split(x, nb)])
        summ[k] = [float(x.mean()), float(bm.std(ddof=1) / math.sqrt(nb)) if nb > 1 else float("nan")]
    summ["L1"] = summ["fbar1"]
    summ["L3"] = [summ["fbar3"][0] / args.eps, summ["fbar3"][1] / abs(args.eps)]
    trace = [r for r in csv.DictReader(open(run_dir / "trace.csv")) if int(r["iter"]) >= args.burn]
    summary = dict(run_id=run_id, wall_s=time.time() - t_start - paused, n_obs=len(rows), posterior=summ,
                   acc=float(np.mean([float(r["acc"]) for r in trace])), theory_central=theory_central(args))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    registry(args.runs_root, run_id, "done", wall_s=round(summary["wall_s"], 1),
             d_w0sq=summ["d_w0sq"][0], L3=summ["L3"][0], H11=summ["H11"][0], frac_in_out=summ["frac_in_out"][0])
    log(json.dumps({k: summ[k] for k in ("d_w0sq", "L1", "L3", "H11", "H33", "H13", "qh_over_qbar", "q_over_q0",
                                         "frac_in_out", "frac_mid_out")}, indent=1))
    log("theory (central branch): " + json.dumps(summary["theory_central"]))


if __name__ == "__main__":
    main()
