#!/usr/bin/env python3
"""
Collapsed-HMC posterior sampling for FCN2 erf nets, with optional per-neuron Gibbs jumps.

HMC moves all of W0 along the collapsed potential (readout integrated out);
--gibbs-every k adds an exact per-neuron Gibbs sweep (neuron_gibbs.py) before every
k-th HMC iteration so neurons can jump between unaligned and aligned wells.

Each run gets runs/<run_id>/ (see bookkeeping.py):
    config.json, trace.csv (U, acceptance, step, L, Gibbs changed/crossed fractions),
    obs.csv, samples/iter_*.pt {W0, A, U}, checkpoint.pt, summary.json
and lifecycle events in runs/registry.jsonl (runs.py list / compare).

Example (β=100 steepwell point):
    python train_collapsed_hmc.py --d 50 --P 11998 --N 3162 --T 0.004743 --sa0 0.0009 --eps 0.5 \
        --dtype float32 --leapfrog 50 --step0 0.03 --target-acc 0.8 --iters 200 --burn 50 \
        --gibbs-every 5 --init prior --rng-seed 1 --group b100_gibbs --tag b100
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from bookkeeping import OBS_KEYS, RunRecorder, stamp
from collapsed_hmc_fcn2 import CollapsedFCN2, data_fingerprint, hmc, make_data, observables
from common import (add_bookkeeping_args, add_gibbs_args, add_init_args, add_physics_args, gibbs_sweeps,
                    initial_W, langevin_comparison, print_table)
from fast_moves import BlockGibbs, GraphedNeuronGibbs
from neuron_gibbs import NeuronGibbs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_physics_args(ap)
    add_init_args(ap)
    s = ap.add_argument_group("HMC")
    s.add_argument("--chains", type=int, default=1, help="chains batched in this process (all share --init)")
    s.add_argument("--iters", type=int, default=200)
    s.add_argument("--burn", type=int, default=50)
    s.add_argument("--leapfrog", type=int, default=50, help="max leapfrog steps; jittered in [L/2, L]")
    s.add_argument("--step0", type=float, default=0.03)
    s.add_argument("--target-acc", type=float, default=0.8)
    s.add_argument("--no-flips", action="store_true")
    add_gibbs_args(ap)
    add_bookkeeping_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.langevin_dir:
        args.langevin_dir = str(Path(args.langevin_dir).resolve())

    if args.resume:
        rec = RunRecorder.load(args.resume)
        iters, resume = args.iters, args.resume
        for k, v in rec.cfg["args"].items():
            setattr(args, k, v)
        args.iters = max(iters, rec.cfg["args"]["iters"])  # allow extending a finished run
        args.resume = resume
    else:
        run_id = f"{stamp()}_{args.tag}_d{args.d}_P{args.P}_N{args.N}_{args.init}_r{args.rng_seed}"
        rec = RunRecorder.create(args.runs_root, run_id, args, kind="collapsed_hmc")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    X, Y, Xinf, Yinf = make_data(args.d, args.P, args.eps, args.he1, args.seed, device)
    X, Y, Xinf, Yinf = (t.to(dtype) for t in (X, Y, Xinf, Yinf))
    fp = data_fingerprint(X)
    if rec.cfg.setdefault("data_fingerprint", fp) != fp:
        raise RuntimeError(f"training data differ from this run's config: {fp} vs {rec.cfg['data_fingerprint']}")
    rec.save_config()
    model = CollapsedFCN2(X, Y, args.N, args.T, args.s0, args.sa0, args.chi)
    gibbs = None
    if args.gibbs_every and not args.gibbs_block:
        if getattr(args, "gibbs_graph", False):
            gibbs = GraphedNeuronGibbs(X, Y, args.N, args.T, args.s0, args.sa0, args.chi, omega=args.gibbs_omega,
                                       B=args.gibbs_cands, inv_dtype=dtype)
        else:
            gibbs = NeuronGibbs(X, Y, args.N, args.T, args.s0, args.sa0, args.chi, omega=args.gibbs_omega,
                                s_perp=args.gibbs_sperp)
    block = (BlockGibbs(X, Y, args.N, args.T, args.s0, args.sa0, args.chi, block=args.gibbs_block)
             if args.gibbs_every and args.gibbs_block else None)

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    start_iter, step_init = 0, None
    ckpt_path = rec.run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        Z0 = ck["Z"].to(device, dtype)
        step_init = ck["step"].to(device, dtype)
        start_iter = ck["iter"] + 1
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"], device)
        np.random.set_state(ck["np_rng"])
    else:
        W0, snap = initial_W(args, args.chains, args.N, args.d, model.sw, device, dtype)
        Z0 = W0 / model.sw
        if snap:
            rec.cfg["init_snapshot"] = snap
            rec.save_config()

    print(f"[{rec.run_id}] d={args.d} P={args.P} N={args.N} T={args.T} σ_W²={model.sw ** 2:.4g} σ_A²={model.sa2:.4g}"
          f" form={'NxN Woodbury' if model.use_woodbury else 'PxP'} chains={args.chains} {args.dtype}"
          f" gibbs_every={args.gibbs_every} start_iter={start_iter}", flush=True)
    rec.event("resumed" if args.resume else "running", kind="collapsed_hmc", group=args.group, tag=args.tag,
              d=args.d, P=args.P, N=args.N, T=args.T, sa0=args.sa0, eps=args.eps, init=args.init,
              rng_seed=args.rng_seed, iters=args.iters, burn=args.burn, gibbs_every=args.gibbs_every,
              langevin_dir=args.langevin_dir)
    rec.open_tables(["iter", "chain", "U", "acc", "step", "L", "wall_s", "gibbs_changed", "gibbs_crossed"],
                    ["iter", "chain", "burn"] + OBS_KEYS, append=start_iter > 0)

    log_every = max(args.iters // 20, 1)
    t_start = time.time()

    def pre_step(it, Z):
        if not args.gibbs_every or it % args.gibbs_every != 0:
            return None
        if block is not None:
            block.adapt = it < args.burn and not args.gibbs_block_fixed
            W = torch.stack([block.sweep(Z[c] * model.sw) for c in range(Z.shape[0])])
            return W / model.sw, {"changed": block.stats["changed"], "crossed": block.stats["crossed"]}
        W, st = gibbs_sweeps(gibbs, Z * model.sw, args)
        return W / model.sw, st

    def callback(it, Z, u, info):
        pre = info["pre"] or {}
        for c in range(Z.shape[0]):
            rec.trace_w.writerow([it, c, f"{u[c].item():.8e}", f"{info['acc'][c].item():.4f}",
                                  f"{info['step'][c].item():.5e}", info["L"], f"{info['wall']:.2f}",
                                  f"{pre['changed']:.4f}" if pre else "", f"{pre['crossed']:.4f}" if pre else ""])
        save = it >= args.burn and args.save_every and (it - args.burn) % args.save_every == 0
        if it % args.obs_every == 0 or save:
            A = model.sample_A(Z)
        if it % args.obs_every == 0:
            rec.write_obs(it, args.burn, observables(model, Z, A, Xinf, Yinf))
        if save:
            rec.save_sample(it, W0=Z * model.sw, A=A, U=u)
        if (it + 1) % args.checkpoint_every == 0 or it == args.iters - 1:
            rec.flush()
            torch.save({"iter": it, "Z": Z.cpu(), "step": info["step"].cpu(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                        "np_rng": np.random.get_state()}, ckpt_path)
        if it % log_every == 0 or it == args.iters - 1 or pre:
            extra = f"  gibbs changed={pre['changed']:.3f} crossed={pre['crossed']:.3f}" if pre else ""
            print(f"  it {it:6d}  U={u.mean().item():.6e}  acc={info['acc'].mean().item():.3f}"
                  f"  step={info['step'].mean().item():.3e}  L={info['L']}  t={time.time() - t_start:.1f}s{extra}",
                  flush=True)

    try:
        Z, step = hmc(model, Z0, args.iters, args.burn, args.leapfrog, args.step0, args.target_acc,
                      not args.no_flips, callback=callback, start_iter=start_iter, step_init=step_init,
                      pre_step=pre_step)
        rec.close()
        wall = time.time() - t_start
        pooled, per_chain, n_obs = rec.obs_summary()
        summary = {
            "run_id": rec.run_id,
            "kind": "collapsed_hmc",
            "wall_s_this_session": wall,
            "iters": int(args.iters),
            "n_obs_post_burn": n_obs,
            "U_post_burn_mean": float(rec.trace_column("U", args.burn).mean()),
            "acc_post_burn_mean": float(rec.trace_column("acc", args.burn).mean()),
            "gibbs_crossed_post_burn_mean": (float(rec.trace_column("gibbs_crossed", args.burn).mean())
                                             if args.gibbs_every else None),
            "step_final": step.cpu().tolist(),
            "posterior": pooled,
            "per_chain": per_chain,
            "langevin": langevin_comparison(model, args.langevin_dir, Xinf, Yinf, device, dtype),
        }
        rec.write_summary(summary)
    except BaseException as e:
        rec.close()
        rec.event("failed", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    rec.event("done", wall_s=round(wall, 1), U_post_burn=summary["U_post_burn_mean"],
              acc=summary["acc_post_burn_mean"], L3=pooled["L3"][0], w0_sq=pooled["w0_sq"][0],
              frac_aligned=pooled["frac_aligned"][0], wperp_sq=pooled["wperp_sq"][0],
              test_mse=pooled["test_mse"][0])
    print(f"\n[{rec.run_id}] done in {wall:.1f}s")
    print_table(summary)


if __name__ == "__main__":
    main()
