#!/usr/bin/env python3
"""
Langevin training of FCN2 erf nets with per-layer step sizes and optional exact jumps.

Same objective and stationary distribution as fcn2_erf_hidden_kernel/train_fcn2_erf_sigma_a.py:
    E = Σ_μ (f_μ - y_μ)² + (wd_W/2)|W0|² + (wd_A/2)|A|²,   wd = T/σ²,   p ∝ exp(-E/T)
    θ ← θ - lr_θ ∇_θ E + sqrt(2 lr_θ T) ξ           (separately for θ = W0 and θ = A)
A different lr per layer only changes the speed, not the target. The readout is ~1e3-1e4×
stiffer than W0, so a shared lr (capped by A) makes W0 crawl; here
    lr_A = c_A / λ_max(H_AA),   lr_W = c_W / λ_max(H_WW)
with the curvatures re-estimated by power iteration every --curv-every steps.

Optional exact moves (both leave the posterior invariant):
    --gibbs-every k     per-neuron Gibbs sweep on W0 with A integrated out (neuron_gibbs.py),
                        followed by an exact draw of A | W0
    --exact-A-every k   exact draw of A | W0 alone

Like the original trainer, plain Langevin has O(lr) discretization bias; --final-frac f and
--final-scale s shrink both step sizes by s over the last fraction f of steps.

Example (β=1 steepwell point):
    python train_langevin_split.py --d 5 --P 160 --N 1000 --T 0.002 --sa0 0.03 --eps 0.5 \
        --steps 200000 --burn 50000 --obs-every 1000 --tag b1_split
"""

import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from bookkeeping import OBS_KEYS, RunRecorder, stamp
from collapsed_hmc_fcn2 import CollapsedFCN2, data_fingerprint, make_data, observables
from common import (add_bookkeeping_args, add_gibbs_args, add_init_args, add_physics_args, gibbs_sweeps,
                    initial_W, langevin_comparison, print_table)
from neuron_gibbs import NeuronGibbs

SQRT_PI = math.sqrt(math.pi)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_physics_args(ap)
    add_init_args(ap)
    s = ap.add_argument_group("Langevin")
    s.add_argument("--chains", type=int, default=1, help="independent networks trained in parallel")
    s.add_argument("--steps", type=int, default=100000)
    s.add_argument("--burn", type=int, default=20000)
    s.add_argument("--c-W", type=float, default=0.2, help="lr_W = c_W / λ_max(H_WW)")
    s.add_argument("--c-A", type=float, default=0.2, help="lr_A = c_A / λ_max(H_AA)")
    s.add_argument("--curv-every", type=int, default=5000)
    s.add_argument("--final-frac", type=float, default=0.0, help="fraction of steps run at reduced lr")
    s.add_argument("--final-scale", type=float, default=0.25, help="lr multiplier over --final-frac")
    s.add_argument("--exact-A-every", type=int, default=0)
    s.add_argument("--trace-every", type=int, default=100)
    add_gibbs_args(ap)
    add_bookkeeping_args(ap)
    ap.set_defaults(obs_every=1000, save_every=5000, checkpoint_every=10000)
    return ap.parse_args()


class SplitLangevin:
    def __init__(self, X, Y, N, T, s0, sa0, chi=1.0):
        self.X, self.Y = X, Y
        self.P, self.d = X.shape
        self.T = T / chi
        self.wd_W = self.T / (s0 / self.d)
        self.wd_A = self.T / (sa0 / (N * chi))

    def energy_terms(self, W, A):
        H = torch.einsum("cnd,pd->cpn", W, self.X)
        Phi = torch.erf(H)
        r = torch.einsum("cpn,cn->cp", Phi, A) - self.Y
        return H, Phi, r

    def energy(self, W, A):
        _, _, r = self.energy_terms(W, A)
        return (r * r).sum(-1) + 0.5 * self.wd_W * (W * W).sum((1, 2)) + 0.5 * self.wd_A * (A * A).sum(-1)

    def grads(self, W, A):
        H, Phi, r = self.energy_terms(W, A)
        gA = 2.0 * torch.einsum("cpn,cp->cn", Phi, r) + self.wd_A * A
        dphi = (2.0 / SQRT_PI) * torch.exp(-H * H)
        G = 2.0 * r[:, :, None] * A[:, None, :] * dphi  # (C, P, N)
        gW = torch.einsum("cpn,pd->cnd", G, self.X) + self.wd_W * W
        E = (r * r).sum(-1) + 0.5 * self.wd_W * (W * W).sum((1, 2)) + 0.5 * self.wd_A * (A * A).sum(-1)
        return E, gW, gA

    def curvatures(self, W, A, iters=30):
        """λ_max of the W0-W0 and A-A Hessian blocks (max over chains), by HVP power iteration."""
        out = []
        for block in ("W", "A"):
            Wv = W.detach().clone().requires_grad_(True)
            Av = A.detach().clone().requires_grad_(True)
            v = torch.randn_like(Wv if block == "W" else Av)
            lam = None
            for _ in range(iters):
                v = v / v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
                E = self.energy(Wv, Av).sum()
                (g,) = torch.autograd.grad(E, Wv if block == "W" else Av, create_graph=True)
                (hv,) = torch.autograd.grad((g * v).sum(), Wv if block == "W" else Av)
                lam = (hv * v).flatten(1).sum(1)
                v = hv.detach()
            out.append(lam.max().item())
        return out


def main():
    args = parse_args()
    if args.langevin_dir:
        args.langevin_dir = str(Path(args.langevin_dir).resolve())

    if args.resume:
        rec = RunRecorder.load(args.resume)
        steps, resume = args.steps, args.resume
        for k, v in rec.cfg["args"].items():
            setattr(args, k, v)
        args.steps = max(steps, rec.cfg["args"]["steps"])
        args.resume = resume
    else:
        run_id = f"{stamp()}_{args.tag}_d{args.d}_P{args.P}_N{args.N}_{args.init}_r{args.rng_seed}"
        rec = RunRecorder.create(args.runs_root, run_id, args, kind="langevin_split")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    X, Y, Xinf, Yinf = make_data(args.d, args.P, args.eps, args.he1, args.seed, device)
    X, Y, Xinf, Yinf = (t.to(dtype) for t in (X, Y, Xinf, Yinf))
    fp = data_fingerprint(X)
    if rec.cfg.setdefault("data_fingerprint", fp) != fp:
        raise RuntimeError(f"training data differ from this run's config: {fp} vs {rec.cfg['data_fingerprint']}")
    rec.save_config()
    lang = SplitLangevin(X, Y, args.N, args.T, args.s0, args.sa0, args.chi)
    model = CollapsedFCN2(X, Y, args.N, args.T, args.s0, args.sa0, args.chi)  # exact A | W0 and observables
    gibbs = (NeuronGibbs(X, Y, args.N, args.T, args.s0, args.sa0, args.chi, omega=args.gibbs_omega,
                         s_perp=args.gibbs_sperp) if args.gibbs_every else None)

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    start = 0
    ckpt_path = rec.run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W, A = ck["W"].to(device, dtype), ck["A"].to(device, dtype)
        start = ck["step"] + 1
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"], device)
        np.random.set_state(ck["np_rng"])
    else:
        W, snap = initial_W(args, args.chains, args.N, args.d, model.sw, device, dtype)
        if snap:
            A = torch.load(snap)["A"].to(device, dtype).expand(args.chains, -1).clone()
            rec.cfg["init_snapshot"] = snap
            rec.save_config()
        else:
            A = model.sample_A(W / model.sw)

    print(f"[{rec.run_id}] d={args.d} P={args.P} N={args.N} T={args.T} wd_W={lang.wd_W:.3g} wd_A={lang.wd_A:.3g}"
          f" chains={args.chains} {args.dtype} gibbs_every={args.gibbs_every} start={start}", flush=True)
    rec.event("resumed" if args.resume else "running", kind="langevin_split", group=args.group, tag=args.tag,
              d=args.d, P=args.P, N=args.N, T=args.T, sa0=args.sa0, eps=args.eps, init=args.init,
              rng_seed=args.rng_seed, iters=args.steps, burn=args.burn, gibbs_every=args.gibbs_every,
              langevin_dir=args.langevin_dir)
    rec.open_tables(["iter", "chain", "E_over_T", "lr_W", "lr_A", "lam_W", "lam_A", "wall_s", "gibbs_changed",
                     "gibbs_crossed"], ["iter", "chain", "burn"] + OBS_KEYS, append=start > 0)

    final_start = int(args.steps * (1.0 - args.final_frac))
    lam_W = lam_A = None
    t_start = time.time()
    log_every = max(args.steps // 20, 1)
    try:
        for step in range(start, args.steps):
            pre = None
            if gibbs is not None and step % args.gibbs_every == 0:
                W, pre = gibbs_sweeps(gibbs, W, args)
                A = model.sample_A(W / model.sw)
            elif args.exact_A_every and step % args.exact_A_every == 0:
                A = model.sample_A(W / model.sw)
            if lam_W is None or step % args.curv_every == 0:
                lam_W, lam_A = lang.curvatures(W, A)
            scale = args.final_scale if step >= final_start else 1.0
            lr_W = scale * args.c_W / lam_W
            lr_A = scale * args.c_A / lam_A
            with torch.no_grad():
                E, gW, gA = lang.grads(W, A)
                W = W - lr_W * gW + math.sqrt(2.0 * lr_W * lang.T) * torch.randn_like(W)
                A = A - lr_A * gA + math.sqrt(2.0 * lr_A * lang.T) * torch.randn_like(A)
            if not torch.isfinite(E).all():
                raise FloatingPointError(f"non-finite energy at step {step} (lr_W={lr_W:.3g}, lr_A={lr_A:.3g})")

            if step % args.trace_every == 0 or pre is not None:
                for c in range(W.shape[0]):
                    rec.trace_w.writerow([step, c, f"{E[c].item() / lang.T:.8e}", f"{lr_W:.4e}", f"{lr_A:.4e}",
                                          f"{lam_W:.4e}", f"{lam_A:.4e}", f"{time.time() - t_start:.2f}",
                                          f"{pre['changed']:.4f}" if pre else "",
                                          f"{pre['crossed']:.4f}" if pre else ""])
            if step % args.obs_every == 0:
                rec.write_obs(step, args.burn, observables(model, W / model.sw, A, Xinf, Yinf))
            if step >= args.burn and args.save_every and (step - args.burn) % args.save_every == 0:
                rec.save_sample(step, W0=W, A=A)
            if (step + 1) % args.checkpoint_every == 0 or step == args.steps - 1:
                rec.flush()
                torch.save({"step": step, "W": W.cpu(), "A": A.cpu(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(device), "np_rng": np.random.get_state()},
                           ckpt_path)
            if step % log_every == 0 or step == args.steps - 1:
                extra = f"  gibbs crossed={pre['crossed']:.3f}" if pre else ""
                print(f"  step {step:8d}  E/T={E.mean().item() / lang.T:.6e}  lr_W={lr_W:.3e} lr_A={lr_A:.3e}"
                      f"  λ_W={lam_W:.3e} λ_A={lam_A:.3e}  t={time.time() - t_start:.1f}s{extra}", flush=True)
        rec.close()
        wall = time.time() - t_start
        pooled, per_chain, n_obs = rec.obs_summary()
        summary = {
            "run_id": rec.run_id,
            "kind": "langevin_split",
            "wall_s_this_session": wall,
            "steps": int(args.steps),
            "n_obs_post_burn": n_obs,
            "E_over_T_post_burn_mean": float(rec.trace_column("E_over_T", args.burn).mean()),
            "lam_W_final": lam_W,
            "lam_A_final": lam_A,
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
    rec.event("done", wall_s=round(wall, 1), L3=pooled["L3"][0], w0_sq=pooled["w0_sq"][0],
              frac_aligned=pooled["frac_aligned"][0], wperp_sq=pooled["wperp_sq"][0],
              test_mse=pooled["test_mse"][0])
    print(f"\n[{rec.run_id}] done in {wall:.1f}s")
    print_table(summary)


if __name__ == "__main__":
    main()
