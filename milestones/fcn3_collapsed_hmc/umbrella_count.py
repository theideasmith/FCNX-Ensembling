#!/usr/bin/env python3
"""
Equilibrium distribution of the read-in outer-well count n by umbrella sampling on one tagged read-in neuron.

Why: the outer-well count n = #{i : r_i^2 > thr},  r^2 = w0^2 / (1 + 2|w|^2), never changes in the chains (no read-in
neuron crosses the barrier), so each chain samples the posterior conditioned on its starting n. The chains are still
exact given n; what is missing is P(n).

Identity (read-in neurons are exchangeable). Tag neuron j and let m = number of OTHER neurons with r^2 > thr. Then

    P(n = m+1) / P(n = m) = R(m) * (N1 - m) / (m + 1),     R(m) = P(r_j^2 > thr, m) / P(r_j^2 < thr, m).

R(m) needs the distribution of r_j^2 with everything else in equilibrium. A sequence of windows adds a spring

    V_k = (k_u / 2) (r_j^2 - c_k)^2

to the collapsed potential; HMC on (W, B) plus the middle and read-in jump sweeps sample each biased posterior. The
read-in sweep carries the spring for neuron j (FastSweep.spring). It is needed: r_j^2 mixes w0 and q = |w_perp|^2, and
at fixed r_j^2 HMC alone does not move between (small w0, large q) and (larger w0, q ~ q0) states (first d = 150 try:
profiles from the two wells disagreed by ~30 nats). The reference (CPU) path has no spring in its read-in sweep and
runs the middle sweep only. WHAM removes the springs and gives the profile
F(s) = -log p(r_j^2 = s) and R(m).

Subcommands
    pull     from a run's final state, tag a neuron and drag it through the windows (short runs), saving a start state
             per window:  python umbrella_count.py pull --from runs/<run> --tag outer --out umb/<name>
    prod     run windows (a list of indices) from their start states and store r_j^2, m per iteration
    analyze  WHAM over all windows -> profile, R(m), P(m+1)/P(m)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fcn3_sampler import FCN3, make_data

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------------------------------------ spring on neuron j
class Spring:
    """Mixin: adds (k/2)(r_j^2 - c)^2 to the collapsed potential, in the reference (U) and fast (U_exact, grad32)
    paths. r^2 = w0^2 / (1 + 2|w|^2) with w = sw * ZW[j]."""

    def set_spring(self, j, c, k):
        self.sp_j, self.sp_c, self.sp_k = int(j), float(c), float(k)
        fs = getattr(self, "fsweep", None)
        if fs is not None:                      # the read-in jump sweep sees the spring too
            fs.spring.copy_(torch.tensor([float(j), float(c), float(k)], dtype=torch.float64))

    def _spring(self, ZW):
        w = ZW[self.sp_j].double() * self.sw
        S = 1 + 2 * (w * w).sum()
        r2 = w[0] ** 2 / S
        dV = self.sp_k * (r2 - self.sp_c)
        dr2 = -4 * r2 / S * w
        dr2[0] = dr2[0] + 2 * w[0] / S
        g = torch.zeros_like(ZW)
        g[self.sp_j] = (self.sw * dV * dr2).to(ZW.dtype)
        return 0.5 * self.sp_k * (r2 - self.sp_c) ** 2, g

    # reference path
    def U(self, ZW, ZB, grad=True):
        u, gW, gB = super().U(ZW, ZB, grad)
        V, g = self._spring(ZW)
        u = u + float(V)
        if gW is not None:
            gW = gW + g
        return u, gW, gB

    # fast path
    def U_exact(self, ZW, ZB):
        u, fac = super().U_exact(ZW, ZB)
        return u + float(self._spring(ZW)[0]), fac

    def grad32(self, ZW, ZB):
        gW, gB, ok = super().grad32(ZW, ZB)
        return gW + self._spring(ZW)[1].to(gW.dtype), gB, ok


def make_model(cfg, dev, fast=True):
    a = cfg["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
    X, Y = X.float(), Y.float()
    base = FCN3
    if fast:
        from fcn3_fast import FastFCN3 as base
    cls = type("SpringFCN3", (Spring, base), {})
    ks = tuple(range(1, a["nmax"] + 1, 2))
    return cls(X, Y, a["N1"], a["N2"], a["chi"], a["kappa"], a["sigA"], a["sigB"], a["sigw"], ks=ks,
               n_cand=a["n_cand"], cg_iters=a["cg_iters"], redraw_frac=a["redraw_frac"], seed=a["rng_seed"] + 1000)


def r2_all(m, ZW):
    return m._r2(ZW)


def run_window(m, ZW, ZB, step, iters, burn, leap, sweep_every, target_acc=0.8, rec=None, thr=0.18):
    """iters iterations of: sign flips (j's flip leaves r_j^2 unchanged), HMC with spring, middle sweep.
    Step size adapts during the first `burn` iterations. rec: list to append (r_j^2, m_others) to after burn."""
    fast = any(c.__name__ == "FastFCN3" for c in type(m).__mro__)
    u, gW, gB = (None, None, None)
    accs = []
    for it in range(iters):
        ZW, ZB = m.flip_signs(ZW, ZB)
        if not fast:
            u, gW, gB = m.U(ZW, ZB)
        ZW, ZB, u, gW, gB, acc, L = m.hmc_step(ZW, ZB, u, gW, gB, step, leap)
        if it < burn:
            step = math.exp(math.log(step) + (acc - target_acc) * 2.0 / math.sqrt(it + 10))
        accs.append(acc)
        if sweep_every and it % sweep_every == 0:
            W, B = m.sw * ZW, m.sb * ZB
            a = m.draw_a(W, B)
            W, B, a, _ = m.sweep(W, B, a, ("middle", "readin") if fast else ("middle",))
            ZW, ZB = W / m.sw, B / m.sb
        if rec is not None and it >= burn:
            r2 = r2_all(m, ZW)
            out = r2 > thr
            rec.append((float(r2[m.sp_j]), int(out.sum()) - int(out[m.sp_j])))
    return ZW, ZB, step, float(np.mean(accs[burn:] if iters > burn else accs))


# ------------------------------------------------------------------------------------------------ subcommands
def load_run(path, dev):
    run = Path(path)
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    return cfg, ck["ZW"].to(dev, torch.float32), ck["ZB"].to(dev, torch.float32), ck["step"]


def cmd_pull(a):
    dev = torch.device(a.device)
    cfg, ZW, ZB, step = load_run(a.src, dev)
    m = make_model(cfg, dev)
    r2 = r2_all(m, ZW).cpu().numpy()
    outer = np.where(r2 > a.thr)[0]
    inner = np.where(r2 < 0.03)[0]
    if a.tag == "outer":           # the outer neuron nearest the median outer r^2
        j = int(outer[np.argmin(abs(r2[outer] - np.median(r2[outer])))])
    else:                          # the inner neuron nearest the median inner r^2
        j = int(inner[np.argmin(abs(r2[inner] - np.median(r2[inner])))])
    cs = np.round(np.arange(0.0, a.cmax + 1e-9, a.dc), 5)
    # an outer neuron is dragged inward from the top window, an inner one outward from c = 0
    order = np.arange(len(cs))[::-1] if a.tag == "outer" else np.arange(len(cs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(src=str(a.src), tag=a.tag, j=j, r2_j0=float(r2[j]), n_outer=int(len(outer)), thr=a.thr, k=a.k,
                cs=cs.tolist(), N1=cfg["args"]["N1"], outer_r2=np.sort(r2[outer]).tolist(), cfg=cfg)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"tag {a.tag}: neuron {j}, r2 = {r2[j]:.4f}, n_outer = {len(outer)}, windows {cs[0]}..{cs[-1]} step {a.dc}",
          flush=True)
    for i in order:
        t0 = time.time()
        m.set_spring(j, cs[i], a.k)
        ZW, ZB, step, acc = run_window(m, ZW, ZB, step, a.iters, a.iters, a.leapfrog, a.sweep_every, thr=a.thr)
        torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step), out / f"start_{i:02d}.pt")
        print(f"  window {i:2d} c={cs[i]:.4f}: r2_j={float(r2_all(m, ZW)[j]):.4f} step={step:.4f} acc={acc:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)


def cmd_prod(a):
    dev = torch.device(a.device)
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    m = make_model(meta["cfg"], dev)
    j, cs = meta["j"], meta["cs"]
    for i in a.windows:
        f = out / f"win_{i:02d}.npz"
        if f.exists():
            continue
        ck = torch.load(out / f"start_{i:02d}.pt", map_location="cpu", weights_only=False)
        ZW, ZB, step = ck["ZW"].to(dev), ck["ZB"].to(dev), ck["step"]
        m.set_spring(j, cs[i], meta["k"])
        rec, t0 = [], time.time()
        ZW, ZB, step, acc = run_window(m, ZW, ZB, step, a.iters, a.burn, a.leapfrog, a.sweep_every, rec=rec,
                                       thr=meta["thr"])
        x = np.array(rec)
        np.savez(f, r2=x[:, 0], m=x[:, 1].astype(int), c=cs[i], k=meta["k"], acc=acc, step=step)
        print(f"window {i:2d} c={cs[i]:.4f}: <r2_j>={x[:, 0].mean():.4f} sd={x[:, 0].std():.4f} "
              f"m in {sorted(set(x[:, 1].astype(int)))} acc={acc:.2f} ({time.time() - t0:.0f}s)", flush=True)


def wham(r2s, cs, k, iters=20000, tol=1e-10):
    """Self-consistent WHAM / MBAR on the pooled samples. Returns per-sample log weights (unbiased, normalised)."""
    x = np.concatenate(r2s)
    Nk = np.array([len(r) for r in r2s], float)
    bias = 0.5 * k * (x[None, :] - np.asarray(cs)[:, None]) ** 2           # (K, n)
    f = np.zeros(len(cs))
    for _ in range(iters):
        a = np.log(Nk)[:, None] + f[:, None] - bias
        lden = np.logaddexp.reduce(a, axis=0)                                # log sum_k N_k e^{f_k - V_k}
        fn = -np.logaddexp.reduce(-bias - lden[None, :], axis=1)
        fn -= fn[0]
        if np.max(abs(fn - f)) < tol:
            f = fn
            break
        f = fn
    lw = -lden
    return x, lw - np.logaddexp.reduce(lw), f


def odds(x, mm, lw, thr, N1):
    """log R(m) and log P(m+1)/P(m) for every m present."""
    res = {}
    for m_ in sorted(set(mm.tolist())):
        s = mm == m_
        o, i = s & (x > thr), s & (x <= thr)
        if o.any() and i.any():
            lR = np.logaddexp.reduce(lw[o]) - np.logaddexp.reduce(lw[i])
            res[int(m_)] = dict(logR=float(lR), log_ratio=float(lR + math.log((N1 - m_) / (m_ + 1))),
                                n_samples=int(s.sum()))
    return res


def cmd_analyze(a):
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    files = sorted(out.glob("win_*.npz"))
    D = [np.load(f) for f in files]
    r2s, ms, cs = [d["r2"] for d in D], [d["m"] for d in D], [float(d["c"]) for d in D]
    x, lw, f = wham(r2s, cs, meta["k"])
    mm = np.concatenate(ms)
    res = odds(x, mm, lw, meta["thr"], meta["N1"])
    # block error: WHAM on halves of every window
    halves = []
    for h in (0, 1):
        sl = [r[h * len(r) // 2:(h + 1) * len(r) // 2] for r in r2s]
        sm = [q[h * len(q) // 2:(h + 1) * len(q) // 2] for q in ms]
        xh, lwh, _ = wham(sl, cs, meta["k"])
        halves.append(odds(xh, np.concatenate(sm), lwh, meta["thr"], meta["N1"]))
    # profile on a grid
    edges = np.linspace(0, max(cs) + 0.03, 61)
    idx = np.digitize(x, edges) - 1
    prof = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        s = idx == b
        if s.any():
            prof[b] = -np.logaddexp.reduce(lw[s])
    prof -= np.nanmin(prof)
    # overlap between neighbouring windows
    ov = [float(np.mean((r2s[i] > np.percentile(r2s[i + 1], 1)) & (r2s[i] < np.percentile(r2s[i + 1], 99))))
          for i in range(len(r2s) - 1)]
    summ = dict(tag=meta["tag"], j=meta["j"], n_outer_start=meta["n_outer"], windows=len(files), result=res,
                halves=halves, profile=dict(s=((edges[1:] + edges[:-1]) / 2).tolist(), F=prof.tolist()),
                window_mean=[float(r.mean()) for r in r2s], overlap_min=min(ov) if ov else None)
    (out / "analysis.json").write_text(json.dumps(summ, indent=1))
    print(f"{meta['tag']} neuron {meta['j']} (start n_outer = {meta['n_outer']}), {len(files)} windows, "
          f"min neighbour overlap {summ['overlap_min']}")
    for m_, r in res.items():
        hs = [h.get(m_, {}).get("log_ratio", float("nan")) for h in halves]
        print(f"  m = {m_:3d} others outer ({r['n_samples']} samples): log R = {r['logR']:+.3f}   "
              f"log P(n={m_ + 1})/P(n={m_}) = {r['log_ratio']:+.3f}   halves {hs[0]:+.3f} {hs[1]:+.3f}")
    print("  profile F(r_j^2):", " ".join(f"{s:.3f}:{F:.1f}" for s, F in zip(summ["profile"]["s"], prof)
                                           if np.isfinite(F)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--tag", choices=["outer", "inner"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thr", type=float, default=0.18)
    p.add_argument("--k", type=float, default=3e4)
    p.add_argument("--dc", type=float, default=0.01)
    p.add_argument("--cmax", type=float, default=0.30)
    p.add_argument("--iters", type=int, default=20)
    q = sub.add_parser("prod")
    q.add_argument("--out", required=True)
    q.add_argument("--windows", type=int, nargs="+", required=True)
    q.add_argument("--iters", type=int, default=250)
    q.add_argument("--burn", type=int, default=50)
    s = sub.add_parser("analyze")
    s.add_argument("--out", required=True)
    for x in (p, q):
        x.add_argument("--device", default="cuda:0")
        x.add_argument("--leapfrog", type=int, default=24)
        x.add_argument("--sweep-every", type=int, default=2)
    a = ap.parse_args()
    dict(pull=cmd_pull, prod=cmd_prod, analyze=cmd_analyze)[a.cmd](a)


if __name__ == "__main__":
    main()
