#!/usr/bin/env python3
"""Surrogate theory: the FCN3 equations with the sampler's perpendicular residual (and readout factor r) put in by hand.

With --r: each hidden neuron's readout is r (sigma_A^2/N2)(t.e + c) as measured in the networks. The theory's
potential N1 D + N2 log Z_h(v) + (chi/r)(lambda/2 |v|^2 - y_eff . v) gives exactly that readout and leaves the
hidden-neuron action and the read-in coupling unchanged, i.e. the original solver with sigma_A^2 -> r sigma_A^2,
chi -> chi / r (g~ = sigma_A^2 chi / N2 unchanged) and target y_eff = y + lambda gamma.

In a network the mean output has a perpendicular part f_bar_perp (not a function of x0 alone), so the pull on hidden
neuron k is 1/2 g~ (t . e_k + c_k)^2, c_k = <t_perp | erf o h_k>, t_perp = -f_bar_perp / lambda, and its readout mean
follows t . e_k + c_k. In the sampler's networks c_k ~ gamma . e_k (R^2 0.80-0.86 on He1, He3; measure_perp_pull.py).
With that, the hidden-layer pull becomes 1/2 g~ ((t + gamma) . e)^2 and the readout condition
(sigma_A^2 K + lambda)(t + gamma) = y + lambda gamma: the theory with the target replaced by y_eff = y + lambda gamma.
The mean output is f_bar = y_eff - lambda (t + gamma), the solver's own f_bar for target y_eff; L3 = f_bar_3 / eps
against the true target.

gamma is fitted on He1 and He3 by default (--modes 6: all, whose higher components are not stable across networks), pooled over the
networks in plots/cache/perp_pull_P{P}.npz, and applied with factor 0 (the plain theory, as a check), 1 and 2.

    PYTHONPATH=.:../fcn3_matrix_theory python perp_surrogate.py [--Ps 24000 48000] [--factors 0 1 2]
"""
import argparse
import gc
import math
from dataclasses import replace

import numpy as np
import torch

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import DT

HERE = po.HERE


def shift_from_sampler(P, modes=2):
    z = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
    g, *_ = np.linalg.lstsq(z["e"][:, :modes], z["c"], rcond=None)
    return float(z["lam"]) * g                                   # target shift on the first `modes` Hermite modes


def solve(P, shift, device, r=1.0):
    import fcn3_fixed as F
    p = po._params(P)
    p = replace(p, sigA=p.sigA * math.sqrt(r), chi=p.chi / r)   # readout r sigma_A^2 K v; g~ unchanged
    F3 = F.BlazingFreeEnergy(replace(p, nmax=3), device=device, verbose=False)
    F3.y = F3.y + torch.tensor(shift[:2], dtype=DT, device=F3.dev)
    o3 = F3.solve()
    Fn = F.BlazingFreeEnergy(p, device=device, verbose=False)
    Fn.y = Fn.y.clone()
    Fn.y[:len(shift)] += torch.tensor(shift, dtype=DT, device=Fn.dev)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    fb = (Fn.y - p.lam * Fn.last["t"]).cpu().numpy()
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(o["M"]) + 0.5 * p.gt * (e @ Fn.t) ** 2, 0)
        h, w = h.cpu().numpy(), w.cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    z = {f"dens_{P}": mass / np.gradient(w0), f"w0_{P}": w0}
    cut = po.barrier_cut(z, P)
    out = dict(L1=fb[0], L3=fb[1] / p.eps, fb=fb, W=o["W"], outer=po.N * mass[w0 >= cut].sum() / mass.sum(), cut=cut,
               far=po.N * w[reg["far He1"]].sum(), big3=po.N * w[reg["large h3"]].sum(),
               sd1=np.sqrt(w @ h[:, 0] ** 2), sd3=np.sqrt(w @ h[:, 1] ** 2), ess=o["ess"])
    del F3, Fn
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[24000, 48000])
    ap.add_argument("--factors", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--modes", type=int, default=2, help="Hermite modes the shift is fitted on (2: He1, He3; 6: all)")
    ap.add_argument("--r", type=float, default=1.0, help="readout factor: each neuron's readout is r (sigma_A^2/N2) "
                    "(t.e + c) (networks: 0.66 at P = 48000); implemented as sigma_A^2 -> r sigma_A^2, chi -> chi/r")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        sh = shift_from_sampler(P, a.modes)
        zs = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
        H, n = po.sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()])
        rs = groups[P]
        L3s = np.mean([r["L3"][0] for r in rs])
        reg = hr.regions(H[:, 0], H[:, 1])
        sam = dict(L3=L3s, fb=zs["fn"].mean(0), outer=np.mean([po.N * r["outer"] for r in rs]) if "outer" in rs[0] else np.nan,
                   far=reg["far He1"].sum() / n, big3=reg["large h3"].sum() / n, sd1=H[:, 0].std(), sd3=H[:, 1].std())
        print(f"\nP = {P}: target shift lambda*gamma on the first {a.modes} modes = {np.round(sh, 4)}")
        hdr = f"   {'':14s} {'L3':>6s} {'f_bar He1..He11':>46s} {'far He1':>8s} {'large h3':>9s} {'sd h1':>6s} {'sd h3':>6s}"
        print(hdr)
        for fac in a.factors:
            r = solve(P, fac * sh, a.device, a.r)
            print(f"   theory x{fac:<4g}   {r['L3']:6.3f} {np.array2string(np.round(r['fb'], 3), max_line_width=200):>46s} "
                  f"{r['far']:8.2f} {r['big3']:9.2f} {r['sd1']:6.3f} {r['sd3']:6.3f}   (outer read-in {r['outer']:.1f}, "
                  f"ESS {r['ess']:.0f})", flush=True)
        print(f"   {'sampler':14s} {sam['L3']:6.3f} {np.array2string(np.round(sam['fb'], 3), max_line_width=200):>46s} "
              f"{sam['far']:8.2f} {sam['big3']:9.2f} {sam['sd1']:6.3f} {sam['sd3']:6.3f}")


if __name__ == "__main__":
    main()
