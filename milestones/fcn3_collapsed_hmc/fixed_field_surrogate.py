#!/usr/bin/env python3
"""Surrogate theory with the networks' residual held fixed in the hidden-neuron action.

The hidden-neuron action term is -1/2 g~ <t | erf o h>^2 with t the residual / (kappa/P) over the whole input space.
Its x0 part gives t . e; its perpendicular part gives c = <t_perp | erf o h>. Here both are taken from the sampled
networks and NOT re-solved:
    t . e  ->  t_s . e        t_s: the networks' population residual in He1..He11 (measure_perp_pull.py, mean of 9)
    c      ->  gamma . e      gamma: least-squares fit of the measured c on e over all hidden neurons (all six modes)
so the term is -1/2 g~ ((t_s + gamma) . e)^2. The read-in layer, G, qbar, M and the hidden-field distribution are
solved self-consistently as in the theory; only the readout equation (E4) is switched off (t is fixed).
Runs per P:
    theory t       t fixed at the theory's own solution (check: must reproduce the theory)
    t_s, c = 0     the networks' x0 residual only
    t_s + c        the networks' x0 residual and perpendicular part
Reported: hidden neurons per network in the far-He1 / large-h3 regions, widths of h1 and h3, outer read-in count, and
sigma_A^2 K u (the x0 output the theory's readout relation would give for this u).

    PYTHONPATH=.:../fcn3_matrix_theory python fixed_field_surrogate.py [--Ps 24000 48000]
"""
import argparse
import math
import gc
from dataclasses import replace

import numpy as np
import torch

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import DT

HERE = po.HERE


def make_solver_class():
    import fcn3_fixed as F

    class FixedT(F.BlazingFreeEnergy):
        """BlazingFreeEnergy with t held at self.u_fixed (no readout equation)."""

        def solve_t(self, L, qbar, M, t0, tol=1e-13, maxit=60):
            p = self.p
            u = self.u_fixed[:self.S]
            with torch.no_grad():
                e, h, q = self.mc.features(L, qbar)
                lt = self.mc.logw(M) + 0.5 * p.gt * (e @ u) ** 2
                w = torch.exp(lt - torch.logsumexp(lt, 0))
                K = (e * w[:, None]).T @ e
            return u.clone(), dict(K=K, w=w, ess=float(1 / (w * w).sum()), e=e, h=h, q=q, grad=0.0, its=0)

        def hessian(self, phi, M):
            t = self.u_fixed[:self.S]
            if len(self.mc.z) < 50000:
                Hz = torch.autograd.functional.hessian(lambda ph: self.f_obj(torch.cat([ph, t]), M), phi.detach(),
                                                       vectorize=True)
            else:
                from torch.func import hessian as fhess
                Hz = fhess(lambda ph: self.f_obj(torch.cat([ph, t]), M))(phi.detach())
            return Hz

    return FixedT


def solve(P, u, device):
    import fcn3_fixed as F
    FixedT = make_solver_class()
    p = po._params(P)
    ut = torch.tensor(u, dtype=DT, device=device)
    F3 = FixedT(replace(p, nmax=3), device=device, verbose=False)
    F3.u_fixed = ut[:2].clone()
    F3.t = F3.u_fixed.clone()
    o3 = F3.solve()
    Fn = FixedT(p, device=device, verbose=False)
    Fn.u_fixed = ut.clone()
    Fn.t = ut.clone()
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(o["M"]) + 0.5 * p.gt * (e @ ut) ** 2, 0)
        fb = (p.sigA ** 2 * ((e * w[:, None]).T @ e) @ ut).cpu().numpy()
        h, w = h.cpu().numpy(), w.cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    cut = po.barrier_cut({f"dens_{P}": mass / np.gradient(w0), f"w0_{P}": w0}, P)
    out = dict(fb=fb, L3=fb[1] / p.eps, outer=po.N * mass[w0 >= cut].sum() / mass.sum(),
               far=po.N * w[reg["far He1"]].sum(), big3=po.N * w[reg["large h3"]].sum(),
               sd1=np.sqrt(w @ h[:, 0] ** 2), sd3=np.sqrt(w @ h[:, 1] ** 2), ess=o["ess"], conv=o["converged"])
    del F3, Fn
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[24000, 48000])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        pp = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
        gam, *_ = np.linalg.lstsq(pp["e"], pp["c"], rcond=None)
        t_s, t_th = pp["t"].mean(0), z[f"t_{P}"]
        H, n = po.sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()])
        reg = hr.regions(H[:, 0], H[:, 1])
        print(f"\nP = {P}\n   theory t      {np.round(t_th, 0)}\n   networks t_s  {np.round(t_s, 0)}\n"
              f"   gamma (c fit) {np.round(gam, 0)}")
        print(f"   {'':16s} {'far He1':>8s} {'large h3':>9s} {'sd h1':>6s} {'sd h3':>6s} {'outer read-in':>14s}   "
              f"sigma_A^2 K u (He1..He7)")
        for lab, u in (("theory t", t_th), ("t_s, c = 0", t_s), ("t_s + c", t_s + gam)):
            try:
                r = solve(P, u, a.device)
                print(f"   {lab:16s} {r['far']:8.2f} {r['big3']:9.2f} {r['sd1']:6.3f} {r['sd3']:6.3f} {r['outer']:14.1f}   "
                      f"{np.round(r['fb'][:4], 3)}  (converged {r['conv']}, ESS {r['ess']:.0f})", flush=True)
            except Exception as ex:
                print(f"   {lab:16s} failed: {ex!r}"[:300], flush=True)
        print(f"   {'networks':16s} {reg['far He1'].sum() / n:8.2f} {reg['large h3'].sum() / n:9.2f} "
              f"{H[:, 0].std():6.3f} {H[:, 1].std():6.3f}   output {np.round(pp['fn'].mean(0)[:4], 3)}")


if __name__ == "__main__":
    main()


def solve_stationary(P, u, theta, M, device, maxit=40, gtol=1e-6, log=None):
    """Stationary point (not a maximum) of the read-in / hidden-field equations with t fixed at u, by plain Newton
    (step = -H^-1 g, line search on |g|), warm-started from read-in exponents theta and M. With t fixed the theory's
    equilibrium is a saddle in the read-in variables, so the solver's own ascent walks away from it."""
    import fcn3_fixed as F
    FixedT = make_solver_class()
    p = po._params(P)
    Fn = FixedT(p, device=device, verbose=False)
    ut = torch.tensor(u, dtype=DT, device=Fn.dev)
    Fn.u_fixed, Fn.t = ut, ut.clone()
    Fn.prune_nats = math.inf
    Fn.mask, Fn.nsteps, Fn.nguard, Fn.nprop = None, 0, 0, 0
    Fn.theta_ref = torch.tensor(theta, dtype=DT, device=Fn.dev)
    Fn.rebase()
    Fn.last = {}
    Fn._M_now()
    Fn.set_samples(15, M, fresh=True)
    gn = math.inf
    for it in range(maxit):
        Fn.rebase()
        phi = torch.zeros(Fn.nphi, dtype=DT, device=Fn.dev)
        W, g = Fn._value_grad(phi, M)
        gn = float(g.norm())
        if gn < gtol:
            break
        Hm = Fn.hessian(phi, M)
        Hm = 0.5 * (Hm + Hm.T)
        ev, U = torch.linalg.eigh(Hm)
        lam = torch.where(ev.abs() < 1e-10 * float(ev.abs().max()), torch.full_like(ev, -1e-10 * float(ev.abs().max())), ev)
        step = -U @ ((U.T @ g) / lam)
        mx = float(step.abs().max())
        if mx > 2.0:
            step = step * (2.0 / mx)
        a = 1.0
        while a > 1e-6:
            W1, g1 = Fn._value_grad(phi + a * step, M)
            if W1 is not None and float(g1.norm()) < (1 - 1e-4 * a) * gn:
                break
            a *= 0.5
        else:
            break
        Fn.rebase(phi + a * step)
        M = Fn._M_now()
        if Fn.last.get("ess", 1e9) < 0.02 * len(Fn.mc.z):
            Fn.set_samples(15, M)
        if log:
            log(f"      newton {it}: |g| {gn:.2e} -> step {a:.3g}, M {M:.2f}, positive Hessian eigenvalues {int((ev > 0).sum())}")
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(M) + 0.5 * p.gt * (e @ ut) ** 2, 0)
        fb = (p.sigA ** 2 * ((e * w[:, None]).T @ e) @ ut).cpu().numpy()
        h, w = h.cpu().numpy(), w.cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    cut = po.barrier_cut({f"dens_{P}": mass / np.gradient(w0), f"w0_{P}": w0}, P)
    out = dict(theta=Fn.theta_ref.cpu().numpy(), M=M, gnorm=gn, fb=fb, outer=po.N * mass[w0 >= cut].sum() / mass.sum(),
               far=po.N * w[reg["far He1"]].sum(), big3=po.N * w[reg["large h3"]].sum(),
               sd1=np.sqrt(w @ h[:, 0] ** 2), sd3=np.sqrt(w @ h[:, 1] ** 2), ess=float(1 / (w * w).sum()))
    del Fn
    gc.collect()
    torch.cuda.empty_cache()
    return out
