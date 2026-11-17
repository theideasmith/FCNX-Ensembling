#!/usr/bin/env python3
"""The theory with the networks' residual held fixed in the hidden-neuron action, by continuation.

Hidden-neuron action term: -1/2 g~ <t | erf o h>^2 = -1/2 g~ (t . e + c)^2 (t: residual / (kappa/P); its x0 part gives
t . e, its perpendicular part c). The theory has c = 0 and t from its readout equation. Here t . e + c is replaced by
u . e with u fixed, moved from the theory's own t (alpha = 0) to the networks' values (alpha = 1):
    u = t_s            the networks' population residual in He1..He11, c = 0
    u = t_s + gamma    the same plus c = gamma . e (gamma: fit of the measured c on e, all neurons, all six modes)
(measure_perp_pull.py, mean of 9 networks). Everything else (read-in layer, G, qbar, M, hidden-field distribution) is
solved self-consistently.

Procedure: the theory is solved as usual; then, in the same solver object (same hidden-field samples and proposal,
including the far basins it found), t is frozen and the read-in equations are solved for a stationary point by plain
Newton (step = -H^-1 g, line search on |g|). With t fixed the theory's equilibrium is a saddle in the read-in variables,
so the solver's ascent cannot be used. alpha = 0 must stay at the theory's solution (check).

    PYTHONPATH=.:../fcn3_matrix_theory python fixed_field_continuation.py [--P 48000] [--steps 10]
"""
import argparse
import math
from dataclasses import replace

import numpy as np
import torch

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import DT

HERE = po.HERE


def solver_class():
    import fcn3_fixed as F

    class Switchable(F.BlazingFreeEnergy):
        fix = False

        def solve_t(self, L, qbar, M, t0, tol=1e-13, maxit=60):
            if not self.fix:
                return super().solve_t(L, qbar, M, t0, tol, maxit)
            p, u = self.p, self.u_fixed
            with torch.no_grad():
                e, h, q = self.mc.features(L, qbar)
                lt = self.mc.logw(M) + 0.5 * p.gt * (e @ u) ** 2
                w = torch.exp(lt - torch.logsumexp(lt, 0))
                K = (e * w[:, None]).T @ e
            return u.clone(), dict(K=K, w=w, ess=float(1 / (w * w).sum()), e=e, h=h, q=q, grad=0.0, its=0)

        def hessian(self, phi, M):
            if not self.fix:
                return super().hessian(phi, M)
            t = self.u_fixed
            from torch.func import hessian as fhess
            return fhess(lambda ph: self.f_obj(torch.cat([ph, t]), M))(phi.detach())

    return Switchable


def theory_then_freeze(P, device):
    S = solver_class()
    p = po._params(P)
    F3 = S(replace(p, nmax=3), device=device, verbose=False)
    o3 = F3.solve()
    Fn = S(p, device=device, verbose=False)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    del F3
    return Fn, o


def stationary(Fn, M, maxit=30, gtol=1e-3):
    gn = math.inf
    for it in range(maxit):
        Fn.rebase()
        phi = torch.zeros(Fn.nphi, dtype=DT, device=Fn.dev)
        W, g = Fn._value_grad(phi, M)
        if g is None:
            return M, math.inf, None
        gn = float(g.norm())
        if gn < gtol:
            break
        Hm = Fn.hessian(phi, M)
        Hm = 0.5 * (Hm + Hm.T)
        ev, U = torch.linalg.eigh(Hm)
        floor = 1e-10 * float(ev.abs().max())
        lam = torch.where(ev.abs() < floor, torch.full_like(ev, -floor), ev)
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
        if Fn.last.get("ess", 1e9) < 0.05 * len(Fn.mc.z):
            Fn.refit(M)
    return M, gn, int((ev > 0).sum()) if gn >= gtol else None


def state(Fn, M, P):
    import fcn3_fixed as F
    p = Fn.p
    u = Fn.t
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(M) + 0.5 * p.gt * (e @ u) ** 2, 0)
        fb = (p.sigA ** 2 * ((e * w[:, None]).T @ e) @ u).cpu().numpy()
        h, w = h.cpu().numpy(), w.cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    cut = po.barrier_cut({f"dens_{P}": mass / np.gradient(w0), f"w0_{P}": w0}, P)
    return dict(fb=fb, outer=po.N * mass[w0 >= cut].sum() / mass.sum(), far=po.N * w[reg["far He1"]].sum(),
                big3=po.N * w[reg["large h3"]].sum(), sd1=np.sqrt(w @ h[:, 0] ** 2), sd3=np.sqrt(w @ h[:, 1] ** 2),
                ess=float(1 / (w * w).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=48000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    P = a.P
    pp = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
    gam, *_ = np.linalg.lstsq(pp["e"], pp["c"], rcond=None)
    t_s = pp["t"].mean(0)
    for lab, target in (("t_s + c", t_s + gam), ("t_s, c = 0", t_s)):
        Fn, o = theory_then_freeze(P, a.device)
        t_th = Fn.t.clone()
        M = o["M"]
        Fn.fix = True
        print(f"\nP = {P}: u from the theory's t (alpha 0) to {lab} (alpha 1)")
        print(f"   theory t {np.round(t_th.cpu().numpy(), 0)}\n   target   {np.round(target, 0)}")
        print(f"   {'alpha':>5s} {'|g|':>8s} {'far He1':>8s} {'large h3':>9s} {'sd h1':>6s} {'sd h3':>6s} {'outer':>6s}"
              f"  sigma_A^2 K u (He1..He7)")
        tgt = torch.tensor(target, dtype=DT, device=Fn.dev)
        al, dal, report = 0.0, 1.0 / a.steps, 0.0
        while True:
            save = (Fn.theta_ref.clone(), M, Fn.mc, Fn.far_comps)
            Fn.u_fixed = t_th + al * (tgt - t_th)
            Fn.t = Fn.u_fixed.clone()
            M1, gn, npos = stationary(Fn, M)
            if gn > 1e-3:                                    # failed: back up and take a smaller step
                Fn.theta_ref, M, Fn.mc, Fn.far_comps = save
                Fn.mask = None
                if al == 0.0 or dal < 1.0 / 1280:
                    print(f"   no stationary point beyond alpha {al - dal:.4f} (|g| {gn:.1e} at {al:.4f}); stopping")
                    break
                al -= dal
                dal /= 2
                al += dal
                continue
            M = M1
            if al >= report - 1e-9 or al >= 1.0:
                r = state(Fn, M, P)
                print(f"   {al:5.3f} {gn:8.1e} {r['far']:8.2f} {r['big3']:9.2f} {r['sd1']:6.3f} {r['sd3']:6.3f} "
                      f"{r['outer']:6.1f}  {np.round(r['fb'][:4], 3)}  ESS {r['ess']:.0f}", flush=True)
                report = al + 0.1
            if al >= 1.0:
                break
            al = min(1.0, al + dal)
        del Fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
