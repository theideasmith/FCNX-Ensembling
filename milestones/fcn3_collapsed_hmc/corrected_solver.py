#!/usr/bin/env python3
"""The theory's own solver with the networks' perpendicular term added to the hidden-neuron action only.

Hidden-neuron action term: theory -1/2 g~ (t . e)^2; here -1/2 g~ (t . e + c)^2 with c = alpha * gamma . e, gamma the
fit of the networks' measured c on e (measure_perp_pull.py; --modes 2: He1, He3; 6: all). t is still solved from the
theory's readout equation, unchanged: (sigma_A^2 K + kappa/P) t = y, K = E[e e^T] over the corrected hidden
distribution. So c changes the hidden-field distribution (and through H the read-in layer) but not the readout relation.

Implementation: the solver's internal tilt vector is v = t + alpha gamma (so its proposal and weights code uses the
corrected action as is); its readout solve is replaced by the root of lambda (v - alpha gamma) + sigma_A^2 K(v) (v -
alpha gamma) = y. The read-in stationarity (gradient in the read-in exponents at the current v) is the theory's own.
The theory is solved first; alpha is then raised from 0 in steps, each solved by Newton on |gradient| (the solver's
ascent assumes t minimises its potential, which no longer holds), warm-started from the previous step.
Reported: L3 = f_bar_3 / eps with f_bar = sigma_A^2 K t, hidden neurons per network in the far-He1 / large-h3 regions,
widths of h1, h3, outer read-in count.

    PYTHONPATH=.:../fcn3_matrix_theory python corrected_solver.py [--P 48000] [--modes 6] [--alphas 0 .25 .5 .75 1]
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

    class Corrected(F.BlazingFreeEnergy):
        shift = None                      # alpha * gamma (None: plain theory)
        r = 1.0                           # readout factor: f_bar = r sigma_A^2 K (t + kc * shift)
        kc = 0.0                          # 1: c also enters the readout (each neuron's readout follows t.e + c)

        def solve_t(self, L, qbar, M, t0, tol=1e-12, maxit=60):
            if self.shift is None and self.r == 1.0:
                return super().solve_t(L, qbar, M, t0, tol, maxit)
            p, y = self.p, self.y
            g = self.shift if self.shift is not None else torch.zeros_like(t0)
            r_, kc = self.r, self.kc
            with torch.no_grad():
                e, h, q = self.mc.features(L, qbar)
                lw = self.mc.logw(M)

                def resid(v):
                    lt = lw + 0.5 * p.gt * (e @ v) ** 2
                    w = torch.exp(lt - torch.logsumexp(lt, 0))
                    K = (e * w[:, None]).T @ e
                    return p.lam * (v - g) + r_ * p.sigA ** 2 * (K @ (v - g + kc * g)) - y, K, w

                v = t0.clone()
                r, K, w = resid(v)
                for it in range(maxit):
                    if float(r.abs().max()) < tol:
                        break
                    S = len(v)
                    Jm = torch.zeros(S, S, dtype=DT, device=v.device)
                    for i in range(S):                       # finite-difference Jacobian (S <= 6)
                        dv = torch.zeros_like(v)
                        dv[i] = 1e-6 * max(1.0, float(v[i].abs()))
                        Jm[:, i] = (resid(v + dv)[0] - r) / dv[i]
                    step = -torch.linalg.solve(Jm, r)
                    a = 1.0
                    while a > 1e-8:
                        r1, K1, w1 = resid(v + a * step)
                        if float(r1.norm()) < float(r.norm()):
                            break
                        a *= 0.5
                    v, r, K, w = v + a * step, r1, K1, w1
            return v, dict(K=K, w=w, ess=float(1 / (w * w).sum()), e=e, h=h, q=q, grad=float(r.abs().max()), its=it)

    return Corrected


def stationary(Fn, M, maxit=30, gtol=1e-3):
    gn = math.inf
    for it in range(maxit):
        Fn.rebase()
        phi = torch.zeros(Fn.nphi, dtype=DT, device=Fn.dev)
        W, g = Fn._value_grad(phi, M)
        if g is None:
            return M, math.inf
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
    return M, gn


def state(Fn, M, P):
    import fcn3_fixed as F
    p = Fn.p
    v = Fn.t
    sh = Fn.shift if Fn.shift is not None else torch.zeros_like(v)
    t = v - sh
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(M) + 0.5 * p.gt * (e @ v) ** 2, 0)
        fb = (Fn.r * p.sigA ** 2 * ((e * w[:, None]).T @ e) @ (t + Fn.kc * sh)).cpu().numpy()
        h, w = h.cpu().numpy(), w.cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    cut = po.barrier_cut({f"dens_{P}": mass / np.gradient(w0), f"w0_{P}": w0}, P)
    return dict(fb=fb, L3=fb[1] / p.eps, outer=po.N * mass[w0 >= cut].sum() / mass.sum(),
                far=po.N * w[reg["far He1"]].sum(), big3=po.N * w[reg["large h3"]].sum(),
                sd1=np.sqrt(w @ h[:, 0] ** 2), sd3=np.sqrt(w @ h[:, 1] ** 2), ess=float(1 / (w * w).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=48000)
    ap.add_argument("--modes", type=int, default=6, help="0: c = 0")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--r", type=float, default=1.0, help="readout factor (networks: 0.66 at P = 48000)")
    ap.add_argument("--c-in-readout", action="store_true", help="c also enters the readout relation")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    P = a.P
    pp = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
    gam = np.zeros(6)
    if a.modes:
        gam[:a.modes] = np.linalg.lstsq(pp["e"][:, :a.modes], pp["c"], rcond=None)[0]
    S = solver_class()
    p = po._params(P)
    F3 = S(replace(p, nmax=3), device=a.device, verbose=False)
    o3 = F3.solve()
    Fn = S(p, device=a.device, verbose=False)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    del F3
    M = o["M"]
    gt = torch.tensor(gam, dtype=DT, device=Fn.dev)
    Fn.kc = 1.0 if a.c_in_readout else 0.0
    print(f"P = {P}: c = alpha * gamma . e in the action{' and the readout' if a.c_in_readout else ''}, readout factor "
          f"1 + alpha ({a.r} - 1); gamma on {a.modes} modes = {np.round(gam, 0)}; theory t = "
          f"{np.round(Fn.t.cpu().numpy(), 0)}")
    print(f"   {'alpha':>5s} {'|g|':>8s} {'L3':>6s} {'far He1':>8s} {'large h3':>9s} {'sd h1':>6s} {'sd h3':>6s} "
          f"{'outer':>6s}  f_bar He1..He7          t")
    al_prev, k = 0.0, 0
    targets = sorted(set(a.alphas) | {0.0})
    al = 0.0
    while k < len(targets):
        tgt = targets[k]
        save = (Fn.theta_ref.clone(), M, Fn.mc, Fn.far_comps, Fn.t.clone(), Fn.shift)
        Fn.shift = tgt * gt if (tgt > 0 and np.any(gam != 0)) else None
        Fn.r = 1.0 + tgt * (a.r - 1.0)
        if Fn.shift is not None:
            Fn.t = Fn.t - (save[5] if save[5] is not None else 0) + Fn.shift      # keep t, move v
        M1, gn = stationary(Fn, M) if tgt > 0 else (M, 0.0)
        if gn > 1e-3:
            Fn.theta_ref, M, Fn.mc, Fn.far_comps, Fn.t, Fn.shift = save
            Fn.r = 1.0 + al_prev * (a.r - 1.0)
            Fn.mask = None
            mid = 0.5 * (al_prev + tgt)
            if tgt - al_prev < 1e-3:
                print(f"   no solution reached beyond alpha {al_prev:.4f} (|g| {gn:.1e}); stopping")
                break
            targets.insert(k, mid)
            continue
        M, al_prev = M1, tgt
        if any(abs(tgt - x) < 1e-12 for x in a.alphas) or tgt == 0.0:
            r = state(Fn, M, P)
            t = (Fn.t - (Fn.shift if Fn.shift is not None else 0)).cpu().numpy()
            print(f"   {tgt:5.3f} {gn:8.1e} {r['L3']:6.3f} {r['far']:8.2f} {r['big3']:9.2f} {r['sd1']:6.3f} "
                  f"{r['sd3']:6.3f} {r['outer']:6.1f}  {np.round(r['fb'][:4], 3)}  {np.round(t[:3], 0)}", flush=True)
        k += 1
    zs = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    H, n = po.sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()])
    reg = hr.regions(H[:, 0], H[:, 1])
    print(f"   networks       {np.mean([r['L3'][0] for r in groups[P]]):6.3f} {reg['far He1'].sum() / n:8.2f} "
          f"{reg['large h3'].sum() / n:9.2f} {H[:, 0].std():6.3f} {H[:, 1].std():6.3f}         "
          f"{np.round(pp['fn'].mean(0)[:4], 3)}  {np.round(pp['t'].mean(0)[:3], 0)}")


if __name__ == "__main__":
    main()
