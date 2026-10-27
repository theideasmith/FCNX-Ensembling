#!/usr/bin/env python3
"""
Step-by-step check that readin_action_theory.py solves the closed loop correctly at the
parameters of an HMC group, with independent re-computations at every step:

  1. raw HMC parameters -> reduced (T, β, λ, Ĉ), printed with their formulas
  2. Newton solve (production code): D and the loop residual |(I + J(D)/λ) D - √Ĉ y|
  3. J(D) recomputed by adaptive quadrature (scipy.integrate.quad), not the sinh grid;
     loop residual re-evaluated with that J
  4. an independent solver: scipy.optimize.root (MINPACK hybrd) on the loop equation, with J on a
     plain uniform grid with Simpson weights
  5. uniqueness: Newton from several very different starting points
  6. grid convergence: Newton on coarser/finer/longer sinh grids
  7. the plotted -log p(w) integrates to 1 on the full line; L1, L3 from D

    python verify_theory_solution.py --group grok_k0p07
"""

import argparse

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import root

from diag_cavity_action import group_args
from readin_action_theory import Kernel, action, analyse, features, reduced_params, solve_loop
from plot_readin_action import theory_curve


def J_quad(D, T, beta):
    """J = <c c^T> under p(w) ∝ exp(-β v~(w)) on w ≥ 0, by adaptive quadrature."""
    wf = np.linspace(0, 20, 200001)
    v = action(wf, D, T)
    vmin = v.min()
    wcut = wf[np.where(beta * (v - vmin) < 800)[0][-1]] + 0.5      # weight < e^-800 beyond
    crit = wf[1:-1][np.diff(np.sign(np.diff(v))) != 0]              # wells and barriers as breakpoints
    pts = [p for p in crit if 0 < p < wcut]

    def integ(f):
        return quad(lambda w: f(w) * np.exp(-beta * (action(np.array([w]), D, T)[0] - vmin)), 0, wcut,
                    points=pts or None, limit=500, epsabs=0, epsrel=1e-11)[0]

    c = lambda w, k: features(np.array([w]), T)[k][0]
    Z = integ(lambda w: 1.0)
    J = np.array([[integ(lambda w: c(w, 0) ** 2), integ(lambda w: c(w, 0) * c(w, 1))],
                  [0.0, integ(lambda w: c(w, 1) ** 2)]]) / Z
    J[1, 0] = J[0, 1]
    return J


def J_uniform(D, T, beta, wmax=20.0, n=400001):
    """J on a plain uniform grid with Simpson weights (independent of the sinh grid)."""
    w = np.linspace(0, wmax, n)
    v = action(w, D, T)
    e = np.exp(-beta * (v - v.min()))
    c1, c3 = features(w, T)
    Z = simpson(e, x=w)
    return np.array([[simpson(e * c1 * c1, x=w), simpson(e * c1 * c3, x=w)],
                     [simpson(e * c1 * c3, x=w), simpson(e * c3 * c3, x=w)]]) / Z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="grok_k0p07")
    args = ap.parse_args()
    a = group_args(args.group)
    d, P, N, s0, sa0, chi, eps = a["d"], a["P"], a["N"], a["s0"], a["sa0"], a["chi"], a["eps"]
    kappa = a["T"] / 2
    y = np.array([1.0, eps])

    print(f"== 1. parameters ({args.group}) ==")
    print(f"HMC run: d={d} P={P} N={N} T_langevin={a['T']} s0={s0} sa0={sa0:.6g} chi={chi} eps={eps:.6f}")
    print(f"  read-in prior per component  σ_W² = s0/d          = {s0 / d:.6g}")
    print(f"  readout prior per weight     σ_A²/N = sa0/(N χ)    = {sa0 / (N * chi):.6g}   (= 8/N² for sa0 = 8/N)")
    print(f"  noise variance               κ = T/2              = {kappa:.6g}")
    T, beta, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
    print(f"  theory σ_w² = s0, σ_A² = sa0 (so the kernel is σ_A² <c c>):")
    print(f"  T  = 1 + 2 σ_w² (d-1)/d                 = {T:.6f}")
    print(f"  β  = d / (2 σ_w²)                       = {beta:.6f}")
    print(f"  λ  = κ / (P σ_A²)                       = {lam:.6g}")
    print(f"  Ĉ  = σ_w² σ_A² χ P² / (N d κ²)          = {Chat:.6g}    √Ĉ y = {np.round(np.sqrt(Chat) * y, 6)}")

    print("\n== 2. production Newton solve (sinh grid, 24001 pts to w=300) ==")
    K = Kernel(T, beta)
    D, res = solve_loop(Chat, lam, eps, K)
    _, p, logZ, J, _ = K.stats(D)
    loop = (np.eye(2) + J / lam) @ D - np.sqrt(Chat) * y
    print(f"  D = {D}")
    print(f"  J(D) = {J.ravel()}")
    print(f"  loop residual (I + J/λ) D - √Ĉ y = {loop}   (|.| = {np.abs(loop).max():.2e})")

    print("\n== 3. J by adaptive quadrature at the same D ==")
    Jq = J_quad(D, T, beta)
    loop_q = (np.eye(2) + Jq / lam) @ D - np.sqrt(Chat) * y
    print(f"  J_quad = {Jq.ravel()}")
    print(f"  max |J_grid - J_quad| / |J_quad| = {np.abs(J - Jq).max() / np.abs(Jq).max():.2e}")
    print(f"  loop residual with J_quad = {loop_q}   (relative to √Ĉ y: {np.abs(loop_q / (np.sqrt(Chat) * y)).max():.2e})")

    print("\n== 4. independent solver: scipy.optimize.root on the loop equation, uniform Simpson grid ==")
    F = lambda x: (np.eye(2) + J_uniform(x, T, beta, n=40001) / lam) @ x - np.sqrt(Chat) * y
    sol = root(F, np.sqrt(Chat) * y, method="hybr", tol=1e-13)
    print(f"  success={sol.success} ({sol.message.strip()}), {sol.nfev} evaluations: D = {sol.x}")
    print(f"  max |D_newton - D_root| = {np.abs(D - sol.x).max():.2e}")

    print("\n== 5. uniqueness: Newton from different starts ==")
    for D0 in ([0.0, 0.0], [10 * D[0], 10 * D[1]], [-D[0], -D[1]], [D[0], -3 * D[1]], [5.0, 5.0]):
        Ds, r = solve_loop(Chat, lam, eps, K, D0=D0)
        print(f"  D0 = {np.round(D0, 3)!s:>22} -> D = {Ds}  (residual {r:.1e})")

    print("\n== 6. grid convergence ==")
    for g in (dict(nw=6001), dict(nw=96001), dict(wmax=1200.0, nw=48001), dict(a=0.5, nw=24001)):
        Dg, _ = solve_loop(Chat, lam, eps, Kernel(T, beta, **g))
        print(f"  {g!s:>34}: D = {Dg}   max|ΔD| = {np.abs(Dg - D).max():.2e}")

    print("\n== 7. plotted density and observables ==")
    w = np.linspace(-6, 6, 600001)
    nlp, th = theory_curve(a, T, w)
    print(f"  ∫ exp(-nlp) dw over the full line = {simpson(np.exp(-nlp), x=w):.10f}   (should be 1)")
    out = analyse(D, lam, eps, K)
    fbar = J @ np.linalg.solve(J + lam * np.eye(2), y)
    print(f"  f̄ = J (J + λ I)⁻¹ y = {fbar}  ->  L1 = {out['L1']:.6f}, L3 = f̄_3/ε = {out['L3']:.6f}")
    print(f"  residual Δ = y - f̄ = {y - fbar};  D / √Ĉ = {D / np.sqrt(Chat)}  (equal by the loop equation)")
    print("  basins: " + ", ".join(f"{b['basin']} weight={b['weight']:.4f}" for b in out["basins"]))


if __name__ == "__main__":
    main()
