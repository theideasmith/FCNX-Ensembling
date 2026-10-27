#!/usr/bin/env python3
"""
Read-in mean-field theory (readin_action_theory.py) with the effective-ridge correction from the
kernel modes the He1/He3(x0) block leaves out.

Kernel. K(x,x') = σ_A² E_w[erf(w·x) erf(w·x')], w = (w0, w_perp), w0 ~ theory p(w0), |w_perp|² = q
frozen at the prior s0 (d-1)/d. With s = |w| and unit direction ŵ, erf(s ŵ·x) = Σ_k a_k(s) He_k(ŵ·x)/√k!,
    a_k(s) = (2/√π) (-1)^{(k-1)/2} (k-1)!! / √k! · s^k / (1+2s²)^{k/2}       (odd k)
Degree-k modes split by the number j of x0 factors (m = k - j orthogonal factors). In the large-d
diagonal approximation the family (k, j) holds n_m = C(d-2+m, m) modes, each with eigenvalue
    η_{k,j} = σ_A² E[a_k² C(k, j) ŵ0^{2j} ŵ⊥^{2m}] / n_m,
summing over j to σ_A² E[a_k²]. The j = k families are He1(x0), He3(x0), ... ; (k, j) = (1, 1) and
(3, 3) are exactly the theory's J (a_1 ŵ0 = c1, a_3 ŵ0³ = c3).

Effective ridge (Canatar-Bordelon-Pehlevan / Jacot et al.), over every mode outside the theory's
He1/He3(x0) block plus that block's two eigenvalues:
    κ̃ = κ + κ̃ Σ_k η_k / (P η_k + κ̃)
Closed loop with κ̃: λ̃ = κ̃/(P σ_A²), training residual = (κ/κ̃) × population residual, and the drive
uses the training residual:
    (I + J/λ̃) D = (κ/κ̃) √Ĉ y
iterated jointly with p(w0) (through J and the η's) to self-consistency.

    python effective_ridge_theory.py
"""

from math import comb, factorial, sqrt

import numpy as np

from readin_action_theory import Kernel, action, analyse, reduced_params, solve_loop

KS = (1, 3, 5, 7)


def dfact(n):
    return 1 if n <= 0 else n * dfact(n - 2)


def a_k(s, k):
    return (2 / sqrt(np.pi)) * (-1) ** ((k - 1) // 2) * dfact(k - 1) / sqrt(factorial(k)) * s ** k / (1 + 2 * s * s) ** (k / 2)


def mode_families(p, w, q, d, sa0):
    """[(k, j, n_modes, eta)] for every family; p = theory weights on the w0 ≥ 0 grid w."""
    s2 = w * w + q
    s = np.sqrt(s2)
    u0, uperp = w * w / s2, q / s2
    out = []
    for k in KS:
        ak2 = a_k(s, k) ** 2
        for j in range(k + 1):
            m = k - j
            n = comb(d - 2 + m, m)
            power = sa0 * (p @ (ak2 * comb(k, j) * u0 ** j * uperp ** m))
            out.append((k, j, n, power / n))
    return out


def effective_ridge(fams, kappa, P, J_eigs, iters=500):
    """Solve κ̃ = κ + κ̃ Σ η/(Pη + κ̃) over bulk families (with multiplicities) and the J block eigenvalues."""
    kt = kappa
    for _ in range(iters):
        s = sum(n * eta / (P * eta + kt) for (k, j, n, eta) in fams if not (j == k and k in (1, 3)))
        s += sum(e / (P * e + kt) for e in J_eigs)
        new = kappa / (1 - s)
        if abs(new - kt) < 1e-14:
            break
        kt = 0.5 * kt + 0.5 * new
    return kt


def solve_with_ridge(N, d, s0, sa0, P, kappa, eps, chi=1.0, iters=100, verbose=False):
    T, beta, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
    q = s0 * (d - 1) / d
    K = Kernel(T, beta)
    y = np.array([1.0, eps])
    D, _ = solve_loop(Chat, lam, eps, K)
    kt = kappa
    for it in range(iters):
        _, p, _, J, _ = K.stats(D)
        fams = mode_families(p, K.w, q, d, sa0)
        kt_new = effective_ridge(fams, kappa, P, sa0 * np.linalg.eigvalsh(J))
        r = kappa / kt_new
        lam_e = kt_new / (P * sa0)
        D_new, _ = solve_loop(Chat * r * r, lam_e, eps, K, D0=D)
        done = abs(kt_new - kt) < 1e-12 and np.abs(D_new - D).max() < 1e-10
        kt, D = kt_new, D_new
        if verbose:
            print(f"   it {it}: κ̃/κ = {kt / kappa:.6f}  D = {D}")
        if done:
            break
    _, p, _, J, _ = K.stats(D)
    fams = mode_families(p, K.w, q, d, sa0)
    o = analyse(D, kt / (P * sa0), eps, K)
    return dict(T=T, beta=beta, lam=lam, Chat=Chat, kt=kt, r=kappa / kt, D=D, p=p, K=K, fams=fams,
                L1=o["L1"], L3=o["L3"], w0_sq=p @ K.w ** 2)


def outer_well(D, T, beta):
    w = np.linspace(0, 3, 30001)
    V = beta * action(w, D, T)
    i = [j for j in range(1, len(w) - 1) if V[j] < V[j - 1] and V[j] <= V[j + 1] and w[j] > 0.3]
    return w[i[0]] if i else float("nan")


def main():
    eps = -0.3 * sqrt(6)
    # measured from HMC (training-set / population He3 residual) and HMC outer-well positions
    cases = [("κ=0.07 d=150", dict(N=700, d=150, P=3000, kappa=0.07, sa0=8 / 700), 0.956, 0.98),
             ("κ=0.1 d=150", dict(N=700, d=150, P=3000, kappa=0.1, sa0=8 / 700), 0.960, 0.88),
             ("κ=0.15 d=150", dict(N=700, d=150, P=3000, kappa=0.15, sa0=8 / 700), 0.964, None),
             ("κ=0.2 d=150", dict(N=700, d=150, P=3000, kappa=0.2, sa0=8 / 700), 0.969, None),
             ("κ=0.3 d=150", dict(N=700, d=150, P=3000, kappa=0.3, sa0=8 / 700), 0.977, None)]
    for b, r_meas, well in ((1, 0.926, 1.12), (2 ** 0.5, 0.890, 1.12)):
        al = b ** 1.25
        cases.append((f"ray d={round(150 * b ** 0.5)}", dict(N=round(1500 * b ** 0.25), d=round(150 * b ** 0.5),
                      P=round(3000 * al ** 0.75), kappa=0.015 * (al / b) ** 0.75, sa0=(8 / 1500) * b ** -0.75), r_meas, well))
    for b in (2, 4, 16, 64, 256):
        al = b ** 1.25
        cases.append((f"ray d={round(150 * b ** 0.5)}", dict(N=round(1500 * b ** 0.25), d=round(150 * b ** 0.5),
                      P=round(3000 * al ** 0.75), kappa=0.015 * (al / b) ** 0.75, sa0=(8 / 1500) * b ** -0.75), None, None))

    print(f"{'case':>14s} | {'κ/κ̃ th':>7s} {'meas':>6s} | {'L3 plain':>8s} {'L3 ridge':>8s} | "
          f"{'well plain':>10s} {'well ridge':>10s} {'HMC':>5s} | {'w0² plain':>9s} {'w0² ridge':>9s} | top bulk families (k,j: share of Σ)")
    for lab, c, r_meas, well in cases:
        T, beta, lam, Chat = reduced_params(c["N"], c["d"], 0.5, c["sa0"], c["P"], c["kappa"])
        K = Kernel(T, beta)
        D0, _ = solve_loop(Chat, lam, eps, K)
        _, p0, _, _, _ = K.stats(D0)
        o0 = analyse(D0, lam, eps, K)
        s = solve_with_ridge(c["N"], c["d"], 0.5, c["sa0"], c["P"], c["kappa"], eps)
        terms = [(k, j, n * eta / (c["P"] * eta + s["kt"])) for (k, j, n, eta) in s["fams"] if not (j == k and k in (1, 3))]
        tot = sum(t for _, _, t in terms)
        top = sorted(terms, key=lambda t: -t[2])[:3]
        print(f"{lab:>14s} | {s['r']:>7.4f} {r_meas if r_meas else float('nan'):>6.3f} | {o0['L3']:>8.4f} {s['L3']:>8.4f} | "
              f"{outer_well(D0, T, beta):>10.3f} {outer_well(s['D'], T, beta):>10.3f} {well if well else float('nan'):>5.2f} | "
              f"{p0 @ K.w ** 2:>9.5f} {s['w0_sq']:>9.5f} | " + ", ".join(f"({k},{j}):{t / tot:.2f}" for k, j, t in top))


if __name__ == "__main__":
    main()
