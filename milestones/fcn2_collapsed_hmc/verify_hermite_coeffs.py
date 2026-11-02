#!/usr/bin/env python3
"""
Check every copy of the erf Hermite coefficient against direct 2D Gauss-Hermite quadrature:
    c_k(w0, q) = E[He_k(x0) erf(w0 x0 + sqrt(q) z)]  (normalized He_k, x0, z ~ N(0,1)),
    closed form (2/sqrt(pi)) (-2)^m (k-2)!!/sqrt(k!) w0^k / (1 + 2 w0² + 2q)^{k/2},   k = 2m+1.
The copies in readin_action_theory_q, readin_action_theory_radial, effective_ridge_theory, analyze_large_d
and diagnostics/ray_trend_exact once used (-1)^m (k-1)!!, which is right only for k = 1, 3.

    python verify_hermite_coeffs.py
"""

import ast
from math import factorial, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermegauss
from scipy.special import erf

from analyze_large_d import a_k as a_k_large_d
from effective_ridge_theory import a_k as a_k_ridge
from readin_action_theory_q import he_coeff
from readin_action_theory_radial import e_k

HERE = Path(__file__).resolve().parent


def quad_coeff(w0, q, k, n=200):
    x, wt = hermegauss(n)
    wt = wt / wt.sum()
    X0, Z = np.meshgrid(x, x, indexing="ij")
    hk = hermeval(X0, np.eye(k + 1)[k]) / sqrt(factorial(k))
    return float((np.outer(wt, wt) * hk * erf(w0 * X0 + np.sqrt(q) * Z)).sum())


def ray_trend_a_k():
    """a_k from diagnostics/ray_trend_exact.py (that module runs on import, so load just the two defs)."""
    src = (HERE / "diagnostics" / "ray_trend_exact.py").read_text()
    defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("dfact", "a_k")]
    ns = {"np": np, "sqrt": sqrt, "factorial": factorial}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "ray_trend_exact", "exec"), ns)
    return ns["a_k"]


def main():
    a_k_ray = ray_trend_a_k()
    worst = 0.0
    print(f"{'w0':>5s} {'q':>5s} {'k':>3s} | {'quadrature':>11s} | max |copy - quad| over the 5 copies")
    for w0, q in [(0.3, 0.5), (1.1, 0.5), (1.1, 0.17), (2.0, 0.05)]:
        s = np.sqrt(w0 * w0 + q)                   # |w|; the a_k copies take |w| and need ŵ0^k = (w0/s)^k
        for k in (1, 3, 5, 7, 9, 11):
            ref = quad_coeff(w0, q, k)
            copies = [he_coeff(w0, q, k), e_k(w0 / np.sqrt(1 + 2 * q), k),
                      a_k_large_d(s, k) * (w0 / s) ** k, a_k_ridge(s, k) * (w0 / s) ** k, a_k_ray(s, k) * (w0 / s) ** k]
            err = max(abs(float(c) - ref) for c in copies)
            worst = max(worst, err / max(abs(ref), 1e-12))
            print(f"{w0:>5.2f} {q:>5.2f} {k:>3d} | {ref:+11.6f} | {err:.1e}")
    print(f"\nworst relative error: {worst:.1e}  ->  {'OK' if worst < 1e-8 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
