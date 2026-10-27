"""
Read-in theory with rotation-invariant (radial) perp modes, extending readin_action_theory_q.py.

Averaging a neuron's activation over the direction of its perp weight (|w_perp|² = q, isotropic) gives
    phibar(x0, rho) = erf(w0 x0 / sqrt(1 + 2 q rho)),      rho = |x_perp|²/(d-1) ~ chi²_{d-1}/(d-1),
which still depends on x_perp through rho. Its coefficients on the orthonormal modes He_k(x0) psi_m(rho)
(psi_m orthonormal polynomials for the law of rho) are
    c_{k,m}(w0, q) = E_rho[ e_k(w0 / sqrt(1 + 2 q rho)) psi_m(rho) ],
    e_k(a) = normalized He_k coefficient of erf(a x0) = (2/sqrt(pi)) (-1)^((k-1)/2) (k-1)!!/sqrt(k!) a^k/(1+2a²)^{k/2}.
m = 0 reproduces the He_k(x0) features of readin_action_theory_q up to O(1/d); m >= 1 are the radial
modes (e.g. x0 (rho - 1)): absent from the target, present in every neuron, so they enter J and the
residual like He5/He7. Single-neuron action, loop and convexity are as in readin_action_theory_q with
the feature vector c_{k,m}; target y = 1 on (1,0), eps on (3,0), 0 elsewhere.
"""

from math import factorial, sqrt

import numpy as np
from scipy.special import roots_genlaguerre

from readin_action_theory import reduced_params


def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def e_k(a, k):
    return (2 / np.sqrt(np.pi)) * (-1) ** ((k - 1) // 2) * _dfact(k - 1) / sqrt(factorial(k)) * a ** k / (1 + 2 * a * a) ** (k / 2)


def rho_quadrature(d, n=64, m_max=2):
    """Nodes/weights for rho = chi²_{d-1}/(d-1) and orthonormal polynomials psi_m(rho) at the nodes."""
    alpha = (d - 1) / 2 - 1                       # Gamma(shape=(d-1)/2) = generalized Laguerre with alpha
    x, w = roots_genlaguerre(n, alpha)
    w = w / w.sum()
    rho = x * 2 / (d - 1)
    # orthonormal polynomials by Gram-Schmidt on the discrete measure
    V = np.stack([rho ** m for m in range(m_max + 1)], 1)
    Q = np.zeros_like(V)
    for m in range(m_max + 1):
        v = V[:, m].copy()
        for j in range(m):
            v -= (w * v * Q[:, j]).sum() * Q[:, j]
        Q[:, m] = v / np.sqrt((w * v * v).sum())
    return rho, w, Q


class KernelRadial:
    def __init__(self, d, s0, ks=(1, 3, 5, 7), ms=(0, 1, 2), wmax=6.0, nw=1201, nq=240, a=0.5, nrho=64):
        self.d, self.s0 = d, s0
        self.beta = d / (2 * s0)
        self.modes = [(k, m) for m in ms for k in ks]
        u = np.linspace(0, np.arcsinh(wmax / a), nw)
        self.w = a * np.sinh(u)
        dw = np.diff(self.w); tw = np.zeros(nw); tw[:-1] += dw / 2; tw[1:] += dw / 2
        q0 = s0 * (d - 1) / d
        self.q = np.geomspace(q0 * 1e-3, q0 * 4, nq)
        dq = np.diff(self.q); tq = np.zeros(nq); tq[:-1] += dq / 2; tq[1:] += dq / 2
        self.tw = tw
        W, Q = np.meshgrid(self.w, self.q, indexing="ij")
        rho, wr, psi = rho_quadrature(d, nrho, max(ms))
        A = W[..., None] / np.sqrt(1 + 2 * Q[..., None] * rho[None, None, :])       # (nw, nq, nrho)
        self.C = np.stack([(e_k(A, k) * (wr * psi[:, m])[None, None, :]).sum(-1) for (k, m) in self.modes])
        self.base = self.beta * (W * W + Q) - 0.5 * (d - 3) * np.log(Q)
        self.logm = np.log(np.outer(tw, tq))
        self.W, self.Q = W, Q

    def stats(self, D):
        g = np.tensordot(D, self.C, 1)
        L = -(self.base - self.beta * g * g) + self.logm
        m = L.max()
        E = np.exp(L - m)
        Z = E.sum()
        p = E / Z
        C = self.C.reshape(len(self.modes), -1)
        pr = p.ravel(); gr = g.ravel()
        J = (C * pr) @ C.T
        gc = (C * (pr * gr)).sum(1)
        Cov = (C * (pr * gr * gr)) @ C.T - np.outer(gc, gc)
        return p, m + np.log(Z), J, Cov


def target(modes, eps):
    y = np.zeros(len(modes))
    y[modes.index((1, 0))] = 1.0
    y[modes.index((3, 0))] = eps
    return y


def solve(N, d, s0, sa0, P, kappa, eps, chi=1.0, tol=1e-9, maxit=200, **kw):
    _, _, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
    K = KernelRadial(d, s0, **kw)
    y = target(K.modes, eps)
    z = np.sqrt(Chat) * y
    D = z.copy()
    phi = lambda D: 0.5 * D @ D + K.stats(D)[1] / (2 * K.beta * lam) - z @ D
    for _ in range(maxit):
        _, _, J, Cov = K.stats(D)
        grad = D + J @ D / lam - z
        if np.abs(grad).max() < tol * max(1, np.abs(z).max()):
            break
        H = np.eye(len(D)) + (J + 2 * K.beta * Cov) / lam
        step = -np.linalg.solve(H, grad)
        t, f0 = 1.0, phi(D)
        while phi(D + t * step) > f0 + 1e-4 * t * grad @ step and t > 1e-12:
            t *= 0.5
        D = D + t * step
    p, _, J, _ = K.stats(D)
    fbar = J @ np.linalg.solve(J + lam * np.eye(len(D)), y)
    pw = p.sum(1)
    thr = 2.5 * np.sqrt(s0 / d)
    dens = pw / K.tw / 2
    nlp = -np.log(np.maximum(dens, 1e-300))
    i = np.where(K.w > 0.4)[0]
    loc = [j for j in i[1:-1] if nlp[j] < nlp[j - 1] and nlp[j] <= nlp[j + 1]]
    q0 = s0 * (d - 1) / d
    al = K.w > thr
    qg = (p * K.Q).sum(1) / np.maximum(pw, 1e-300)
    return dict(D=D, res=np.abs(D + J @ D / lam - z).max(), modes=K.modes, L1=fbar[K.modes.index((1, 0))],
                L3=fbar[K.modes.index((3, 0))] / eps, w0_sq=pw @ K.w ** 2, frac_aligned=pw[al].sum(),
                outer_well=K.w[loc[0]] if loc else np.nan, q_aligned=(pw[al] @ qg[al]) / pw[al].sum() / q0,
                residual=y - fbar, w=K.w, nlp=nlp)
