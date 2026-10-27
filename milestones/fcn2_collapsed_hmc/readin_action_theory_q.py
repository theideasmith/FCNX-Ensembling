"""
Read-in mean-field theory with the perp norm as a per-neuron variable (extends readin_action_theory.py,
which freezes |w_perp|² at its prior value s0 (d-1)/d for every neuron).

A neuron w = (w0, w_perp) enters the He1/He3(x0) block only through w0 and q = |w_perp|², via
T = 1 + 2q in the features
    c1(w0, q) = (2/sqrt(pi)) w0 / sqrt(T + 2 w0²),   c3(w0, q) = -sqrt(8/(3 pi)) w0³ / (T + 2 w0²)^{3/2}.
With the perp-shell volume q^{(d-3)/2} dq, the single-neuron action is
    S(w0, q) = beta [w0² + q - (D·c(w0, q))²] - ((d-3)/2) log q,     beta = d / (2 sigma_w²),
and the neuron distribution is p(w0, q) ∝ exp(-S) (two-dimensional quadrature, no Laplace step).
With D = 0, <q> = sigma_w² (d-1)/d, the frozen value. The loop is unchanged in form:
    (I + J(D)/lam) D = sqrt(Chat) (1, eps),   J = <c cᵀ> over p(w0, q),
the gradient condition of the convex Phi(D) = |D|²/2 + log Z/(2 beta lam) - sqrt(Chat) y·D (Newton).
"""

import numpy as np

from readin_action_theory import reduced_params


def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def he_coeff(W, Q, k):
    """Normalized He_k(x0) coefficient of erf(w0 x0 + w_perp·x_perp), |w_perp|² = Q (odd k)."""
    from math import factorial, sqrt
    T = 1 + 2 * Q
    return ((2 / np.sqrt(np.pi)) * (-1) ** ((k - 1) // 2) * _dfact(k - 1) / sqrt(factorial(k))
            * W ** k / (T + 2 * W * W) ** (k / 2))


class KernelQ:
    def __init__(self, d, s0, wmax=6.0, nw=2401, nq=500, a=0.5, ks=(1, 3)):
        self.d, self.s0, self.ks = d, s0, ks
        self.beta = d / (2 * s0)
        u = np.linspace(0, np.arcsinh(wmax / a), nw)
        self.w = a * np.sinh(u)
        dw = np.diff(self.w); tw = np.zeros(nw); tw[:-1] += dw / 2; tw[1:] += dw / 2
        q0 = s0 * (d - 1) / d
        self.q = np.geomspace(q0 * 1e-3, q0 * 4, nq)
        dq = np.diff(self.q); tq = np.zeros(nq); tq[:-1] += dq / 2; tq[1:] += dq / 2
        self.tw, self.tq = tw, tq
        W, Q = np.meshgrid(self.w, self.q, indexing="ij")
        self.C = np.stack([he_coeff(W, Q, k) for k in ks])          # (len(ks), nw, nq)
        self.c1, self.c3 = self.C[0], self.C[1]
        self.base = self.beta * (W * W + Q) - 0.5 * (d - 3) * np.log(Q)
        self.logm = np.log(np.outer(tw, tq))
        self.W, self.Q = W, Q

    def stats(self, D):
        g = np.tensordot(D, self.C, 1)
        S = self.base - self.beta * g * g
        L = -S + self.logm
        m = L.max()
        E = np.exp(L - m)
        Z = E.sum()
        p = E / Z
        logZ = m + np.log(Z)
        C = self.C.reshape(len(self.ks), -1)
        pr = p.ravel(); gr = g.ravel()
        J = (C * pr) @ C.T
        gc = (C * (pr * gr)).sum(1)
        Cov = (C * (pr * gr * gr)) @ C.T - np.outer(gc, gc)
        return p, logZ, J, Cov


def solve_loop_q(Chat, lam, eps, K, D0=None, tol=1e-10, maxit=200):
    y = np.zeros(len(K.ks)); y[0], y[1] = 1.0, eps
    z = np.sqrt(Chat) * y
    D = z.copy() if D0 is None else np.array(D0, float)
    phi = lambda D: 0.5 * D @ D + K.stats(D)[1] / (2 * K.beta * lam) - z @ D
    for _ in range(maxit):
        _, _, J, Cov = K.stats(D)
        grad = D + J @ D / lam - z
        if np.abs(grad).max() < tol:
            break
        H = np.eye(len(K.ks)) + (J + 2 * K.beta * Cov) / lam
        step = -np.linalg.solve(H, grad)
        t, f0 = 1.0, phi(D)
        while phi(D + t * step) > f0 + 1e-4 * t * grad @ step and t > 1e-12:
            t *= 0.5
        D = D + t * step
    return D, np.abs(grad).max()


def analyse_q(N, d, s0, sa0, P, kappa, eps, chi=1.0, **grid):
    """ks=(1,3) is the He1-He3 theory; ks=(1,3,5,7) adds He5/He7(x0) (target y = (1, eps, 0, 0))."""
    """Solve the q-relaxed theory; return predictions comparable with HMC observables."""
    _, _, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
    K = KernelQ(d, s0, **grid)
    D, res = solve_loop_q(Chat, lam, eps, K)
    p, logZ, J, _ = K.stats(D)
    y = np.zeros(len(K.ks)); y[0], y[1] = 1.0, eps
    fbar = J @ np.linalg.solve(J + lam * np.eye(len(K.ks)), y)
    pw = p.sum(1)                                   # marginal over q, on w0 ≥ 0 (density weights)
    q_given_w = (p * K.Q).sum(1) / np.maximum(pw, 1e-300)
    thr = 2.5 * np.sqrt(s0 / d)
    al = K.w > thr
    q0 = s0 * (d - 1) / d
    # -log p(w0) normalized on the full line (density = mass / cell width, symmetric in w0)
    dens = pw / K.tw / 2
    nlp = -np.log(np.maximum(dens, 1e-300))
    i = np.where(K.w > 0.4)[0]
    loc = [j for j in i[1:-1] if nlp[j] < nlp[j - 1] and nlp[j] <= nlp[j + 1]]
    return dict(D=D, res=res, L1=fbar[0], L3=fbar[1] / eps, w0_sq=pw @ K.w ** 2, frac_aligned=pw[al].sum(),
                q_aligned_over_prior=(pw[al] @ q_given_w[al]) / pw[al].sum() / q0 if pw[al].sum() > 0 else np.nan,
                q_all_over_prior=(p * K.Q).sum() / q0, outer_well=K.w[loc[0]] if loc else np.nan,
                w=K.w, nlp=nlp, q_given_w=q_given_w / q0, lam=lam, Chat=Chat)
