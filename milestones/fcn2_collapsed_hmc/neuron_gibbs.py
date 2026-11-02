"""
Exact per-neuron Gibbs moves on the collapsed FCN2 posterior.

With the readout integrated out, the conditional of neuron i given all others is
    U_i(w) = |w|²/(2σ_W²) + ½ log s(w) - ½ (β(w) - c(w)ᵀ z)² / s(w)
in the N×N form, where M = (2/T)ΦᵀΦ + I/σ_A², b = (2/T)Φᵀy and, for the candidate
column φ = erf(X w):
    m = (2/T)|φ|² + 1/σ_A²,  c = (2/T) Φ_{-i}ᵀ φ,  β = (2/T) φᵀy,
    s = m - cᵀ M_{-i}⁻¹ c,   z = M_{-i}⁻¹ b_{-i}
(Schur complement on row/column i). This is the mean-field single-neuron potential,
evaluated exactly at finite N.

Each neuron is updated with an iterated-SIR kernel: draw B candidates from a proposal q
that does not depend on w_i, add the current w_i, and pick one with probability
∝ exp(-U_i) / q. That kernel leaves p(w_i | rest) invariant for any such q, so a broad
proposal on the teacher coordinate w0 lets neurons jump between the unaligned
(w0≈0) and aligned (|w0|≈μ) wells in a single move.

Modes:
    w0    resample w0 only, keeping w_perp (1-d proposal N(0, ω²))
    full  resample the whole neuron: w0 ~ N(0, ω²), w_perp ~ N(0, s_perp² I)
    w0q   resample w0 and the perp norm q = |w_perp|² together, keeping the direction of w_perp
          (w_perp -> sqrt(q'/q) w_perp). Proposal: JointProposal (theory p(w0, q) mixed with a broad
          component), set as self.jprop. In polar coordinates the conditional of (w0, q) given the direction
          carries the volume factor q^{(d-3)/2}, so a candidate's weight is exp(-U_i) q^{(d-3)/2} / q(w0, q).
          Needed at large d: switching wells means moving w0 and q together (aligned neurons have
          q ≈ 0.35-0.55 of the prior), which the w0 move and HMC cannot do on reasonable time scales.

M⁻¹ is kept in float64 and updated by a rank-2 formula after each change,
then rebuilt from scratch at the start of every sweep.
"""

import math

import numpy as np
import torch


class JointProposal:
    """
    Proposal density on (w0, q) for the w0q move: a mixture of
      theory (weight 1 - broad): the read-in theory's p(w0, q) (readin_action_theory_q, per-neuron perp norm),
          piecewise constant on the cells of its (|w0|, q) grid, with a random sign for w0;
      broad (weight broad): w0 ~ N(0, ω²), q log-uniform on [q_lo, q_hi] · q_prior.
    Any fixed density gives an exact move; the theory component makes jumps between wells likely, the broad one
    keeps the move working where the theory is off.
    """

    def __init__(self, w_nodes, q_nodes, f_nodes, q_prior, broad=0.3, omega=0.6, q_lo=0.05, q_hi=2.0):
        self.w = np.asarray(w_nodes, float)            # |w0| grid (ascending, starts at 0)
        self.q = np.asarray(q_nodes, float)            # q grid (ascending)
        fc = 0.25 * (f_nodes[:-1, :-1] + f_nodes[1:, :-1] + f_nodes[:-1, 1:] + f_nodes[1:, 1:])   # cell densities
        area = np.outer(np.diff(self.w), np.diff(self.q))
        mass = fc * area
        self.cell_p = (mass / mass.sum()).ravel()
        self.cdf = np.cumsum(self.cell_p)              # cells drawn by binary search (rng.choice is O(cells) per call)
        self.cell_f = fc / mass.sum()                  # normalized density of |w0|, q on each cell
        self.broad, self.omega = broad, omega
        self.qa, self.qb = q_lo * q_prior, q_hi * q_prior

    @classmethod
    def from_theory(cls, d, N, P, kappa, s0, sa0, eps, chi=1.0, ks=(1, 3, 5, 7, 9, 11), **kw):
        from readin_action_theory import reduced_params
        from readin_action_theory_q import KernelQ, solve_loop_q
        _, _, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
        K = KernelQ(d, s0, ks=ks, nq=int(500 * max(1, math.sqrt(d / 600))))
        D, _ = solve_loop_q(Chat, lam, eps, K)
        p = K.stats(D)[0]
        f = p / np.outer(K.tw, K.tq)                   # density per unit |w0| and q at the nodes
        return cls(K.w, K.q, f, s0 * (d - 1) / d, **kw)

    def sample(self, n, rng):
        """n draws (w0, q) and their log proposal density."""
        nb = rng.binomial(n, self.broad)
        nt = n - nb
        cells = np.minimum(np.searchsorted(self.cdf, rng.random(nt) * self.cdf[-1]), len(self.cell_p) - 1)
        iw, iq = np.unravel_index(cells, (len(self.w) - 1, len(self.q) - 1))
        w_t = self.w[iw] + rng.random(nt) * (self.w[iw + 1] - self.w[iw])
        q_t = self.q[iq] + rng.random(nt) * (self.q[iq + 1] - self.q[iq])
        w_t *= rng.choice([-1.0, 1.0], nt)
        w_b = rng.normal(0, self.omega, nb)
        q_b = np.exp(rng.uniform(np.log(self.qa), np.log(self.qb), nb))
        w0, q = np.concatenate([w_t, w_b]), np.concatenate([q_t, q_b])
        return w0, q, self.logpdf(w0, q)

    def logpdf(self, w0, q):
        w0, q = np.atleast_1d(w0), np.atleast_1d(q)
        a = np.abs(w0)
        iw = np.searchsorted(self.w, a, side="right") - 1
        iq = np.searchsorted(self.q, q, side="right") - 1
        inside = (iw >= 0) & (iw < len(self.w) - 1) & (iq >= 0) & (iq < len(self.q) - 1)
        ft = np.zeros_like(a)
        ft[inside] = 0.5 * self.cell_f[iw[inside], iq[inside]]          # factor 1/2: random sign of w0
        inb = (q >= self.qa) & (q <= self.qb)
        fb = np.where(inb, np.exp(-0.5 * (w0 / self.omega) ** 2) / (self.omega * math.sqrt(2 * math.pi))
                      / (q * math.log(self.qb / self.qa)), 0.0)
        return np.log((1 - self.broad) * ft + self.broad * fb)


class NeuronGibbs:
    def __init__(self, X, Y, N, T, s0, sa0, chi=1.0, omega=0.6, s_perp=None):
        self.X = X
        self.Y = Y
        self.P, self.d = X.shape
        self.N = N
        self.T = T / chi
        self.sw2 = s0 / self.d
        self.sa2 = sa0 / (N * chi)
        self.omega = omega
        self.s_perp = math.sqrt(self.sw2) if s_perp is None else s_perp
        self.g = 2.0 / self.T
        self.jprop = None                              # JointProposal for mode 'w0q'
        self.rng = np.random.default_rng(int(torch.initial_seed()) % 2 ** 32)

    def _rebuild(self, W):
        """W: (N, d). Builds Φ (P,N), M⁻¹ (N,N, float64), b, z_full."""
        Phi = torch.erf(self.X @ W.T)
        G = (Phi.T @ Phi).double()
        M = self.g * G + torch.eye(self.N, dtype=torch.float64, device=W.device) / self.sa2
        L = torch.linalg.cholesky(M)
        Minv = torch.cholesky_inverse(L)
        b = self.g * (Phi.T @ self.Y).double()
        return Phi, Minv, b, Minv @ b

    def _cond_energy(self, i, Fc, prior, Phi, Minv, zf):
        """U_i (up to a constant) for candidate columns Fc (P, K) of neuron i."""
        g = self.g
        Minv_ii = Minv[i, i]
        col = Minv[:, i].clone()
        col[i] = 0.0
        zp = zf - col * (zf[i] / Minv_ii)
        zp[i] = 0.0
        C = g * (Phi.T @ Fc).double()  # (N, K)
        C[i, :] = 0.0
        MC = Minv @ C
        colC = col @ C
        quad = (C * MC).sum(0) - colC ** 2 / Minv_ii
        m = g * (Fc * Fc).sum(0).double() + 1.0 / self.sa2
        s = m - quad
        beta = g * (Fc.T @ self.Y).double()
        r = beta - zp @ C
        Ui = prior.double() + 0.5 * torch.log(s) - 0.5 * r ** 2 / s
        return Ui, {"col": col, "Minv_ii": Minv_ii, "MC": MC, "colC": colC, "s": s, "beta": beta}

    @torch.no_grad()
    def sweep(self, W, B=64, mode="w0", order=None, check=False):
        """One Gibbs sweep over all neurons of a single chain. W: (N, d), modified copy returned."""
        W = W.clone()
        dev = W.device
        Phi, Minv, b, zf = self._rebuild(W)
        X0 = self.X[:, 0]
        Xp = self.X[:, 1:]
        if order is None:
            order = torch.randperm(self.N, device=dev).tolist()
        n_changed = 0
        n_crossed = 0
        thr = 2.5 * math.sqrt(self.sw2)
        for i in order:
            wi = W[i]
            # Candidates (index 0 is the current neuron)
            if mode == "w0":
                w0c = torch.randn(B, device=dev, dtype=W.dtype) * self.omega
                w0all = torch.cat([wi[:1], w0c])
                base = Xp @ wi[1:]  # (P,)
                Fc = torch.erf(X0[:, None] * w0all[None, :] + base[:, None])  # (P, B+1)
                prior = 0.5 * w0all ** 2 / self.sw2
                logq = -0.5 * w0all ** 2 / self.omega ** 2
                Wc = None
            elif mode == "w0q":
                qi = float((wi[1:] ** 2).sum())
                w0n, qn, lqn = self.jprop.sample(B, self.rng)
                w0all = torch.tensor(np.concatenate([[float(wi[0])], w0n]), device=dev, dtype=W.dtype)
                qall = torch.tensor(np.concatenate([[qi], qn]), device=dev, dtype=W.dtype)
                lq = np.concatenate([self.jprop.logpdf(float(wi[0]), qi), lqn])
                scale = torch.sqrt(qall / qi)
                base = Xp @ wi[1:]
                Fc = torch.erf(X0[:, None] * w0all[None, :] + base[:, None] * scale[None, :])
                # prior |w|²/(2σ_W²) minus the polar volume ((d-3)/2) log q, folded into the energy
                prior = 0.5 * (w0all ** 2 + qall) / self.sw2 - 0.5 * (self.d - 3) * torch.log(qall)
                logq = torch.tensor(lq, device=dev, dtype=torch.float64)
                Wc = None
            else:
                wc = torch.randn(B, self.d, device=dev, dtype=W.dtype)
                wc[:, 0] *= self.omega
                wc[:, 1:] *= self.s_perp
                Wc = torch.cat([wi[None, :], wc])  # (B+1, d)
                Fc = torch.erf(self.X @ Wc.T)
                prior = 0.5 * (Wc ** 2).sum(-1) / self.sw2
                logq = -0.5 * Wc[:, 0] ** 2 / self.omega ** 2 - 0.5 * (Wc[:, 1:] ** 2).sum(-1) / self.s_perp ** 2

            Ui, aux = self._cond_energy(i, Fc, prior, Phi, Minv, zf)
            logw = -Ui - logq.double()
            j = int(torch.multinomial(torch.softmax(logw - logw.max(), 0), 1).item())
            if j == 0:
                continue

            # Accept candidate j: rank-2 update of M⁻¹ and bookkeeping
            n_changed += 1
            w_new = W[i].clone()
            if mode == "w0":
                w_new[0] = w0all[j]
            elif mode == "w0q":
                w_new[0] = w0all[j]
                w_new[1:] = wi[1:] * scale[j]
            else:
                w_new = Wc[j].clone()
            if (wi[0].abs() > thr) != (w_new[0].abs() > thr):
                n_crossed += 1
            col, Minv_ii, MC, colC, s, beta = (aux[k] for k in ("col", "Minv_ii", "MC", "colC", "s", "beta"))
            sj = s[j]
            u = MC[:, j] - col * (colC[j] / Minv_ii)  # M_{-i}⁻¹ c on the -i block
            u[i] = 0.0
            Minv = Minv - torch.outer(col, col) / Minv_ii + torch.outer(u, u) / sj
            Minv[:, i] = -u / sj
            Minv[i, :] = -u / sj
            Minv[i, i] = 1.0 / sj
            W[i] = w_new
            Phi[:, i] = Fc[:, j]
            b[i] = beta[j]
            zf = Minv @ b
        stats = {"changed": n_changed / self.N, "crossed": n_crossed / self.N}
        if check:
            _, Minv_ref, _, zf_ref = self._rebuild(W)
            stats["minv_rel_err"] = ((Minv - Minv_ref).norm() / Minv_ref.norm()).item()
            stats["z_rel_err"] = ((zf - zf_ref).norm() / zf_ref.norm()).item()
        return W, stats
