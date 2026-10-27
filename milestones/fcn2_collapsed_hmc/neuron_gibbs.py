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

M⁻¹ is kept in float64 and updated by a rank-2 formula after each change,
then rebuilt from scratch at the start of every sweep.
"""

import math

import torch


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
