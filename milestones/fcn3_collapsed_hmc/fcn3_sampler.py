"""
Exact posterior sampler for the three-layer erf network (FCN3) of the notes "FCN3 in Hermite matrix elements":

    f(x) = sum_k a_k erf(h_k(x)),   h_k(x) = sum_j B_kj erf(w_j . x),
    -log p = N2 chi/(2 sA^2)|a|^2 + N1/(2 sB^2)|B|^2 + d/(2 sw^2)|W|^2 + chi/(2 kappa) sum_mu (f(x_mu) - y_mu)^2.

One iteration (every step leaves the joint posterior invariant; this is a partially collapsed Gibbs sampler):
  1. HMC on (W, B) with the readout a integrated out exactly (Woodbury, N2 x N2 in float64). The potential and its
     gradient are evaluated in chunks over the training set with a hand-written backward pass, so memory is
     O(chunk * N) and P can be large.
  2. a ~ p(a | W, B), exact Gaussian draw.
  3. (every --sweep-every iterations) exact per-neuron jump moves given a:
     - middle neuron k: (B_k, a_k) jointly. a_k is integrated out of the move in closed form (1-D Schur
       complement) and redrawn after it. B_k is parametrised as B_k = C (C^T C)^-1 h_S + r n_perp: h_S = C^T B_k are its
       Hermite components on S (C_jn = c_n(w_j), exact population coefficients of the read-in neurons), r and n_perp
       its remaining norm and direction. Candidates propose (h_S, log r^2), and n_perp either kept or redrawn.
     - read-in neuron j: (w_j, B[:, j]) jointly. w_j = (w0, sqrt(q) n) with n in S^{d-2}. Candidates propose
       (w0, log q), and n either kept or redrawn uniformly; the direction redraw lets the perp directions adapt in
       the jump as well as in HMC. The outgoing column B[:, j] is proposed from N(m(w), sB^2/N1 I), where m(w) is a
       linearised conditional mean computed only from the rest of the state (a few CG steps). Without that, a
       jump to an aligned well would be evaluated against outgoing weights fitted to the old feature and rejected.
     Each move is iterated SIR (multiple-try independence sampler) with the current state as candidate 0. The
     proposal depends only on the rest of the state: a mixture of a kernel density estimate over the OTHER
     neurons of the same layer (they are exchangeable, so their population estimates the single-neuron law and
     carries jumps to wells other neurons have found), the prior, and a broad component that discovers new wells.
  4. Exact sign symmetries: (w_j, B[:, j]) -> -(w_j, B[:, j]) and B_k -> -B_k (a_k -> -a_k).
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fcn2_collapsed_hmc"))
from collapsed_hmc_fcn2 import make_data, data_fingerprint  # noqa: E402

SQPI = 2.0 / math.sqrt(math.pi)
F64 = torch.float64


def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def cn_const(n):
    m = (n - 1) // 2
    return (2 / math.sqrt(math.pi)) * (-2) ** m * _dfact(n - 2) / math.sqrt(math.factorial(n))


def readin_coeffs(W, ks):
    """C_jn = <phi_n | erf(w_j . x)> for the normalised He_n(x0), n in ks. W: (N1, d). Returns (N1, |S|)."""
    w0 = W[:, 0]
    T = 1 + 2 * (W * W).sum(1)
    return torch.stack([cn_const(n) * w0 ** n / T ** (n / 2) for n in ks], 1)


def he_norm(x, nmax):
    H = [torch.ones_like(x), x]
    for n in range(1, nmax):
        H.append(x * H[n] - n * H[n - 1])
    return [H[n] / math.sqrt(math.factorial(n)) for n in range(nmax + 1)]


# ----------------------------------------------------------------------------------------------------------
# population proposal
# ----------------------------------------------------------------------------------------------------------
class PopProposal:
    """Mixture density on R^m for one unit, given the other units' coordinates `pop` (n, m):
         w_kde * KDE(pop, bandwidth bw, symmetrised under the sign flip of the `sym` coordinates)
       + w_prior * N(mu0, sd0^2) + w_broad * N(mu0, sdb^2)  (diagonal Gaussians)."""

    def __init__(self, bw, mu0, sd0, sdb, sym, weights, device):
        t = lambda v: torch.as_tensor(v, dtype=F64, device=device)
        self.bw, self.mu0, self.sd0, self.sdb = t(bw), t(mu0), t(sd0), t(sdb)
        self.sym = torch.as_tensor(sym, dtype=torch.bool, device=device)
        w = np.asarray(weights, float)
        self.w = w / w.sum()
        self.w_t = torch.as_tensor(self.w, device=device)
        self.m = len(bw)

    def _signs(self, n, gen):
        s = torch.ones(n, self.m, dtype=F64, device=self.bw.device)
        flip = torch.randint(0, 2, (n,), device=self.bw.device, generator=gen).bool()
        s[flip] = torch.where(self.sym, -1.0, 1.0).to(F64)
        return s

    def sample(self, pop, n, gen):
        dev = self.bw.device
        comp = torch.multinomial(torch.as_tensor(self.w, device=dev), n, replacement=True, generator=gen)
        z = torch.randn(n, self.m, dtype=F64, device=dev, generator=gen)
        idx = torch.randint(0, pop.shape[0], (n,), device=dev, generator=gen)
        kde = self._signs(n, gen) * pop[idx].to(F64) + self.bw * z
        pri = self.mu0 + self.sd0 * z
        brd = self.mu0 + self.sdb * z
        c = comp[:, None]
        return torch.where(c == 0, kde, torch.where(c == 1, pri, brd))

    def sample_ex(self, pop, j, n, gen):
        """sample() from the population without row j (no copy)."""
        idx = torch.randint(0, pop.shape[0] - 1, (n,), device=pop.device, generator=gen)
        idx = idx + (idx >= j).long()
        return self._sample_idx(pop, idx, n, gen)

    def _sample_idx(self, pop, idx, n, gen):
        dev = self.bw.device
        comp = torch.multinomial(self.w_t, n, replacement=True, generator=gen)
        z = torch.randn(n, self.m, dtype=F64, device=dev, generator=gen)
        kde = self._signs(n, gen) * pop[idx].to(F64) + self.bw * z
        c = comp[:, None]
        return torch.where(c == 0, kde, torch.where(c == 1, self.mu0 + self.sd0 * z, self.mu0 + self.sdb * z))

    def logpdf_ex(self, X, pop, j):
        """logpdf() with row j of the population excluded."""
        return self.logpdf(X, pop, exclude=j)

    def logpdf(self, X, pop, exclude=None):
        pop = pop.to(F64)
        lg = lambda X, mu, sd: (-0.5 * ((X - mu) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)).sum(-1)
        flipv = torch.where(self.sym, -1.0, 1.0).to(F64)
        k1 = lg(X[:, None, :], pop[None], self.bw)
        k2 = lg(X[:, None, :], pop[None] * flipv, self.bw)
        npop = pop.shape[0]
        if exclude is not None:
            k1[:, exclude] = -torch.inf
            k2[:, exclude] = -torch.inf
            npop -= 1
        lkde = torch.logsumexp(torch.cat([k1, k2], 1), 1) - math.log(2 * npop)
        terms = torch.stack([lkde + math.log(self.w[0] + 1e-300), lg(X, self.mu0, self.sd0) + math.log(self.w[1] + 1e-300),
                             lg(X, self.mu0, self.sdb) + math.log(self.w[2] + 1e-300)])
        return torch.logsumexp(terms, 0)


# ----------------------------------------------------------------------------------------------------------
# the sampler
# ----------------------------------------------------------------------------------------------------------
class FCN3:
    def __init__(self, X, Y, N1, N2, chi, kappa, sigA=1.0, sigB=1.0, sigw=1.0, ks=(1, 3), chunk=None,
                 n_cand=16, cg_iters=4, redraw_frac=0.25, prop_weights=(0.5, 0.25, 0.25), seed=0, energy="direct"):
        self.X, self.Y = X, Y
        self.P, self.d = X.shape
        self.N1, self.N2 = N1, N2
        self.dt = X.dtype
        self.dev = X.device
        self.v = kappa / chi                                   # noise variance: chi/(2 kappa) (f-y)^2
        self.sa2 = sigA ** 2 / (N2 * chi)
        self.sw = sigw / math.sqrt(self.d)
        self.sb = sigB / math.sqrt(N1)
        self.ks = ks
        self.chunk = chunk or max(256, int(2 ** 26 // max(N1, N2, self.d)))
        self.yy = float((Y.double() ** 2).sum())
        self.n_cand, self.cg_iters, self.redraw_frac = n_cand, cg_iters, redraw_frac
        # read-in candidate energies: "direct" (one P x N2 erf per candidate; faster up to ~40 candidates) or
        # "series" (Taylor series in phi*b with a rigorous remainder bound; faster for many candidates)
        self.energy = energy
        # max_x |erf^(n)(x)| for the series remainder bound, and the f tolerance (energy error <= ~1e-6)
        xs = np.linspace(-12, 12, 48001)
        Hm2, Hm1, Cn = None, np.ones_like(xs), [1.0]
        for n_ in range(1, 26):
            Cn.append(float(np.abs(SQPI * np.exp(-xs ** 2) * Hm1).max()))
            Hn = 2 * xs * Hm1 - (2 * (n_ - 1) * Hm2 if Hm2 is not None else 0)
            Hm2, Hm1 = Hm1, Hn
        self.Cn = Cn
        self.tol_f = 1e-6 * self.v / self.P
        self.mem_elems = 2 ** 25                                   # elements per (rows x N2) work block
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        d, S = self.d, len(ks)
        q0 = sigw ** 2 * (d - 1) / d
        self.t0_w = math.log(q0)
        # read-in proposal in (w0, log q)
        self.prop_w = PopProposal(bw=[0.5 * self.sw, 0.5 * math.sqrt(2 / (d - 1))], mu0=[0.0, self.t0_w],
                                  sd0=[self.sw, math.sqrt(2 / (d - 1))], sdb=[1.5, 1.0], sym=[True, False],
                                  weights=prop_weights, device=self.dev)
        # middle proposal in (z_S whitened, log r^2)
        r0 = self.sb ** 2 * (N1 - S)
        self.t0_b = math.log(r0)
        sdb_z = [30.0] * min(S, 2) + [1.0] * max(S - 2, 0)
        self.prop_b = PopProposal(bw=[0.5] * S + [0.5 * math.sqrt(2 / (N1 - S))], mu0=[0.0] * S + [self.t0_b],
                                  sd0=[1.0] * S + [math.sqrt(2 / (N1 - S))], sdb=sdb_z + [0.5],
                                  sym=[True] * S + [False], weights=prop_weights, device=self.dev)

    # ---------------------------------------------------------------------------------------------- collapsed U
    def _chunks(self):
        for i in range(0, self.P, self.chunk):
            yield slice(i, min(i + self.chunk, self.P))

    def suff_stats(self, W, B):
        G = torch.zeros(self.N2, self.N2, dtype=F64, device=self.dev)
        r = torch.zeros(self.N2, dtype=F64, device=self.dev)
        for sl in self._chunks():
            F2 = torch.erf(torch.erf(self.X[sl] @ W.T) @ B.T)
            G += (F2.T @ F2).double()
            r += (F2.T @ self.Y[sl]).double()
        return G, r

    def _solve(self, G, r):
        M = G / self.v + torch.eye(self.N2, dtype=F64, device=self.dev) / self.sa2
        L, info = torch.linalg.cholesky_ex(M)
        b = r / self.v
        z = torch.cholesky_solve(b[:, None], L)[:, 0]
        return M, L, z, b, int(info) == 0

    def U(self, ZW, ZB, grad=True):
        """Collapsed -log p(W, B) (a integrated out), in whitened coordinates W = sw ZW, B = sb ZB."""
        W, B = self.sw * ZW, self.sb * ZB
        G, r = self.suff_stats(W, B)
        M, L, z, b, ok = self._solve(G, r)
        if not ok:
            return float("inf"), None, None
        logdet = 2 * torch.log(torch.diagonal(L)).sum() + self.P * math.log(self.v) + self.N2 * math.log(self.sa2)
        u = 0.5 * (ZW.double() ** 2).sum() + 0.5 * (ZB.double() ** 2).sum() + 0.5 * (self.yy / self.v - b @ z) + 0.5 * logdet
        if not grad:
            return float(u), None, None
        Minv = torch.cholesky_inverse(L)
        Gbar = (0.5 * (torch.outer(z, z) + Minv) / self.v).to(self.dt)       # dU/dG
        rbar = (-z / self.v).to(self.dt)                                    # dU/dr
        gW = torch.zeros_like(W)
        gB = torch.zeros_like(B)
        for sl in self._chunks():
            A1 = self.X[sl] @ W.T
            F1 = torch.erf(A1)
            Hc = F1 @ B.T
            F2 = torch.erf(Hc)
            dF2 = 2 * F2 @ Gbar + self.Y[sl][:, None] * rbar[None, :]
            dH = dF2 * (SQPI * torch.exp(-Hc * Hc))
            gB += dH.T @ F1
            dA1 = (dH @ B) * (SQPI * torch.exp(-A1 * A1))
            gW += dA1.T @ self.X[sl]
        return float(u), ZW + self.sw * gW, ZB + self.sb * gB

    @torch.no_grad()
    def draw_a(self, W, B):
        G, r = self.suff_stats(W, B)
        M, L, z, b, ok = self._solve(G, r)
        xi = torch.randn(self.N2, 1, dtype=F64, device=self.dev, generator=self.gen)
        return (z + torch.linalg.solve_triangular(L.T, xi, upper=True)[:, 0]).to(self.dt)

    # ---------------------------------------------------------------------------------------------- HMC
    def hmc_step(self, ZW, ZB, u, gW, gB, step, n_leap):
        L = int(self.np_rng.integers(max(1, n_leap // 2), n_leap + 1))
        pW = torch.randn(ZW.shape, dtype=self.dt, device=self.dev, generator=self.gen)
        pB = torch.randn(ZB.shape, dtype=self.dt, device=self.dev, generator=self.gen)
        H0 = u + 0.5 * float((pW.double() ** 2).sum() + (pB.double() ** 2).sum())
        zW, zB, gw, gb = ZW, ZB, gW, gB
        pW = pW - 0.5 * step * gw
        pB = pB - 0.5 * step * gb
        un = u
        for li in range(L):
            zW = zW + step * pW
            zB = zB + step * pB
            un, gw, gb = self.U(zW, zB)
            if gw is None:
                break
            if li < L - 1:
                pW = pW - step * gw
                pB = pB - step * gb
        if gw is None or not math.isfinite(un):
            return ZW, ZB, u, gW, gB, 0.0, L
        pW = pW - 0.5 * step * gw
        pB = pB - 0.5 * step * gb
        H1 = un + 0.5 * float((pW.double() ** 2).sum() + (pB.double() ** 2).sum())
        dH = H1 - H0
        acc = math.exp(min(0.0, -dH)) if math.isfinite(dH) else 0.0
        if float(torch.rand(1, device=self.dev, generator=self.gen)) < acc:
            return zW, zB, un, gw, gb, acc, L
        return ZW, ZB, u, gW, gB, acc, L

    def flip_signs(self, ZW, ZB):
        s1 = (torch.randint(0, 2, (self.N1,), device=self.dev, generator=self.gen) * 2 - 1).to(self.dt)
        s2 = (torch.randint(0, 2, (self.N2,), device=self.dev, generator=self.gen) * 2 - 1).to(self.dt)
        return ZW * s1[:, None], ZB * s1[None, :] * s2[:, None]

    # ---------------------------------------------------------------------------------------------- jump moves
    @torch.no_grad()
    def _build(self, W, B, a):
        self.W, self.B, self.a = W.clone(), B.clone(), a.clone()
        self.F1 = torch.erf(self.X @ self.W.T)
        self.H = self.F1 @ self.B.T
        self.E = torch.erf(self.H)
        self.f = self.E @ self.a

    @torch.no_grad()
    def sweep(self, W, B, a, layers=("middle", "readin")):
        """One sweep of exact jump moves over all neurons of the given layers. Returns new (W, B, a), stats."""
        self._build(W, B, a)
        st = {}
        if "middle" in layers:
            st.update(self._middle_sweep())
        if "readin" in layers:
            st.update(self._readin_sweep())
        return self.W, self.B, self.a, st

    def _middle_sweep(self):
        S, N1, sb, v = len(self.ks), self.N1, self.sb, self.v
        C = readin_coeffs(self.W, self.ks).double()                   # (N1, S)
        CtC = C.T @ C
        Lc = torch.linalg.cholesky(sb ** 2 * CtC)                     # prior cov of h_S
        Pc = torch.linalg.solve(CtC, C.T).T                           # C (C^T C)^-1  (N1, S)
        Bd = self.B.double()
        hS = Bd @ C                                                   # (N2, S)
        Z = torch.linalg.solve_triangular(Lc, hS.T, upper=False).T    # whitened (N2, S)
        Bperp = Bd - hS @ Pc.T
        T = torch.log((Bperp ** 2).sum(1))                            # log r^2
        pop = torch.cat([Z, T[:, None]], 1)                           # (N2, S+1)
        nc = self.n_cand
        changed = crossed = 0
        for k in torch.randperm(self.N2, device=self.dev, generator=self.gen).tolist():
            cand = torch.cat([pop[k:k + 1], self.prop_b.sample_ex(pop, k, nc, self.gen)])   # (nc+1, S+1)
            logq = self.prop_b.logpdf_ex(cand, pop, k)
            zc, tc = cand[:, :S], cand[:, S]
            nperp = Bperp[k] / Bperp[k].norm()
            ncand = nperp[None].expand(nc + 1, -1).clone()
            if float(torch.rand(1, device=self.dev, generator=self.gen)) < self.redraw_frac:
                xi = torch.randn(nc, N1, dtype=F64, device=self.dev, generator=self.gen)
                xi = xi - (xi @ C) @ Pc.T
                ncand[1:] = xi / xi.norm(dim=1, keepdim=True)
            bc = (zc @ Lc.T) @ Pc.T + torch.exp(0.5 * tc)[:, None] * ncand   # (nc+1, N1)
            bc[0] = Bd[k]
            hc = self.F1 @ bc.T.to(self.dt)                               # (P, nc+1)
            phi = torch.erf(hc)
            fmk = self.f - self.a[k] * self.E[:, k]
            rho = (self.Y - fmk).double()
            ph = phi.double()
            A = 1.0 / self.sa2 + (ph * ph).sum(0) / v
            bb = (ph.T @ rho) / v
            Udata = -0.5 * bb ** 2 / A + 0.5 * torch.log(A)
            Uprior = 0.5 * (zc ** 2).sum(1) + 0.5 * torch.exp(tc) / sb ** 2 - 0.5 * (N1 - S) * tc
            lw = -(Udata + Uprior) - logq
            j = int(torch.multinomial(torch.softmax(lw - lw.max(), 0), 1, generator=self.gen))
            # a_k is redrawn from its exact conditional whether or not B_k moved
            ak = float(bb[j] / A[j] + torch.randn(1, dtype=F64, device=self.dev, generator=self.gen) / torch.sqrt(A[j]))
            if j != 0:
                changed += 1
                crossed += int((zc[0].abs().max() > 5) != (zc[j].abs().max() > 5))
                self.B[k] = bc[j].to(self.dt)
                Bd[k] = bc[j]
                self.H[:, k] = hc[:, j]
                self.E[:, k] = phi[:, j]
                pop[k] = cand[j]
                Bperp[k] = bc[j] - (bc[j] @ C) @ Pc.T
            self.a[k] = ak
            self.f = fmk + self.a[k] * self.E[:, k]
        return {"mid_changed": changed / self.N2, "mid_crossed": crossed / self.N2}

    def _erf_series(self, h0, g, phi, Bm, K):
        """sum_k a_k [erf(h0 + phi b_k) - erf(h0)] for every candidate column (phi (rows, c), Bm (N2, c)), by the
        Taylor series sum_n phi^n (G_n @ b^n), G_n = a erf^(n)(h0)/n!, erf^(n) = (-1)^(n-1) H_(n-1)(h0) g."""
        a = self.a
        ag = g * a[None, :]
        Hm2, Hm1 = None, torch.ones_like(h0)                       # physicists' H_{n-2}, H_{n-1}
        phin = torch.ones_like(phi)
        Bn = torch.ones_like(Bm)
        out = torch.zeros_like(phi)
        for n in range(1, K + 1):
            Gn = ag * Hm1 * ((-1) ** (n - 1) / math.factorial(n))
            phin = phin * phi
            Bn = Bn * Bm
            out = out + phin * (Gn @ Bn)
            Hn = 2 * h0 * Hm1 - (2 * (n - 1) * Hm2 if Hm2 is not None else 0)
            Hm2, Hm1 = Hm1, Hn
        return out

    def _series_order(self, bmax):
        """Smallest K whose Lagrange remainder bound sum_k |a_k| C_(K+1) bmax^(K+1)/(K+1)! is below tol_f (phi <= 1)."""
        A1 = float(self.a.abs().sum())
        for K in range(1, len(self.Cn) - 1):
            if A1 * self.Cn[K + 1] * bmax ** (K + 1) / math.factorial(K + 1) < self.tol_f:
                return K
        return None

    def _row_chunks(self, width):
        per = max(1, int(self.mem_elems // max(width, 1)))
        return [slice(i, min(i + per, self.P)) for i in range(0, self.P, per)]

    def _readin_sweep(self):
        d, sw, sb, v, nc = self.d, self.sw, self.sb, self.v, self.n_cand
        X0, Xp = self.X[:, 0], self.X[:, 1:]
        Wd = self.W.double()
        pop = torch.stack([Wd[:, 0], torch.log((Wd[:, 1:] ** 2).sum(1))], 1)
        changed = crossed = n_direct = 0
        thr = 0.5
        a = self.a
        chunks = self._row_chunks(4 * self.N2)
        single = len(chunks) == 1
        for j in torch.randperm(self.N1, device=self.dev, generator=self.gen).tolist():
            cand = torch.cat([pop[j:j + 1], self.prop_w.sample_ex(pop, j, nc, self.gen)])   # (nc+1, 2): (w0, log q)
            logq = self.prop_w.logpdf_ex(cand, pop, j)
            w0c, tc = cand[:, 0], cand[:, 1]
            qc = torch.exp(tc)
            wj = self.W[j]
            n = wj[1:] / wj[1:].norm()
            redraw = float(torch.rand(1, device=self.dev, generator=self.gen)) < self.redraw_frac
            if redraw:
                nn = torch.randn(nc + 1, d - 1, dtype=self.dt, device=self.dev, generator=self.gen)
                nn = nn / nn.norm(dim=1, keepdim=True)
                nn[0] = n
                up = Xp @ nn.T                                         # (P, nc+1)
            else:
                nn = None
                up = (Xp @ n)[:, None]
            phic = torch.erf(X0[:, None] * w0c.to(self.dt)[None] + up * torch.sqrt(qc).to(self.dt)[None])  # (P, nc+1)
            phic[:, 0] = self.F1[:, j]
            bj = self.B[:, j]
            # pass 1: rest of the state (middle pre-activations without read-in neuron j), field at B[:, j] = 0
            K0 = self._series_order(float(bj.abs().max())) if self.energy == "series" else None
            f0 = torch.empty_like(self.f)
            field = torch.zeros(self.N2, nc + 1, dtype=self.dt, device=self.dev)
            Dm_cache = None
            for sl in chunks:
                h0 = self.H[sl] - self.F1[sl, j:j + 1] * bj[None, :]
                g = SQPI * torch.exp(-h0 * h0)
                if K0 is None:
                    d0 = (torch.erf(self.H[sl]) - torch.erf(h0)) @ a
                else:
                    d0 = self._erf_series(h0, g, self.F1[sl, j:j + 1], bj[:, None], K0)[:, 0]
                f0[sl] = self.f[sl] - d0
                Dm = g * a[None, :]
                field += Dm.T @ ((f0[sl] - self.Y[sl])[:, None] * phic[sl]) / v
                if single:
                    Dm_cache = Dm
            # m solves (I/sb^2 + Dm^T diag(phi^2) Dm / v) m = -field (linearised conditional mean of B[:, j]), CG
            m = torch.zeros_like(field)
            rr = -field
            pp = rr.clone()
            rs = (rr * rr).sum(0)
            for _ in range(self.cg_iters):
                Ap = pp / sb ** 2
                for sl in chunks:
                    Dm = Dm_cache if single else a[None, :] * SQPI * torch.exp(
                        -(self.H[sl] - self.F1[sl, j:j + 1] * bj[None, :]) ** 2)
                    Ap = Ap + Dm.T @ ((phic[sl] ** 2 / v) * (Dm @ pp))
                al = rs / (pp * Ap).sum(0).clamp_min(1e-30)
                m = m + al * pp
                rr = rr - al * Ap
                rsn = (rr * rr).sum(0)
                pp = rr + (rsn / rs.clamp_min(1e-30)) * pp
                rs = rsn
            Bc = m + sb * torch.randn(m.shape, dtype=self.dt, device=self.dev, generator=self.gen)
            Bc[:, 0] = bj
            logN = -0.5 * ((Bc - m).double() ** 2).sum(0) / sb ** 2
            # pass 2: f for every candidate
            K = self._series_order(float(Bc.abs().max())) if self.energy == "series" else None
            fc = torch.empty(self.P, nc + 1, dtype=self.dt, device=self.dev)
            phT, BcT = phic.T.contiguous(), Bc.T.contiguous()
            for sl in chunks:
                h0 = self.H[sl] - self.F1[sl, j:j + 1] * bj[None, :]
                rows = h0.shape[0]
                if K is None:                                          # direct erf, batched over candidates
                    e0a = torch.erf(h0) @ a
                    per = max(1, int(self.mem_elems // (rows * self.N2)))
                    for i0 in range(0, nc + 1, per):
                        i1 = min(i0 + per, nc + 1)
                        c = i1 - i0
                        buf = torch.addcmul(h0.expand(c, -1, -1), phT[i0:i1, sl, None], BcT[i0:i1, None, :])
                        buf.erf_()
                        fc[sl, i0:i1] = ((buf.view(c * rows, self.N2) @ a).view(c, rows) - e0a).T + f0[sl, None]
                else:                                                  # Taylor series in phi b (many candidates)
                    g = SQPI * torch.exp(-h0 * h0)
                    fc[sl] = f0[sl, None] + self._erf_series(h0, g, phic[sl], Bc, K)
            n_direct += int(K is None)
            Ud = ((fc - self.Y[:, None]).double() ** 2).sum(0) / (2 * v)
            Up = 0.5 * (w0c ** 2 + qc) / sw ** 2 - 0.5 * (d - 1) * tc + 0.5 * (Bc.double() ** 2).sum(0) / sb ** 2
            lw = -(Ud + Up) - (logq + logN)
            i = int(torch.multinomial(torch.softmax(lw - lw.max(), 0), 1, generator=self.gen))
            if i == 0:
                continue
            changed += 1
            crossed += int((abs(float(w0c[0])) > thr) != (abs(float(w0c[i])) > thr))
            ni = nn[i] if redraw else n
            self.W[j, 0] = w0c[i].to(self.dt)
            self.W[j, 1:] = ni * torch.sqrt(qc[i]).to(self.dt)
            self.H.addr_(phic[:, i], Bc[:, i]).addr_(self.F1[:, j], bj, alpha=-1.0)
            self.F1[:, j] = phic[:, i]
            self.B[:, j] = Bc[:, i]
            self.f = fc[:, i].clone()
            pop[j] = cand[i]
        self.E = None                                                  # rebuilt at the next sweep
        return {"in_changed": changed / self.N1, "in_crossed": crossed / self.N1, "in_direct": n_direct / self.N1}

    # ---------------------------------------------------------------------------------------------- observables
    @torch.no_grad()
    def observables(self, W, B, a, xq):
        """Exact population quantities of the current weights, plus f-bar from the x0-quadrature test set xq."""
        S = len(self.ks)
        C = readin_coeffs(W, self.ks).double()
        Wd = W.double()
        w0 = Wd[:, 0]
        q = (Wd[:, 1:] ** 2).sum(1)
        q0 = self.sw ** 2 * (self.d - 1)
        J = C.T @ C / self.N1
        hS = B.double() @ C
        H = hS.T @ hS / self.N2
        # perp norm of each middle neuron: q_h = B_k (Jfull - C C^T) B_k^T
        nrm = 1 + 2 * (Wd * Wd).sum(1)
        Jf = (2 / math.pi) * torch.asin(2 * (Wd @ Wd.T) / torch.sqrt(torch.outer(nrm, nrm)))
        Jp = Jf - C @ C.T
        Bd = B.double()
        qh = ((Bd @ Jp) * Bd).sum(1)
        qbar = self.sb ** 2 * torch.trace(Jp)
        Lc = torch.linalg.cholesky(self.sb ** 2 * (C.T @ C))
        Z = torch.linalg.solve_triangular(Lc, hS.T, upper=False).T
        fb = xq.fbar(W, B, a)
        out = dict(d_w0sq=float((w0 ** 2).mean() / self.sw ** 2),
                   frac_in_out=float((w0.abs() > 0.5).double().mean()),
                   q_over_q0=float(q.mean() / q0),
                   qh_over_qbar=float(qh.mean() / qbar), frac_mid_out=float((Z.abs().max(1).values > 5).double().mean()),
                   a2_over_sa2=float((a.double() ** 2).mean() / self.sa2))
        for m in range(S):
            for n in range(m, S):
                out[f"J{self.ks[m]}{self.ks[n]}"] = float(J[m, n])
                out[f"H{self.ks[m]}{self.ks[n]}"] = float(H[m, n])
        for n, val in fb.items():
            out[f"fbar{n}"] = val
        return out


class X0Quadrature:
    """Test set for f-bar_n = E_x[phi_n(x0) f(x)]: Gauss-Hermite nodes in x0 times n_perp Gaussian samples of x_perp.
    Resolving x0 exactly removes most of the Monte Carlo error of a plain test set."""

    def __init__(self, d, n_perp, nx=48, nmax=7, device="cuda", dtype=torch.float32, seed=12345, chunk=8192):
        from numpy.polynomial.hermite_e import hermegauss
        x, w = hermegauss(nx)
        g = torch.Generator(device=device).manual_seed(seed)
        self.xp = torch.randn(n_perp, d - 1, device=device, dtype=dtype, generator=g)
        self.x0 = torch.tensor(x, device=device, dtype=dtype)
        self.w0 = torch.tensor(w / w.sum(), device=device, dtype=torch.float64)
        self.phis = he_norm(self.x0.double(), nmax)
        self.nmax = nmax
        self.chunk = chunk

    def fbar(self, W, B, a):
        acc = torch.zeros(len(self.x0), dtype=torch.float64, device=W.device)
        w0, wp = W[:, 0], W[:, 1:]
        for i in range(0, self.xp.shape[0], self.chunk):
            up = self.xp[i:i + self.chunk] @ wp.T                         # (m, N1)
            for g in range(len(self.x0)):
                F1 = torch.erf(up + self.x0[g] * w0[None])
                acc[g] += (torch.erf(F1 @ B.T) @ a).double().sum()
        fx0 = acc / self.xp.shape[0]                                        # E[f | x0] at the nodes
        return {n: float((self.w0 * self.phis[n] * fx0).sum()) for n in range(1, self.nmax + 1, 2)}
