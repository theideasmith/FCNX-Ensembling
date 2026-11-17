"""
Solver for the FCN3 equations of state (E1)-(E8) of notes/fcn3_matrix_eos.tex, exactly as written there:
every expectation is over the FULL distribution of the read-in rows and of the hidden field, on grids wide enough
that no part of either distribution is cut off.

Unknowns: J (s x s), qbar_h, M for the read-in layer, and t (length s) for the readout, with s = (n_max + 1)/2.

Numerics
  read-in    (w0, q) tensor grid, w0 >= 0 (the distribution is even in w0), sinh-spaced in w0, uniform in log q
             from q0 e^-16 to q0 e^1.5. E_{w,w'} k(w,w')^2 from the grid compressed into atoms (block sums), with
             Gauss-Jacobi quadrature in the cosine u, weight (1-u^2)^((d-4)/2).
  hidden     h = L z with L L^T = G. z = s u + Q_perp zeta, u = L^T t / |L^T t|. s on a lattice s >= 0 (the
             distribution is even under h -> -h), q_h on a uniform grid in log q_h from qbar e^-12 to qbar e^2,
             zeta by a product Gauss-Hermite rule in the directions to which the erf argument is sensitive.
  readout    Newton on t, with backtracking on the residual of (E4).

Why the iteration needs more than Anderson mixing
  The read-in and hidden-field distributions can each have local minima of -log p far from the origin. Their
  mass depends exponentially on the state: in the read-in action the c_3 term carries the coefficient
  (H33 - G33)/G33^2 with G33 ~ 1/d^3, so a fraction ~1e-4 of the rows in the far minimum changes J33 by a large
  factor, which in turn removes the minimum. Plain iteration then jumps between "all mass far" and "no mass
  far" and never converges.
  The solver therefore treats the mass of each far minimum as an unknown. Each grid point is assigned to the local
  minimum of -log p that discrete steepest descent reaches from it (basins). With basins B_0 (containing the prior
  mode), B_1, ..., the distribution used in an inner solve is
        p_ell(x) = p(x) * pi_g / P(B_g)    for x in B_g,   pi = softmax(0, ell_1, ell_2, ...),
  where P(B_g) is the mass of the full distribution p in B_g. At fixed ell the inner problem is smooth and is
  solved by Anderson acceleration. The logits ell are then solved from ell_g = log P(B_g) - log P(B_0) by Newton.
  At that root pi_g = P(B_g) for every g, so p_ell = p exactly: the solution is that of the full distributions,
  whatever the basins are. The basins only make the inner problem smooth.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import roots_jacobi

DT = torch.float64


# ----------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Params:
    d: int = 150
    P: float = 3000.0
    N1: float = 1600
    N2: int = 1600
    chi: float = 10.0
    kappa: float = 0.1
    eps: float = 0.074
    sigA: float = 1.0
    sigB: float = 1.0
    sigw: float = 1.0
    nmax: int = 11                    # odd Hermite subspace He_1, He_3, ..., He_nmax

    @property
    def ks(self):
        return tuple(range(1, self.nmax + 1, 2))

    @property
    def nS(self):
        return len(self.ks)

    @property
    def beta(self):
        return self.d / (2 * self.sigw ** 2)

    @property
    def gt(self):          # g~ = sigma_A^2 chi / N2
        return self.sigA ** 2 * self.chi / self.N2

    @property
    def lam(self):         # kappa / P
        return self.kappa / self.P

    @property
    def rB(self):          # sigma_B^2 N2 / N1
        return self.sigB ** 2 * self.N2 / self.N1

    def y(self):
        y = np.zeros(self.nS)
        y[0] = 1.0
        if self.nS > 1:
            y[1] = self.eps
        return y


@dataclass
class Numerics:
    # read-in grid
    nw: int = 1201
    wmax: float = 6.0
    wa: float = 0.5
    nq: int = 1401
    lq_below: float = 16.0       # log q from log q0 - lq_below ...
    lq_above: float = 1.5        # ... to log q0 + lq_above
    pair_block: tuple = (6, 7)   # grid block summed into one atom for E k(w,w')^2
    pair_prune: float = 1e-13    # atoms below this fraction of the largest are dropped
    nu: int = 8                  # Gauss-Jacobi nodes in u (8 and 48 agree to 1e-12 in M)
    # hidden field
    nx: int = 64                 # Gauss-Hermite nodes in x0 for e_n
    ds: float = 0.3              # s lattice spacing
    smax: float = 45.0
    nqh: int = 160
    lqh_below: float = 12.0      # log q_h from log qbar - lqh_below ...
    lqh_above: float = 2.0       # ... to log qbar + lqh_above
    # transverse Gauss-Hermite rule: (sensitivity ratio threshold, nodes). At least 2 nodes on every direction:
    # with 1 node per spectator direction the solution can be an artefact of the rule (a self-consistent but
    # wrong answer on the kappa0 = 0.015 ray, where |t| ~ 1e5 makes weakly sensitive directions matter).
    # FastSolver checks the rule by re-solving with every count doubled.
    zeta_levels: tuple = ((1e-2, 16), (1e-3, 6), (1e-5, 3), (0.0, 2))
    chunk: int = 1 << 23
    # solver
    newton_tol: float = 1e-11
    newton_maxit: int = 40
    outer_tol: float = 1e-8
    outer_maxit: int = 60
    cont_tol: float = 1e-5
    anderson_m: int = 6
    root_tol: float = 1e-6       # |ell - (log P(B_g) - log P(B_0))| at the target chi
    root_tol_cont: float = 1e-4  # ... at intermediate continuation steps
    root_maxit: int = 40
    basin_floor: float = -40.0   # far basins with log mass ratio below this stay in B_0


ZETA_ACCURATE = ((1e-2, 24), (1e-4, 6), (1e-6, 3), (0.0, 2))


class NoConvergence(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------
# features and helpers
# ----------------------------------------------------------------------------------------------------------
def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def cn_const(n):
    m = (n - 1) // 2
    return (2 / math.sqrt(math.pi)) * (-2) ** m * _dfact(n - 2) / math.sqrt(math.factorial(n))


def readin_features(W, Q, ks):
    """c_n(w0, q) = <phi_n | erf(w.x)>_x. Returns (s, *W.shape)."""
    T = 1 + 2 * W * W + 2 * Q
    return torch.stack([cn_const(n) * W ** n / T ** (n / 2) for n in ks])


def jxx(n2):
    """||erf(w.x)||_x^2 = k(w, w) as a function of |w|^2."""
    return (2 / math.pi) * torch.asin(2 * n2 / (1 + 2 * n2))


def hermite_norm(ks, x):
    """phi_n(x) = He_n(x)/sqrt(n!) for n in ks. Returns (s, len(x))."""
    H = [torch.ones_like(x), x.clone()]
    for n in range(1, max(ks)):
        H.append(x * H[n] - n * H[n - 1])
    return torch.stack([H[n] / math.sqrt(math.factorial(n)) for n in ks])


def trapz_weights(x):
    dx = x[1:] - x[:-1]
    w = torch.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def basin_labels(A):
    """Label every point of a 2-D action A (lower = more probable) by the flat index of the local minimum that
    discrete steepest descent over the 8 neighbours reaches. Row 0 lies on a mirror line (w0 = 0 or s = 0):
    its missing neighbours in row -1 are those of row 1."""
    n0, n1 = A.shape
    dev = A.device
    big = torch.full((n0 + 2, n1 + 2), math.inf, dtype=A.dtype, device=dev)
    big[1:-1, 1:-1] = A
    idx = torch.arange(n0 * n1, device=dev).reshape(n0, n1)
    bidx = torch.full((n0 + 2, n1 + 2), -1, dtype=torch.long, device=dev)
    bidx[1:-1, 1:-1] = idx
    if n0 > 1:
        big[0, 1:-1] = A[1]
        bidx[0, 1:-1] = idx[1]
    best, ptr = A.clone(), idx.clone()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            cand = big[1 + di:1 + di + n0, 1 + dj:1 + dj + n1]
            better = cand < best
            best = torch.where(better, cand, best)
            ptr = torch.where(better, bidx[1 + di:1 + di + n0, 1 + dj:1 + dj + n1], ptr)
    ptr = ptr.reshape(-1)
    for _ in range(int(math.ceil(math.log2(n0 * n1))) + 2):
        ptr = ptr[ptr]
    return ptr.reshape(n0, n1)


class Groups:
    """Partition of a grid into B_0 (the basin of the prior mode, plus negligible basins) and far basins
    B_1..B_m, with logits ell for the masses used in an inner solve (None = use the full distribution)."""

    def __init__(self, labels=None, roots=()):
        self.labels, self.roots = labels, list(roots)
        self.ell = None

    @property
    def m(self):
        return len(self.roots)

    def group_index(self):
        g = torch.zeros_like(self.labels)
        for k, r in enumerate(self.roots):
            g = torch.where(self.labels == r, k + 1, g)
        return g

    def apply(self, Lp):
        """Lp: log mass per grid point of the FULL distribution (unnormalised). Returns the log weights actually
        used, and delta_g = log P(B_g) - log P(B_0) for g = 1..m."""
        if self.m == 0:
            return Lp, []
        g = self.group_index()
        lse = [torch.logsumexp(torch.where(g == k, Lp, -torch.inf).reshape(-1), 0) for k in range(self.m + 1)]
        delta = [float(lse[k] - lse[0]) for k in range(1, self.m + 1)]
        if self.ell is None:
            return Lp, delta
        logits = torch.tensor([0.0] + list(self.ell), dtype=Lp.dtype, device=Lp.device)
        logpi = logits - torch.logsumexp(logits, 0)
        out = Lp.clone()
        for k in range(self.m + 1):
            out = torch.where(g == k, Lp - lse[k] + logpi[k], out)
        return out, delta


def make_groups(A, seed_index, floor):
    """Basins of action A; B_0 = basin of the grid point seed_index; far basins kept if their log mass relative
    to B_0 is above floor."""
    lab = basin_labels(A)
    r0 = int(lab.reshape(-1)[seed_index])
    Lp = -A
    l0 = torch.logsumexp(torch.where(lab == r0, Lp, -torch.inf).reshape(-1), 0)
    roots = []
    for r in torch.unique(lab).tolist():
        if r == r0:
            continue
        lr = torch.logsumexp(torch.where(lab == r, Lp, -torch.inf).reshape(-1), 0)
        if float(lr - l0) > floor:
            roots.append(r)
    return Groups(lab, roots)


# ----------------------------------------------------------------------------------------------------------
# read-in layer
# ----------------------------------------------------------------------------------------------------------
class ReadIn:
    def __init__(self, p: Params, nu: Numerics, dev):
        self.p, self.nu, self.dev = p, nu, dev
        d = p.d
        self.q0 = p.sigw ** 2 * (d - 1) / d
        w = nu.wa * torch.sinh(torch.linspace(0, math.asinh(nu.wmax / nu.wa), nu.nw, dtype=DT, device=dev))
        lq = torch.linspace(math.log(self.q0) - nu.lq_below, math.log(self.q0) + nu.lq_above, nu.nq,
                            dtype=DT, device=dev)
        W, LQ = torch.meshgrid(w, lq, indexing="ij")
        Q = torch.exp(LQ)
        C = readin_features(W, Q, p.ks)
        self.w, self.lq, self.W, self.Q, self.C = w, lq, W, Q, C
        # prior: density q^((d-3)/2) e^{-beta q} in q, i.e. q^((d-1)/2) e^{-beta q} per d(log q)
        self.base = p.beta * (W * W + Q) - 0.5 * (d - 1) * LQ
        self.knorm = jxx(W * W + Q)
        self.Pperp = self.knorm - (C * C).sum(0)
        self.logm = torch.log(torch.outer(trapz_weights(w), trapz_weights(lq)))
        self.seed = int(torch.argmin((lq - math.log(self.q0)).abs()))            # (w0 = 0, q = q0)
        x, wj = roots_jacobi(nu.nu, (d - 4) / 2, (d - 4) / 2)
        self.u = torch.tensor(x, dtype=DT, device=dev)
        self.wu = torch.tensor(wj / wj.sum(), dtype=DT, device=dev)
        self.groups = Groups()

    def action(self, Linv, V, mcoef):
        """-log p per (w0, log q) point: prior + (r_B/2) c^T G^-1 (G - H) G^-1 c + (r_B/2) mcoef P_perp,
        with V = I - E[z z^T] in the coordinates z = L^-1 h (so G^-1 (G - H) G^-1 = L^-T V L^-1)."""
        ct = torch.einsum("ij,jab->iab", Linv, self.C)
        return self.base + 0.5 * self.p.rB * torch.einsum("iab,ij,jab->ab", ct, V, ct) \
            + 0.5 * self.p.rB * mcoef * self.Pperp

    def solve(self, A):
        """Moments of the read-in distribution with action A (E6-E8 inputs)."""
        Lp = -A + self.logm
        Lr, delta = self.groups.apply(Lp)
        pw = torch.exp(Lr - torch.logsumexp(Lr.reshape(-1), 0))
        J = torch.einsum("iab,jab,ab->ij", self.C, self.C, pw)
        EJxx = (pw * self.knorm).sum()
        EJ2, J_at = self.pair_trace(pw)
        return dict(J=J, EJxx=EJxx, EJ2=EJ2, J_atoms=J_at, delta=delta, pw=pw,
                    w0sq=(pw * self.W ** 2).sum(), qmean=(pw * self.Q).sum())

    def pair_trace(self, pw):
        """E_{w,w'} k(w,w')^2 and the matching ||J||_F^2, both from atoms (block sums of the grid weights)."""
        bw, bq = self.nu.pair_block
        n0, n1 = pw.shape
        P0, P1 = -n0 % bw, -n1 % bq
        pad = lambda X: torch.nn.functional.pad(X, (0, P1, 0, P0))
        blk = lambda X: pad(X).reshape((n0 + P0) // bw, bw, (n1 + P1) // bq, bq).sum((1, 3)).reshape(-1)
        m = blk(pw)
        w0 = blk(pw * self.W) / m.clamp_min(1e-300)
        q = blk(pw * self.Q) / m.clamp_min(1e-300)
        keep = m > self.nu.pair_prune * m.max()
        m, w0, q = m[keep], w0[keep], q[keep]
        m = m / m.sum()
        Ca = readin_features(w0, q, self.p.ks)                                   # (s, natoms)
        J_at = (Ca * m) @ Ca.T
        T = 1 + 2 * (w0 * w0 + q)
        sq = torch.sqrt(q)
        acc = torch.zeros((), dtype=DT, device=self.dev)
        ch = max(1, (1 << 24) // (len(m) * len(self.u)))
        for i in range(0, len(m), ch):
            sl = slice(i, i + ch)
            z = (2 * torch.outer(w0[sl], w0)[..., None] + 2 * torch.outer(sq[sl], sq)[..., None] * self.u) \
                / torch.sqrt(torch.outer(T[sl], T))[..., None]
            k2 = ((2 / math.pi) * torch.asin(z.clamp(-1, 1))) ** 2 @ self.wu
            acc = acc + (m[sl, None] * m[None, :] * k2).sum()
        return acc, J_at

    def regroup(self, A):
        self.groups = make_groups(A, self.seed, self.nu.basin_floor)


# ----------------------------------------------------------------------------------------------------------
# hidden field
# ----------------------------------------------------------------------------------------------------------
class Middle:
    def __init__(self, p: Params, nu: Numerics, dev):
        self.p, self.nu, self.dev = p, nu, dev
        x, w = hermegauss(nu.nx)
        self.x = torch.tensor(x, dtype=DT, device=dev)
        self.wx = torch.tensor(w / math.sqrt(2 * math.pi), dtype=DT, device=dev)
        self.Phi = hermite_norm(p.ks, self.x)                                   # (s, nx)
        self.WPhi = (self.Phi * self.wx).T.contiguous()                          # (nx, s)
        self.s = nu.ds * torch.arange(0, int(math.ceil(nu.smax / nu.ds)) + 1, dtype=DT, device=dev)
        ws = torch.full_like(self.s, 2 * nu.ds)
        ws[0] = nu.ds
        self.logws = torch.log(ws)                                               # s >= 0 stands for +-s
        if p.nS == 1:
            self._set_zeta(torch.zeros(1, 0, dtype=DT), torch.ones(1, dtype=DT))
        self.counts = None
        self.groups = Groups()

    def _set_zeta(self, zeta, wz):
        self.zeta = zeta.to(self.dev)
        self.wz = wz.to(self.dev)
        self.logwz = torch.log(self.wz)

    @staticmethod
    def _product_rule(counts):
        zs, ws = [], []
        for n in counts:
            gz, gw = hermegauss(n)
            zs.append(torch.tensor(gz, dtype=DT))
            ws.append(torch.tensor(gw / gw.sum(), dtype=DT))
        Z = torch.meshgrid(*zs, indexing="ij")
        Wm = torch.meshgrid(*ws, indexing="ij")
        zeta = torch.stack([z.reshape(-1) for z in Z], 1)
        wz = torch.ones(len(zeta), dtype=DT)
        for w in Wm:
            wz = wz * w.reshape(-1)
        return zeta, wz

    def setup(self, L, t_dir, qbar, M, recount=False):
        p, dev, nu = self.p, self.dev, self.nu
        v = L.T @ t_dir
        if float(v.norm()) == 0.0:
            v = torch.zeros(p.nS, dtype=DT, device=dev)
            v[0] = 1.0
        u = v / v.norm()
        Qfull, _ = torch.linalg.qr(torch.cat([u[:, None], torch.eye(p.nS, dtype=DT, device=dev)], 1))
        Qp = Qfull[:, 1:p.nS]
        self.a = (L @ u) @ self.Phi                                              # erf argument per unit s
        if p.nS > 1:
            sw = torch.sqrt(self.wx)
            U, sv, _ = torch.linalg.svd(((L @ Qp).T @ self.Phi) * sw, full_matrices=False)
            Qp = Qp @ U
            ratio = (sv / (self.a * sw).norm()).cpu().numpy()
            if recount or self.counts is None:
                counts = [next(n for thr, n in nu.zeta_levels if rt >= thr) for rt in ratio]
                if counts != self.counts:
                    self.counts = counts
                    self._set_zeta(*self._product_rule(counts))
        self.u, self.Qp, self.L = u, Qp, L
        self.A = self.zeta @ ((L @ Qp).T @ self.Phi)                              # (nz, nx)
        k = M / 2
        lqh = math.log(qbar) + torch.linspace(-nu.lqh_below, nu.lqh_above, nu.nqh, dtype=DT, device=dev)
        self.qh = torch.exp(lqh)
        lw = torch.log(trapz_weights(lqh)) + k * (lqh - math.log(qbar)) - k * self.qh / qbar
        self.logwq = lw - torch.logsumexp(lw, 0)                                 # prior of q_h, eq. (19)
        self.rq = 1 / torch.sqrt(1 + 2 * self.qh)
        self.qbar, self.M = qbar, M
        self.seed = int(torch.argmin((lqh - math.log(qbar)).abs()))              # (s = 0, q_h = qbar)
        self.zz = (self.zeta[:, :, None] * self.zeta[:, None, :]).reshape(len(self.zeta), -1)
        S1 = p.nS - 1
        self.Zz2 = (self.wz[:, None] * self.zz).sum(0).reshape(S1, S1)

    def cells(self, t):
        """Per (s, q_h) cell: log mass of the full distribution and conditional means over zeta."""
        p, nu = self.p, self.nu
        S, nz, nqh = p.nS, len(self.zeta), len(self.qh)
        per = max(1, nu.chunk // (nqh * nz * nu.nx))
        out = {k: [] for k in ("L", "BB", "tB2", "t2BB", "e1", "zz2", "Vt")}
        tphi = (t @ self.Phi) * self.wx                                          # (nx,)
        PP = (self.Phi[:, None, :] * self.Phi[None, :, :]).reshape(S * S, -1).T  # (nx, S*S)
        for i0 in range(0, len(self.s), per):
            s = self.s[i0:i0 + per]
            ns = len(s)
            arg = (s[:, None, None] * self.a + self.A[None])[:, None] * self.rq[None, :, None, None]
            ep = (2 / math.sqrt(math.pi)) * torch.exp(-arg * arg)               # erf'(a)
            G1 = ((ep * tphi) @ self.Phi.T) * self.rq[None, :, None, None]       # grad_h (t.e)
            H2 = ((-2 * arg * ep * tphi) @ PP).reshape(*arg.shape[:3], S, S) \
                * (self.rq ** 2)[None, :, None, None, None]                      # Hess_h (t.e)
            arg.erf_()
            B = (arg @ self.WPhi)                                                # (ns, nqh, nz, S)  = e_n
            tB = B @ t
            lw = self.logwz + 0.5 * p.gt * tB * tB                               # (ns, nqh, nz)
            m = lw.amax(2, keepdim=True)
            e = torch.exp(lw - m)
            Z = e.sum(2)
            f = e / Z[..., None]
            out["L"].append(m[..., 0] + torch.log(Z) + self.logwq[None] - 0.5 * (s * s)[:, None]
                            + self.logws[i0:i0 + ns, None])
            out["BB"].append(torch.einsum("abz,abzi,abzj->abij", f, B, B))
            out["tB2"].append(torch.einsum("abz,abzi->abi", f * tB, B))
            out["t2BB"].append(torch.einsum("abz,abzi,abzj->abij", f * tB * tB, B, B))
            out["e1"].append(f @ self.zeta)
            out["zz2"].append((f @ self.zz).reshape(ns, nqh, S - 1, S - 1))
            # Stein: G^-1 (G - H) G^-1 = -E[Hess T + grad T grad T^T], T = (g~/2)(t.e)^2
            Wt = ((1 + p.gt * tB * tB)[..., None, None] * G1[..., :, None] * G1[..., None, :]
                  + tB[..., None, None] * H2)
            out["Vt"].append(torch.einsum("abz,abzij->abij", f, Wt))
        return {k: torch.cat(v) for k, v in out.items()}

    def stats(self, c):
        """Moments over the cells, using the group weights (full distribution when ell is None)."""
        Lr, delta = self.groups.apply(c["L"])
        logZ = torch.logsumexp(c["L"].reshape(-1), 0)
        pi = torch.exp(Lr - torch.logsumexp(Lr.reshape(-1), 0))                 # (ns, nqh)
        avg = lambda X: torch.einsum("ab,ab...->...", pi, X)
        K = avg(c["BB"])
        Ktb = avg(c["tB2"])
        if self.groups.m and self.groups.ell is not None:
            # group masses are held fixed: the Jacobian of K t sees only the covariance within each group
            gi = self.groups.group_index()
            Cov = torch.zeros_like(K)
            for k in range(self.groups.m + 1):
                pk = torch.where(gi == k, pi, torch.zeros_like(pi))
                mk = pk.sum()
                if float(mk) > 0:
                    mt = torch.einsum("ab,abi->i", pk, c["tB2"]) / mk
                    Cov = Cov + torch.einsum("ab,abij->ij", pk, c["t2BB"]) - mk * torch.outer(mt, mt)
        else:
            Cov = avg(c["t2BB"]) - torch.outer(Ktb, Ktb)
        ps = pi.sum(1)
        u, Qp = self.u, self.Qp
        cross = Qp @ avg(self.s[:, None, None] * c["e1"])
        Ezz = (ps @ (self.s ** 2)) * torch.outer(u, u) + torch.outer(u, cross) + torch.outer(cross, u) \
            + Qp @ avg(c["zz2"]) @ Qp.T
        Eq = pi.sum(0) @ self.qh
        # The same moments under the prior, on the same nodes. Exactly, E0[z z^T] = I and E0[q_h] = qbar; on the
        # nodes they differ by the quadrature error, which is common to both and cancels in the differences
        # E[zz^T] - E0[zz^T] and E[q_h] - E0[q_h]. Those differences enter (E5) multiplied by 1/G_nn, which reaches
        # 1e17 for n = 11, so the cancellation is needed.
        lp0 = self.logws - 0.5 * self.s ** 2
        p0 = torch.exp(lp0 - torch.logsumexp(lp0, 0))
        Ezz0 = (p0 @ (self.s ** 2)) * torch.outer(u, u) + Qp @ self.Zz2 @ Qp.T
        Eq0 = torch.exp(self.logwq) @ self.qh
        I = torch.eye(len(u), dtype=DT, device=self.dev)
        Vh = -self.p.gt * avg(c["Vt"])                                           # G^-1 (G - H) G^-1
        return dict(K=K, Cov=Cov, Ezz=Ezz - Ezz0 + I, Eq=Eq - Eq0 + self.qbar, Ezz_raw=Ezz, Ezz0=Ezz0,
                    Vh=Vh, logZ=logZ, delta=delta, pi=pi)

    def action_grid(self, c):
        """-log density per (s, log q_h) cell, for the basins (lattice weights removed)."""
        return -(c["L"] - self.logws[:, None])

    def regroup(self, c):
        self.groups = make_groups(self.action_grid(c), self.seed, self.nu.basin_floor)


def solve_t(mid: Middle, p: Params, L, qbar, M, t0, tol, maxit, recount=False):
    """Readout (E4): (sigma_A^2 K + kappa/P) t = y, K from the hidden-field distribution at t. Newton with the
    Jacobian sigma_A^2 (K + g~ Cov[(t.e) e]) + kappa/P and backtracking on the residual."""
    dev = mid.dev
    y = torch.tensor(p.y(), dtype=DT, device=dev)
    I = torch.eye(p.nS, dtype=DT, device=dev)
    t = t0.clone()
    mid.setup(L, t if float(t.norm()) > 0 else y, qbar, M, recount)
    c = mid.cells(t)
    st = mid.stats(c)
    npass = 1
    res = lambda t, st: p.lam * t + p.sigA ** 2 * (st["K"] @ t) - y
    for _ in range(maxit):
        g = res(t, st)
        Hm = p.lam * I + p.sigA ** 2 * (st["K"] + p.gt * st["Cov"])
        step = -torch.linalg.solve(Hm, g)
        if float(step.norm() / t.norm().clamp_min(1e-300)) < tol:
            break
        a, g0 = 1.0, float(g.norm())
        while True:
            tn = t + a * step
            cn = mid.cells(tn)
            stn = mid.stats(cn)
            npass += 1
            if float(res(tn, stn).norm()) < (1 - 1e-4 * a) * g0 or a < 1e-3:
                break
            a *= 0.5
        t, c, st = tn, cn, stn
    st["cells"] = c
    st["Eq0"] = torch.tensor(qbar, dtype=DT, device=dev)
    return t, st, npass


# ----------------------------------------------------------------------------------------------------------
# full solver
# ----------------------------------------------------------------------------------------------------------
class FCN3Solver:
    def __init__(self, p: Params, nu: Numerics = None, device=None, verbose=True):
        self.p = p
        self.nu = nu or Numerics()
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.readin = ReadIn(p, self.nu, self.dev)
        self.mid = Middle(p, self.nu, self.dev)
        self.verbose = verbose
        self.npass = 0
        I = torch.eye(p.nS, dtype=DT, device=self.dev)
        ri0 = self.readin.solve(self.readin.base)                                # prior read-in distribution
        self.J0 = ri0["J"]
        self.Jsc = torch.sqrt(torch.outer(torch.diag(self.J0), torch.diag(self.J0)))
        self.iu = torch.triu_indices(p.nS, p.nS, device=self.dev)
        self.state0 = self._state_from(ri0)
        self.t = torch.zeros(p.nS, dtype=DT, device=self.dev)
        self.last = None
        self._I = I

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    def _state_from(self, ri):
        p = self.p
        qbar = float(p.sigB ** 2 * (ri["EJxx"] - torch.trace(ri["J"])))                # (E7)
        trG2 = float(p.sigB ** 4 * (ri["EJ2"] - (ri["J_atoms"] ** 2).sum()))           # (E8)
        return dict(J=ri["J"], qbar=qbar, M=qbar ** 2 / trG2)

    def encode(self, s):
        """State -> iteration variables: log J_nn, atanh of the correlations J_mn / sqrt(J_mm J_nn), log qbar,
        log M. The diagonal of J spans many decades (J_nn ~ d^-n) and a small far-basin mass changes the high-n
        entries by orders of magnitude, so the iteration has to work in relative (log) units."""
        J = s["J"]
        dg = torch.sqrt(torch.diag(J))
        R = J / torch.outer(dg, dg)
        il = torch.triu_indices(self.p.nS, self.p.nS, offset=1, device=self.dev)
        r = torch.atanh(R[il[0], il[1]].clamp(-1 + 1e-15, 1 - 1e-15))
        return torch.cat([torch.log(torch.diag(J)), r,
                          torch.tensor([math.log(s["qbar"]), math.log(s["M"])], dtype=DT, device=self.dev)])

    def decode(self, x):
        nS = self.p.nS
        dg = torch.exp(0.5 * x[:nS])
        il = torch.triu_indices(nS, nS, offset=1, device=self.dev)
        R = torch.eye(nS, dtype=DT, device=self.dev)
        R[il[0], il[1]] = torch.tanh(x[nS:-2])
        R = R + R.T - torch.eye(nS, dtype=DT, device=self.dev)
        return dict(J=R * torch.outer(dg, dg), qbar=math.exp(float(x[-2])), M=math.exp(float(x[-1])))

    def F(self, s, recount=False):
        """One pass of (E1)-(E8): read-in state -> hidden field -> readout -> read-in distribution -> new state."""
        p = self.p
        G = p.sigB ** 2 * s["J"]                                                         # (E1)
        L = torch.linalg.cholesky(G)
        t, st, npass = solve_t(self.mid, p, L, s["qbar"], s["M"], self.t, self.nu.newton_tol,
                               self.nu.newton_maxit, recount)                            # (E2)-(E4)
        self.npass += npass
        V = st["Vh"]                                                     # G^-1(G-H)G^-1, by Stein's lemma
        mcoef = s["M"] * (s["qbar"] - float(st["Eq"])) / s["qbar"] ** 2
        Linv = self._I
        A = self.readin.action(Linv, V, mcoef)                                           # (E5)
        ri = self.readin.solve(A)                                                        # (E6)
        self.t = t
        self.last = dict(state=s, t=t, mid=st, ri=ri, A=A, L=L, Linv=Linv, V=V, mcoef=mcoef)
        return self._state_from(ri)                                                      # (E7), (E8)

    # -- inner solve at fixed groups and logits -----------------------------------------------------------
    def anderson(self, s, tol=None, label=""):
        nu = self.nu
        tol = tol or nu.outer_tol
        x = self.encode(s)
        X, Fh = [], []
        res = float("inf")
        for k in range(nu.outer_maxit):
            try:
                gx = self.encode(self.F(self.decode(x), recount=(k == 0)))
            except torch.linalg.LinAlgError:
                if not X:
                    raise NoConvergence("singular G at the start")
                x = X[-1] + 0.5 * Fh[-1]
                X, Fh = [], []
                continue
            f = gx - x
            res = float(f.abs().max())
            if res < tol:
                return self.decode(gx), res
            X.append(x.clone())
            Fh.append(f.clone())
            if len(X) > nu.anderson_m + 1:
                X.pop(0)
                Fh.pop(0)
            if len(X) >= 2:
                dX = torch.stack([X[i + 1] - X[i] for i in range(len(X) - 1)], 1).cpu()
                dF = torch.stack([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)], 1).cpu()
                gam = torch.linalg.lstsq(dF, f.cpu()[:, None], rcond=1e-12, driver="gelsd").solution[:, 0]
                x = x + f - ((dX + dF) @ gam).to(self.dev)
            else:
                x = x + f
        raise NoConvergence(f"{label}inner iteration did not converge (res={res:.2e})")

    # -- far-basin masses ---------------------------------------------------------------------------------
    def _ells(self):
        return list(self.readin.groups.ell or []) + list(self.mid.groups.ell or [])

    def _set_ells(self, ell):
        mr = self.readin.groups.m
        self.readin.groups.ell = list(ell[:mr]) if mr else None
        self.mid.groups.ell = list(ell[mr:]) if self.mid.groups.m else None

    def _deltas(self):
        return list(self.last["ri"]["delta"]) + list(self.last["mid"]["delta"])

    def _regroup(self):
        """Basins of both distributions at the last evaluated state."""
        self.readin.regroup(self.last["A"])
        self.mid.regroup(self.last["mid"]["cells"])

    def solve_fixed_chi(self, s, tol, root_tol=None):
        """Solve at the current parameters from state s. Returns (state, residual, logits)."""
        rt = root_tol or self.nu.root_tol
        self.readin.groups, self.mid.groups = Groups(), Groups()
        self.F(s, recount=True)
        saved = None                                      # last non-empty partition seen
        for rnd in range(6):
            self._regroup()
            if self.readin.groups.m + self.mid.groups.m:
                saved = (self.readin.groups, self.mid.groups)
            else:
                try:
                    s2, res = self.anderson(s, tol, "plain ")
                    self._regroup()
                    if self.readin.groups.m + self.mid.groups.m == 0:
                        return s2, res, []
                    s = s2
                    saved = (self.readin.groups, self.mid.groups)
                except NoConvergence:
                    self._regroup()
                    if self.readin.groups.m + self.mid.groups.m == 0:
                        if saved is None:
                            raise
                        self.readin.groups, self.mid.groups = saved
                        self.readin.groups.ell = self.mid.groups.ell = None
            s, res, ell, ok = self._root(s, min(tol, 1e-2 * rt), rt)
            if ok:
                return s, res, ell
        raise NoConvergence("could not find the far-basin masses")

    def _g(self, s, ell, tol):
        """Inner solve at logits ell; returns the state and g = delta - ell."""
        self._set_ells(ell)
        s2, res = self.anderson(s, tol, f"ell={np.round(ell, 3).tolist()} ")
        return s2, res, np.array(self._deltas()) - np.array(ell)

    def _root(self, s, tol, root_tol):
        """Logits ell with ell_g = log P(B_g) - log P(B_0) for every far basin, one logit at a time (each g is
        decreasing in its own logit): a basin whose g is already negative at the floor has a self-consistent
        mass below e^floor and is fixed there; otherwise Brent's method on a bracket [floor, upper]."""
        from scipy.optimize import brentq
        nu = self.nu
        n = self.readin.groups.m + self.mid.groups.m
        floor = nu.basin_floor
        prev = getattr(self, "ell_prev", None)
        ell = np.array(prev, float) if prev is not None and len(prev) == n else np.full(n, floor / 2)
        frozen = np.zeros(n, bool)
        box = dict(s=s, res=None, g=None)

        def f(i, x):
            e = ell.copy()
            e[i] = x
            s2, res, g = self._g(box["s"], e, tol)
            box.update(s=s2, res=res, g=g)
            return g[i]

        for sweep in range(8):
            for i in range(n):
                if frozen[i]:
                    continue
                # bracket outward from the current logit (g is decreasing in its own logit)
                x0 = max(ell[i], floor)
                f0 = f(i, x0)
                if abs(f0) < root_tol:
                    ell[i] = x0
                    continue
                step, x1 = 0.25, x0
                while True:
                    x1 = x0 + step if f0 > 0 else x0 - step
                    if x1 <= floor:
                        x1 = floor
                    f1 = f(i, x1)
                    if (f1 > 0) != (f0 > 0) or x1 == floor or x1 > 80:
                        break
                    x0, f0, step = x1, f1, step * 2
                if x1 == floor and f1 <= 0 and f0 <= 0:
                    ell[i] = floor - 20
                    frozen[i] = True
                    self.log(f"  basin {i}: self-consistent mass below e^{floor:.0f}, fixed at e^{ell[i]:.0f}")
                    continue
                if (f1 > 0) == (f0 > 0):
                    return box["s"], box["res"], ell.tolist(), False
                ell[i] = brentq(lambda x: f(i, x), min(x0, x1), max(x0, x1), xtol=0.1 * root_tol, rtol=1e-14,
                                maxiter=100)
            s2, res, g = self._g(box["s"], ell, tol)
            box.update(s=s2, res=res, g=g)
            gm = np.abs(np.where(frozen, 0, g)).max() if n else 0.0
            self.log(f"  sweep {sweep}: ell={np.round(ell, 4).tolist()} g={np.round(g, 7).tolist()} res={res:.1e}")
            if gm < root_tol:
                self.ell_prev = ell.tolist()
                return s2, res, ell.tolist(), True
        return box["s"], box["res"], ell.tolist(), False

    def solve(self, init=None, chi_start=None):
        """Solve at self.p. From init (another solver's .state()) directly; otherwise by continuation in chi
        from chi/64, since the far basins appear as chi grows."""
        t0 = time.time()
        self.npass = 0
        chi_t = self.p.chi
        if init is not None:
            self.t = init["t"].to(self.dev)
            self.ell_prev = init.get("ell")
            try:
                s, res, ell = self.solve_fixed_chi(dict(J=init["J"].to(self.dev), qbar=init["qbar"], M=init["M"]),
                                                   self.nu.outer_tol)
                return self._finish(s, res, ell, t0)
            except NoConvergence as e:
                self.log(f"  warm start failed ({e}); continuation in chi")
                self.t = torch.zeros_like(self.t)
        lc, lt = math.log(chi_start or chi_t / 64), math.log(chi_t)
        step = math.log(2.0)
        self.p.chi = math.exp(lc)
        s, res, ell = self.solve_fixed_chi(self.state0, self.nu.cont_tol, self.nu.root_tol_cont)
        while lc < lt - 1e-12:
            nxt = min(lc + step, lt)
            self.p.chi = math.exp(nxt)
            tsave, ssave = self.t.clone(), s
            last = nxt >= lt - 1e-12
            try:
                s, res, ell = self.solve_fixed_chi(s, self.nu.outer_tol if last else self.nu.cont_tol,
                                                   None if last else self.nu.root_tol_cont)
                self.log(f"chi={self.p.chi:.4g}: res={res:.1e} far-basin logits={np.round(ell, 3).tolist()} "
                         f"({time.time() - t0:.1f}s)")
                lc = nxt
                step = min(step * 1.5, math.log(2.0))
            except NoConvergence as e:
                self.log(f"chi={self.p.chi:.4g}: {e}; halving the step")
                self.t, s = tsave, ssave
                step /= 2
                if step < 1e-3:
                    self.p.chi = chi_t
                    raise
        self.p.chi = chi_t
        return self._finish(s, res, ell, t0)

    def _finish(self, s, res, ell, t0):
        self.state_ = s
        self.ell = ell
        self.converged = res is not None and res < self.nu.outer_tol
        self.walltime = time.time() - t0
        return self.results()

    def results(self):
        p, lt = self.p, self.last
        st, ri, L = lt["mid"], lt["ri"], lt["L"]
        G = p.sigB ** 2 * lt["state"]["J"]
        H = L @ st["Ezz"] @ L.T
        K, t = st["K"], lt["t"]
        fbar = p.sigA ** 2 * (K @ t)
        npy = lambda a: a.detach().cpu().numpy().tolist()
        # mass in each far basin, from the full distributions
        dr, dh = list(ri["delta"]), list(st["delta"])
        frac = lambda ds: [math.exp(d - float(np.logaddexp.reduce([0.0] + ds))) for d in ds]
        return dict(params=asdict(p), converged=bool(self.converged), walltime_s=self.walltime,
                    middle_passes=self.npass, d_w0sq=float(p.d * ri["w0sq"]),
                    qperp_over_prior=float(ri["qmean"] / self.readin.q0),
                    J=npy(ri["J"]), H=npy(H), K=npy(K), G=npy(G), t=npy(t), fbar=npy(fbar),
                    L1=float(fbar[0]), L3=float(fbar[1] / p.eps) if p.nS > 1 else None,
                    qh_over_qbar=float(st["Eq"] / lt["state"]["qbar"]), qbar=lt["state"]["qbar"],
                    M=lt["state"]["M"], m_perp=lt["mcoef"],
                    readin_far_mass=frac(dr), hidden_far_mass=frac(dh), far_logits=self.ell)

    def state(self):
        s = self.last["state"]
        return dict(J=s["J"], qbar=s["qbar"], M=s["M"], t=self.last["t"], ell=getattr(self, "ell_prev", None))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for f_, v in asdict(Params()).items():
        ap.add_argument(f"--{f_}", type=type(v), default=v)
    for f_ in ("nx", "nqh", "ds", "smax", "nw", "nq"):
        v = getattr(Numerics(), f_)
        ap.add_argument(f"--{f_}", type=type(v), default=v)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write JSON results here")
    a = ap.parse_args()
    p = Params(**{k: getattr(a, k) for k in asdict(Params())})
    nu = Numerics(**{k: getattr(a, k) for k in ("nx", "nqh", "ds", "smax", "nw", "nq")})
    S = FCN3Solver(p, nu, a.device)
    r = S.solve()
    print(json.dumps({k: r[k] for k in ("converged", "walltime_s", "middle_passes", "d_w0sq", "L1", "L3",
                                        "qh_over_qbar", "M", "m_perp", "readin_far_mass", "hidden_far_mass")},
                     indent=1))
    print("H =", np.array(r["H"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()


def solve_t_convex(mid: Middle, p: Params, L, qbar, M, t0, tol=1e-9, maxit=200, recount=True):
    """Readout (E4) at a FIXED prior field (N1 = inf, or any fixed G), over the full hidden-field distribution, as
    the minimum of the strictly convex potential
        Phi(t) = (kappa/2P)|t|^2 + (N2/chi) log Z_h(t) - y.t,
        grad Phi = (sigma_A^2 K + kappa/P) t - y,   Hess Phi = kappa/P + sigma_A^2 (K + g~ Cov[(t.e) e]).
    Newton with Armijo backtracking on Phi converges from any start, however steep the solution is in the
    parameters, so no continuation is needed. The quadrature frame is fixed by the start direction."""
    dev = mid.dev
    y = torch.tensor(p.y(), dtype=DT, device=dev)
    I = torch.eye(p.nS, dtype=DT, device=dev)
    t = t0.clone()
    mid.groups = Groups()
    mid.setup(L, t if float(t.norm()) > 0 else y, qbar, M, recount=recount)
    c_ = p.N2 / p.chi
    phi = lambda t, st: 0.5 * p.lam * float(t @ t) + c_ * float(st["logZ"]) - float(y @ t)
    st = mid.stats(mid.cells(t))
    for it in range(maxit):
        g = p.lam * t + p.sigA ** 2 * (st["K"] @ t) - y
        Hm = p.lam * I + p.sigA ** 2 * (st["K"] + p.gt * st["Cov"])
        step = -torch.linalg.solve(Hm, g)
        if float(g.abs().max()) < tol:
            break
        f0, slope, a = phi(t, st), float(g @ step), 1.0
        while True:
            tn = t + a * step
            stn = mid.stats(mid.cells(tn))
            if phi(tn, stn) <= f0 + 1e-4 * a * slope or a < 1e-10:
                break
            a *= 0.5
        t, st = tn, stn
    else:
        raise NoConvergence(f"solve_t_convex: |grad Phi| = {float(g.abs().max()):.1e} after {maxit} iterations")
    st["cells"] = None
    return t, st, it
