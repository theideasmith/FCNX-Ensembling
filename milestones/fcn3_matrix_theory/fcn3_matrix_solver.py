"""
GPU solver for the FCN3 matrix equations of state (notes "FCN3 in Hermite matrix elements", Nov 2026,
eqs. E1-E8), on the odd x0 Hermite tower S = {1, 3, ..., nmax}.

Structure (nested solve):
  outer  Anderson-accelerated fixed point on the read-in state (J_S, qbar_h, M);
  inner  Newton on the convex potential  Phi(t) = (kappa/2P)|t|^2 + (N2/chi) log Z_h(t) - y.t,
         whose stationarity is the readout condition (sigma_A^2 K + kappa/P) t = y.

Middle-layer integral (|S|+1 dims). Whitened coordinates h = L z, G_S = L L^T, z = s u + Q_perp zeta,
with u = L^T t / |L^T t| the drive direction at h = 0. Quadrature:
  s      uniform lattice (trapezoid; spectrally accurate for the smooth marginal), s >= 0 only, using the
         exact symmetry h -> -h of the law (B is odd, so (t.B)^2 is even);
  q_h    Gauss-Hermite in log q_h, weights exact against the two-moment Gamma prior (eq. 16);
  zeta   the |S|-1 transverse directions, rotated to the singular directions of zeta -> erf argument.
         Their sensitivities fall off geometrically (at the thesis point 7e-2, 6e-5, 1e-6, 3e-8, 8e-10 of the
         s-direction), so a product Gauss-Hermite rule with 24 nodes on the active direction and 2-3 on the
         spectators is exact to ~1e-9 (zeta_rule="qmc" gives antithetic Sobol instead, ~1e-4 at 2048 points).
The per-s sufficient statistics give the exact s-marginal, so the central-basin cut ("first maximum
of -log p") is a cut on the s lattice. Corrections that are differences from the prior
(H - G, <q_h> - qbar_h) use the same nodes with prior weights as a control variate, so quadrature error
in the prior moments cancels.

Read-in layer: (w0, q_perp) tensor grid (sinh-spaced w0 >= 0, log-uniform q), exact features (eq. 3),
action eq. (13), central-basin cut in w0. M = (tr G_perp)^2 / tr G_perp^2 from eq. (18): pairs of read-in
neurons on a coarser grid, with Gauss-Jacobi quadrature for the angle (weight (1-u^2)^((d-4)/2)).

All accumulation is float64 on the GPU.

Branches. "central" (default) restricts both layer laws to the basin around the origin (notes, sec. 4).
"global" is implemented in the quadrature but the solve does not work yet: chi continuation from the
prior follows the central branch and fails where it ends, and the outer-well state needs its own start.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import roots_jacobi

DT = torch.float64
LOG2PI = math.log(2 * math.pi)


# ----------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Params:
    d: int = 150
    P: float = 3000.0
    N1: int = 1600
    N2: int = 1600
    chi: float = 10.0
    kappa: float = 0.1
    eps: float = 0.074
    sigA: float = 1.0
    sigB: float = 1.0
    sigw: float = 1.0
    nmax: int = 3

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
    branch: str = "central"      # "central" (basin around the origin, both layers) or "global"
    # read-in fine grid
    nw: int = 1201
    nq: int = 256
    wmax: float = 6.0
    wa: float = 0.5
    qlo: float = 1.2             # grid extends at least this far below log q0
    nsig_q: float = 10.0
    # read-in coarse grid for tr G_perp^2
    nwc: int = 160
    nqc: int = 40
    nu: int = 24
    # middle layer
    nx: int = 64                 # Gauss-Hermite nodes in x0 for B_n
    nqh: int = 16                # Gauss-Hermite nodes in log q_h
    zeta_rule: str = "auto"      # "auto": product Gauss-Hermite in the SVD-rotated transverse basis; "qmc": Sobol
    # (sensitivity ratio threshold, nodes): direction i gets the first n with ratio_i >= threshold
    zeta_levels: tuple = ((1e-2, 24), (1e-4, 6), (1e-6, 3), (0.0, 2))
    nzeta_qmc: int = 2048        # zeta_rule="qmc" (antithetic pairs -> even)
    ds: float = 0.3              # s-lattice spacing (whitened units)
    smax: float = 45.0           # widest s range (covers the outer well)
    smargin: float = 1.4         # central branch: integrate to smargin * s_cut + 3
    chunk: int = 1 << 23         # elements per erf block
    # solver
    newton_tol: float = 1e-11
    newton_maxit: int = 40
    outer_tol: float = 1e-8
    outer_maxit: int = 80
    cont_tol: float = 1e-4       # outer tolerance at intermediate continuation steps
    anderson_m: int = 6
    seed: int = 0


# ----------------------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------------------
def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def cn_const(n):
    m = (n - 1) // 2
    return (2 / math.sqrt(math.pi)) * (-2) ** m * _dfact(n - 2) / math.sqrt(math.factorial(n))


def readin_features(W, Q, ks):
    """c_n(w0, q_perp) = <phi_n | erf(w.x)>, eq. (3). Returns (|S|, *W.shape)."""
    T = 1 + 2 * W * W + 2 * Q
    return torch.stack([cn_const(n) * W ** n / T ** (n / 2) for n in ks])


def jxx(q):
    return (2 / math.pi) * torch.asin(2 * q / (1 + 2 * q))


def hermite_norm(ks, x):
    """phi_n(x) = He_n(x)/sqrt(n!) for n in ks, x a 1-D tensor. Returns (|S|, len(x))."""
    nmax = max(ks)
    H = [torch.ones_like(x), x.clone()]
    for n in range(1, nmax):
        H.append(x * H[n] - n * H[n - 1])
    return torch.stack([H[n] / math.sqrt(math.factorial(n)) for n in ks])


def trapz_weights(x):
    dx = x[1:] - x[:-1]
    w = torch.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


class BasinLost(RuntimeError):
    pass


def first_max(nlp):
    """Index of the first local maximum of a 1-D profile scanned outward from index 0, or None."""
    v = nlp.detach().cpu().numpy()
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            return i
    return None


# ----------------------------------------------------------------------------------------------------------
# read-in layer
# ----------------------------------------------------------------------------------------------------------
class ReadIn:
    def __init__(self, p: Params, nu: Numerics, dev):
        self.p, self.nu, self.dev = p, nu, dev
        d = p.d
        self.q0 = p.sigw ** 2 * (d - 1) / d
        sq = math.sqrt(2 / max(d - 3, 1))
        self.lo = math.log(self.q0) - max(nu.nsig_q * sq, nu.qlo)
        self.hi = math.log(self.q0) + nu.nsig_q * sq
        self.fine = self._grid(nu.nw, nu.nq)
        self.coarse = self._grid(nu.nwc, nu.nqc)
        x, w = roots_jacobi(nu.nu, (d - 4) / 2, (d - 4) / 2)
        self.u = torch.tensor(x, dtype=DT, device=dev)
        self.wu = torch.tensor(w / w.sum(), dtype=DT, device=dev)

    def _grid(self, nw, nq):
        p, nu, dev = self.p, self.nu, self.dev
        w = nu.wa * torch.sinh(torch.linspace(0, math.asinh(nu.wmax / nu.wa), nw, dtype=DT, device=dev))
        q = torch.exp(torch.linspace(self.lo, self.hi, nq, dtype=DT, device=dev))
        W, Q = torch.meshgrid(w, q, indexing="ij")
        C = readin_features(W, Q, p.ks)
        g = dict(w=w, q=q, W=W, Q=Q, C=C,
                 base=p.beta * (W * W + Q) - 0.5 * (p.d - 3) * torch.log(Q),
                 Jfull=jxx(W * W + Q),
                 logm=torch.log(torch.outer(trapz_weights(w), trapz_weights(q))),
                 tw=trapz_weights(w))
        g["Pperp"] = g["Jfull"] - (C * C).sum(0)
        return g

    def logp(self, g, Ahat, Linv, mperp):
        """-S_W + log(measure) on grid g. Ahat = rB (E zz^T - E_0 zz^T) in whitened middle coordinates."""
        S = g["base"]
        if Ahat is not None:
            ch = torch.einsum("ij,jab->iab", Linv, g["C"])
            S = S - 0.5 * torch.einsum("iab,ij,jab->ab", ch, Ahat, ch)
        if mperp:
            S = S - 0.5 * mperp * g["Pperp"]
        return -S + g["logm"]

    def solve(self, Ahat=None, Linv=None, mperp=0.0, with_M=True):
        g = self.fine
        L = self.logp(g, Ahat, Linv, mperp)
        lw = torch.logsumexp(L, 1)                                  # marginal mass per w0 node
        nlp = -(lw - torch.log(g["tw"]))                            # -log density in w0
        nlp = nlp - nlp[0]
        icut = first_max(nlp) if self.nu.branch == "central" else None
        nwr = (icut + 1) if icut is not None else len(g["w"])
        Lr = L[:nwr]
        p = torch.exp(Lr - torch.logsumexp(Lr.reshape(-1), 0))
        C = g["C"][:, :nwr]
        J = torch.einsum("iab,jab,ab->ij", C, C, p)
        out = dict(J=J, EJxx=(p * g["Jfull"][:nwr]).sum(), w0sq=(p * g["W"][:nwr] ** 2).sum(),
                   qperp=(p * g["Q"][:nwr]).sum() / self.q0, nlp=nlp, w=g["w"],
                   wcut=float(g["w"][icut]) if icut is not None else None,
                   barrier=float(nlp[icut]) if icut is not None else None)
        if icut is not None:                                         # outer well (beyond the cut)
            j = icut + int(torch.argmin(nlp[icut:]))
            out.update(w_outer=float(g["w"][j]), nlp_outer=float(nlp[j]))
        if with_M:
            out["EJ2"], out["trJS2_c"] = self.pair_trace(Ahat, Linv, mperp, out["wcut"])
        return out

    def pair_trace(self, Ahat, Linv, mperp, wcut, chunk=256):
        """E_{x,x'} J(x,x')^2 (eq. 18) and |J_S|_F^2, both on the coarse grid for consistency."""
        g = self.coarse
        L = self.logp(g, Ahat, Linv, mperp)
        if wcut is not None:
            L = torch.where((g["W"] <= wcut), L, torch.full_like(L, -torch.inf))
        p = torch.exp(L - torch.logsumexp(L.reshape(-1), 0)).reshape(-1)
        keep = p > 1e-14 * p.max()
        p = p[keep] / p[keep].sum()
        w0 = g["W"].reshape(-1)[keep]
        q = g["Q"].reshape(-1)[keep]
        C = g["C"].reshape(len(self.p.ks), -1)[:, keep]
        Jc = (C * p) @ C.T
        T = 1 + 2 * (w0 * w0 + q)
        sq = torch.sqrt(q)
        acc = torch.zeros((), dtype=DT, device=self.dev)
        for i in range(0, len(p), chunk):
            sl = slice(i, i + chunk)
            a = 2 * torch.outer(w0[sl], w0)
            b = 2 * torch.outer(sq[sl], sq)
            den = torch.sqrt(torch.outer(T[sl], T))
            z = (a[..., None] + b[..., None] * self.u) / den[..., None]
            k2 = ((2 / math.pi) * torch.asin(z)) ** 2 @ self.wu
            acc = acc + (p[sl, None] * p[None, :] * k2).sum()
        return acc, (Jc * Jc).sum()


# ----------------------------------------------------------------------------------------------------------
# middle layer
# ----------------------------------------------------------------------------------------------------------
class Middle:
    def __init__(self, p: Params, nu: Numerics, dev):
        self.p, self.nu, self.dev = p, nu, dev
        x, w = hermegauss(nu.nx)
        self.x = torch.tensor(x, dtype=DT, device=dev)
        self.wx = torch.tensor(w / math.sqrt(2 * math.pi), dtype=DT, device=dev)
        self.Phi = hermite_norm(p.ks, self.x)                       # (S, nx)
        self.WPhi = (self.Phi * self.wx).T.contiguous()              # (nx, S)
        r, wr = hermegauss(nu.nqh)
        self.r = torch.tensor(r, dtype=DT, device=dev)
        self.logwr = torch.tensor(np.log(wr), dtype=DT, device=dev)
        self.counts = None
        if nu.zeta_rule == "qmc" and p.nS - 1 >= 1:
            k = p.nS - 1
            eng = torch.quasirandom.SobolEngine(k, scramble=True, seed=nu.seed)
            uu = eng.draw(nu.nzeta_qmc // 2, dtype=DT).clamp(1e-12, 1 - 1e-12)
            zh = math.sqrt(2) * torch.erfinv(2 * uu - 1)
            self._set_zeta(torch.cat([zh, -zh]), torch.full((2 * len(zh),), 1.0 / (2 * len(zh)), dtype=DT))
        elif p.nS == 1:
            self._set_zeta(torch.zeros(1, 0, dtype=DT), torch.ones(1, dtype=DT))
        self.recount = True
        self.Kmax = int(math.ceil(nu.smax / nu.ds))
        self.Kcur = self.Kmax

    def _set_zeta(self, zeta, wz):
        self.zeta = zeta.to(self.dev)
        self.wz = wz.to(self.dev)
        self.logwz = torch.log(self.wz)

    def _product_rule(self, counts):
        """Tensor Gauss-Hermite rule with counts[i] nodes along transverse direction i."""
        zs, ws = [], []
        for n in counts:
            gz, gw = hermegauss(n)
            zs.append(torch.tensor(gz, dtype=DT)); ws.append(torch.tensor(gw / gw.sum(), dtype=DT))
        Z = torch.meshgrid(*zs, indexing="ij")
        W = torch.meshgrid(*ws, indexing="ij")
        zeta = torch.stack([z.reshape(-1) for z in Z], 1)
        wz = torch.ones(len(zeta), dtype=DT)
        for w in W:
            wz = wz * w.reshape(-1)
        return zeta, wz

    # geometry for one inner solve -------------------------------------------------------------------------
    def setup(self, L, t_dir, qbar, M):
        p, dev, nu = self.p, self.dev, self.nu
        v = L.T @ t_dir
        if float(v.norm()) == 0.0:
            v = torch.zeros(p.nS, dtype=DT, device=dev); v[0] = 1.0
        u = v / v.norm()
        Qfull, _ = torch.linalg.qr(torch.cat([u[:, None], torch.eye(p.nS, dtype=DT, device=dev)], 1))
        Qp = Qfull[:, 1:p.nS]
        self.a = (L @ u) @ self.Phi                                   # (nx,) arg per unit s
        if nu.zeta_rule == "auto" and p.nS > 1:
            # rotate the transverse block to the singular directions of zeta -> erf argument and give each
            # direction Gauss-Hermite nodes by its sensitivity relative to the s-direction
            sw = torch.sqrt(self.wx)
            U, sv, _ = torch.linalg.svd(((L @ Qp).T @ self.Phi) * sw, full_matrices=False)   # U: (k, k)
            Qp = Qp @ U
            ratio = (sv / (self.a * sw).norm()).cpu().numpy()
            if self.recount or self.counts is None:
                counts = []
                for rt in ratio:
                    n = next(n for thr, n in nu.zeta_levels if rt >= thr)
                    counts.append(n)
                if counts != self.counts:
                    self.counts = counts
                    self._set_zeta(*self._product_rule(counts))
                self.recount = False
            self.ratio = ratio
        self.u, self.Qp, self.L = u, Qp, L
        self.A = self.zeta @ ((L @ Qp).T @ self.Phi)                  # (nz, nx)
        k = M / 2
        sig = 1 / math.sqrt(k)
        self.qh = qbar * torch.exp(sig * self.r)
        lw = self.logwr + 0.5 * self.r ** 2 + k * (torch.log(self.qh / qbar) - self.qh / qbar)
        self.logwq = lw - torch.logsumexp(lw, 0)
        self.rq = 1 / torch.sqrt(1 + 2 * self.qh)
        self.qbar, self.M = qbar, M
        # prior zeta moments (exact on the node set)
        self.Zz2 = torch.einsum("z,zi,zj->ij", self.wz, self.zeta, self.zeta)
        self.zz = (self.zeta[:, :, None] * self.zeta[:, None, :]).reshape(len(self.zeta), -1)
        self.pw = torch.exp(self.logwq[:, None] + self.logwz[None, :]).view(1, -1)

    # one integral pass at fixed t ------------------------------------------------------------------------
    def integrate(self, t):
        p, nu = self.p, self.nu
        S, nz, nqh, nx = p.nS, len(self.zeta), len(self.qh), nu.nx
        K = self.Kcur
        s_all = nu.ds * torch.arange(0, K + 1, dtype=DT, device=self.dev)
        per = max(1, nu.chunk // (nqh * nz * nx))
        logw_qz = self.logwq[:, None] + self.logwz[None, :]           # (nq, nz)
        out = {k: [] for k in ("m", "Z", "BB", "tBB", "t2BB", "q", "e1", "zz2", "ess")}
        for i0 in range(0, K + 1, per):
            s = s_all[i0:i0 + per]
            ns = len(s)
            arg = (s[:, None, None] * self.a + self.A[None])[:, None] * self.rq[None, :, None, None]
            arg.erf_()
            B = (arg @ self.WPhi).view(ns, nqh * nz, S)               # (ns, nq*nz, S)
            tB = B @ t
            lt = (0.5 * p.gt) * tB * tB                               # tilt
            lw = logw_qz.view(1, -1) + lt - 0.5 * (s * s)[:, None]
            m = lw.amax(1)
            e = torch.exp(lw - m[:, None])
            et = e * tB
            out["m"].append(m)
            out["Z"].append(e.sum(1))
            # tiny Gram matrices with long reductions: broadcast-and-sum beats (tensor-core) bmm by ~20x
            BB2 = B[..., :, None] * B[..., None, :]                   # (ns, nq*nz, S, S)
            out["BB"].append((BB2 * e[..., None, None]).sum(1))
            out["tBB"].append((B * et[..., None]).sum(1))
            out["t2BB"].append((BB2 * (et * tB)[..., None, None]).sum(1))
            e3 = e.view(ns, nqh, nz)
            out["q"].append(e3.sum(2) @ self.qh)
            ez = e3.sum(1)                                            # (ns, nz)
            out["e1"].append(ez @ self.zeta)
            out["zz2"].append((ez @ self.zz).view(ns, S - 1, S - 1))
            # Kish ESS of the tilt over (q, zeta), relative to the prior weights
            r = torch.exp(lt - lt.amax(1, keepdim=True))
            out["ess"].append((self.pw * r).sum(1) ** 2 / (self.pw * r * r).sum(1))
        o = {k: torch.cat(v) for k, v in out.items()}
        # s-marginal and the basin cut
        logZs = o["m"] + torch.log(o["Z"])                            # density in s (no lattice weight)
        nlp = -(logZs - logZs[0])
        icut = first_max(nlp) if nu.branch == "central" else None
        n = (icut + 1) if icut is not None else K + 1
        ws = torch.full((n,), 2 * nu.ds, dtype=DT, device=self.dev)
        ws[0] = nu.ds
        Ls = torch.log(ws) + logZs[:n]
        logZ = torch.logsumexp(Ls, 0)
        pi = torch.exp(Ls - logZ)
        f = pi / o["Z"][:n]
        avg = lambda X: torch.einsum("s,s...->...", f, X[:n])
        Kmat = avg(o["BB"])
        Ktb = avg(o["tBB"])
        Cov = avg(o["t2BB"]) - torch.outer(Ktb, Ktb)
        sN = s_all[:n]
        u, Qp = self.u, self.Qp
        # z = s u + Qp zeta. E[zeta] = 0 by the h -> -h symmetry, but E[s zeta] is not:
        cross = Qp @ avg(sN[:, None] * o["e1"][:n])
        Ezz = ((pi @ (sN ** 2)) * torch.outer(u, u) + torch.outer(u, cross) + torch.outer(cross, u)
               + Qp @ avg(o["zz2"]) @ Qp.T)
        # prior on the same nodes and region (control variate)
        lp = torch.log(ws) - 0.5 * sN ** 2
        pi0 = torch.exp(lp - torch.logsumexp(lp, 0))
        Ezz0 = (pi0 @ (sN ** 2)) * torch.outer(u, u) + Qp @ self.Zz2 @ Qp.T
        Eq = avg(o["q"])
        Eq0 = torch.exp(self.logwq) @ self.qh
        ess = float((pi * o["ess"][:n]).sum())
        return dict(logZ=logZ, K=Kmat, Cov=Cov, Ezz=Ezz, Ezz0=Ezz0, Eq=Eq, Eq0=Eq0,
                    s=s_all, nlp=nlp, icut=icut, ess=ess,
                    scut=float(s_all[icut]) if icut is not None else None,
                    barrier=float(nlp[icut]) if icut is not None else None)

    def set_range(self, scut):
        if self.nu.branch == "central" and scut is not None:
            self.Kcur = min(self.Kmax, int(math.ceil((self.nu.smargin * scut + 3) / self.nu.ds)))
        else:
            self.Kcur = self.Kmax


# ----------------------------------------------------------------------------------------------------------
# inner Newton on t
# ----------------------------------------------------------------------------------------------------------
def solve_t(mid: Middle, p: Params, L, qbar, M, t0, tol, maxit, log=None):
    dev = mid.dev
    y = torch.tensor(p.y(), dtype=DT, device=dev)
    I = torch.eye(p.nS, dtype=DT, device=dev)
    t = t0.clone()
    mid.setup(L, t if float(t.norm()) > 0 else y, qbar, M)
    st = mid.integrate(t)
    if st["scut"] is None and mid.Kcur < mid.Kmax:                   # barrier may lie beyond a shortened range
        mid.Kcur = mid.Kmax
        st = mid.integrate(t)
    mid.set_range(st["scut"])                                        # range for later passes
    c = p.sigA ** 2 / p.gt
    phi = lambda t, st: 0.5 * p.lam * (t @ t) + c * st["logZ"] - y @ t
    npass = 1
    for it in range(maxit):
        grad = p.lam * t + p.sigA ** 2 * (st["K"] @ t) - y
        H = p.lam * I + p.sigA ** 2 * (st["K"] + p.gt * st["Cov"])
        step = -torch.linalg.solve(H, grad)
        rel = float(step.norm() / t.norm().clamp_min(1e-300))
        if rel < tol:
            break
        f0, slope = phi(t, st), float(grad @ step)
        a = 1.0
        while True:
            tn = t + a * step
            stn = mid.integrate(tn); npass += 1
            if stn["scut"] is None and mid.Kcur < mid.Kmax:
                mid.Kcur = mid.Kmax
                stn = mid.integrate(tn); npass += 1
            fn = phi(tn, stn)
            if fn <= f0 + 1e-4 * a * slope or abs(slope) < 1e-14 * abs(float(f0)) or a < 1e-8:
                break
            a *= 0.5
        t, st = tn, stn
        if log:
            log(f"    newton {it}: |dt|/|t|={rel:.2e} step={a:g} scut={st['scut']}")
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
        ri0 = self.readin.solve()                                    # prior read-in law
        self.J0 = ri0["J"]
        self.Jsc = torch.sqrt(torch.outer(torch.diag(self.J0), torch.diag(self.J0)))
        self.iu = torch.triu_indices(p.nS, p.nS, device=self.dev)
        self.state0 = self._state_from_readin(ri0)
        self.t = torch.zeros(p.nS, dtype=DT, device=self.dev)
        self.last = None

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    def _state_from_readin(self, ri):
        p = self.p
        qbar = p.sigB ** 2 * (ri["EJxx"] - torch.trace(ri["J"]))
        trG2 = p.sigB ** 4 * (ri["EJ2"] - ri["trJS2_c"])
        return dict(J=ri["J"], qbar=float(qbar), M=float(qbar ** 2 / trG2))

    def encode(self, s):
        Jn = (s["J"] / self.Jsc)[self.iu[0], self.iu[1]]
        return torch.cat([Jn, torch.tensor([math.log(s["qbar"]), math.log(s["M"])], dtype=DT, device=self.dev)])

    def decode(self, x):
        nS = self.p.nS
        Jn = torch.zeros(nS, nS, dtype=DT, device=self.dev)
        Jn[self.iu[0], self.iu[1]] = x[:-2]
        Jn = Jn + Jn.T - torch.diag(torch.diag(Jn))
        return dict(J=Jn * self.Jsc, qbar=math.exp(float(x[-2])), M=math.exp(float(x[-1])))

    def F(self, s):
        """One outer map: read-in state -> middle law -> back-coupling -> new read-in state."""
        p = self.p
        G = p.sigB ** 2 * s["J"]
        L = torch.linalg.cholesky(G)
        t, st, npass = solve_t(self.mid, p, L, s["qbar"], s["M"], self.t,
                               self.nu.newton_tol, self.nu.newton_maxit)
        self.npass += npass
        Ahat = p.rB * (st["Ezz"] - st["Ezz0"])
        dq = float((st["Eq"] - st["Eq0"]) / s["qbar"])
        mperp = p.rB * dq * s["M"] / s["qbar"]
        Linv = torch.linalg.solve_triangular(L, torch.eye(p.nS, dtype=DT, device=self.dev), upper=False)
        ri = self.readin.solve(Ahat, Linv, mperp)
        if self.nu.branch == "central":
            # the origin must be a basin of both actions (a profile that only falls has no central branch)
            if st["scut"] is None and float(st["nlp"].min()) < -1.0:
                raise BasinLost("middle action has no central basin")
            if ri["wcut"] is None and float(ri["nlp"].min()) < -1.0:
                raise BasinLost("read-in action has no central basin")
        self.t = t
        self.last = dict(state=s, t=t, mid=st, ri=ri, L=L, Linv=Linv, Ahat=Ahat, mperp=mperp, dq=dq)
        return self._state_from_readin(ri)

    def _anderson(self, s, label="", tol=None):
        """Anderson-accelerated fixed point from state s at the current parameters. Raises BasinLost."""
        nu = self.nu
        tol = tol or nu.outer_tol
        t0 = time.time()
        x = self.encode(s)
        X, Fh = [], []
        res, fails = float("inf"), 0
        self.mid.recount = True                            # transverse node counts fixed within one run
        for k in range(nu.outer_maxit):
            try:
                gx = self.encode(self.F(self.decode(x)))
            except (torch.linalg.LinAlgError, BasinLost):
                fails += 1
                if not X or fails > 3:
                    raise BasinLost("outer iteration left the central branch")
                x = X[-1] + 0.5 * Fh[-1]                   # extrapolation failed: damped step, reset history
                X, Fh = [], []
                continue
            f = gx - x
            res = float(f.abs().max())
            self.log(f"  {label}outer {k:2d}: res={res:.2e}  t={np.array2string(self.t.cpu().numpy(), precision=1)}"
                     f"  scut={self.last['mid']['scut']}  wcut={self.last['ri']['wcut']}"
                     f"  ({time.time() - t0:.2f}s)")
            if res < tol:
                return self.decode(gx), res
            X.append(x.clone()); Fh.append(f.clone())
            if len(X) > nu.anderson_m + 1:
                X.pop(0); Fh.pop(0)
            if len(X) >= 2:
                dX = torch.stack([X[i + 1] - X[i] for i in range(len(X) - 1)], 1).cpu()
                dF = torch.stack([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)], 1).cpu()
                gam = torch.linalg.lstsq(dF, f.cpu()[:, None], rcond=1e-12, driver="gelsd").solution[:, 0]
                x = x + f - ((dX + dF) @ gam).to(self.dev)
            else:
                x = x + f
        raise BasinLost(f"outer iteration did not converge (res={res:.2e})")

    def solve(self, init=None, chi_start=None):
        """Solve at self.p. From a warm start (init = another solver's .state()), Anderson directly.
        Otherwise continuation in chi (M_S is proportional to chi), with an adaptive step in log chi."""
        t0 = time.time()
        self.npass = 0
        chi_t = self.p.chi
        if init is not None:
            self.t = init["t"].to(self.dev)
            try:
                s, res = self._anderson(dict(J=init["J"].to(self.dev), qbar=init["qbar"], M=init["M"]))
                return self._finish(res, t0)
            except BasinLost as e:
                self.log(f"  warm start failed ({e}); falling back to chi continuation")
                self.t = torch.zeros_like(self.t)
        lc, lt = math.log(chi_start or chi_t / 64), math.log(chi_t)
        step = math.log(4.0)
        self.p.chi = math.exp(lc)
        loose = max(self.nu.cont_tol, self.nu.outer_tol)
        s, res = self._anderson(self.state0, f"chi={self.p.chi:.4g} ", loose)
        while lc < lt - 1e-12:
            nxt = min(lc + step, lt)
            self.p.chi = math.exp(nxt)
            tsave = self.t.clone()
            try:
                s, res = self._anderson(s, f"chi={self.p.chi:.4g} ", None if nxt >= lt - 1e-12 else loose)
                lc = nxt
                step *= 1.5
            except BasinLost as e:
                self.log(f"  chi={self.p.chi:.4g}: {e}; halving the step")
                self.t = tsave
                step /= 2
                if step < 1e-3:
                    self.p.chi = chi_t
                    raise
        self.p.chi = chi_t
        return self._finish(res, t0)

    def _finish(self, res, t0):
        self.converged = res < self.nu.outer_tol
        self.walltime = time.time() - t0
        return self.results()

    def results(self):
        p, lt = self.p, self.last
        st, ri, L = lt["mid"], lt["ri"], lt["L"]
        G = p.sigB ** 2 * lt["state"]["J"]
        H = G + L @ (st["Ezz"] - st["Ezz0"]) @ L.T
        K = st["K"]
        t = lt["t"]
        fbar = p.sigA ** 2 * (K @ t)
        y = p.y()
        MS = lt["Linv"].T @ lt["Ahat"] @ lt["Linv"]
        ev = torch.linalg.eigvalsh(MS).cpu().numpy()
        npy = lambda a: a.detach().cpu().numpy().tolist()
        out = dict(params=asdict(p), branch=self.nu.branch, converged=bool(self.converged),
                   walltime_s=self.walltime, middle_passes=self.npass,
                   d_w0sq=float(p.d * ri["w0sq"]), qperp_over_prior=float(ri["qperp"]),
                   J=npy(ri["J"]), H=npy(H), K=npy(K), G=npy(G), t=npy(t), fbar=npy(fbar),
                   L1=float(fbar[0]), L3=float(fbar[1] / p.eps) if p.nS > 1 else None,
                   qh_over_qbar=float(st["Eq"] / st["Eq0"]), qbar=lt["state"]["qbar"], M=lt["state"]["M"],
                   m_perp=lt["mperp"], MS_eig=ev.tolist(),
                   readin_barrier=ri["barrier"], readin_wcut=ri["wcut"],
                   readin_w_outer=ri.get("w_outer"), readin_nlp_outer=ri.get("nlp_outer"),
                   middle_barrier=st["barrier"], middle_scut=st["scut"],
                   middle_hcut=npy(L @ self.mid.u * st["scut"]) if st["scut"] is not None else None,
                   middle_ess=st["ess"])
        return out

    def state(self):
        s = self.last["state"]
        return dict(J=s["J"], qbar=s["qbar"], M=s["M"], t=self.last["t"])

    def middle_profile(self, smax=None):
        """Full s-marginal of the middle action at the solution (both wells), for plots."""
        mid, nu = self.mid, self.nu
        keep = (mid.Kcur, nu.branch)
        mid.Kcur = int(math.ceil((smax or nu.smax) / nu.ds)); nu.branch = "global"
        st = mid.integrate(self.last["t"])
        mid.Kcur, nu.branch = keep
        h = (self.last["L"] @ mid.u)[None, :] * st["s"][:, None]
        return st["s"].cpu().numpy(), h.cpu().numpy(), st["nlp"].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for f_, v in asdict(Params()).items():
        ap.add_argument(f"--{f_}", type=type(v), default=v)
    for f_ in ("branch", "zeta_rule", "nx", "nqh", "nzeta_qmc", "ds", "smax", "nw", "nq"):
        v = getattr(Numerics(), f_)
        ap.add_argument(f"--{f_}", type=type(v), default=v)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep", default=None, help="param=v1,v2,... (warm-started continuation)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    a = ap.parse_args()
    pk = {k: getattr(a, k) for k in asdict(Params())}
    nk = {k: getattr(a, k) for k in ("branch", "zeta_rule", "nx", "nqh", "nzeta_qmc", "ds", "smax", "nw", "nq")}
    runs = [dict()]
    if a.sweep:
        name, vals = a.sweep.split("=")
        typ = type(pk[name])
        runs = [{name: typ(v)} for v in vals.split(",")]
    results, init = [], None
    for r in runs:
        p = Params(**{**pk, **r})
        print(f"== {r or 'single point'}  S={p.ks}")
        S = FCN3Solver(p, Numerics(**nk), a.device)
        res = S.solve(init)
        init = S.state()
        results.append(res)
        print(json.dumps({k: res[k] for k in ("converged", "walltime_s", "middle_passes", "d_w0sq", "L1", "L3",
                                               "qh_over_qbar", "M", "m_perp", "readin_barrier",
                                               "middle_barrier", "middle_ess")}, indent=1))
        print("H =", np.array(res["H"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
