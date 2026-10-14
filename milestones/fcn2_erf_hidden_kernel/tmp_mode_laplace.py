"""TMP: Laplace theory in a finite mode space, with the finite-P sample couplings.

Modes psi_a(x) (population-orthonormal, 154): He1,He3,He5,He7 of x0; x_i (15); x0 x_i x_j, i<j (105); x0 He2(x_i)/sqrt2 (15);
He2(x0) x_i/sqrt2 (15). Neuron overlaps c_a(w) = <erf(w.x) psi_a> are analytic (Stein): E[erf^(k)(u)] w^alpha.
Everything outside the modes acts as a ridge: kappa' = kappa + sig_A^2 N E_w[<phi^2> - |c|^2].
Finite P enters only through the training-set mode Gram S = Psi^T Psi / P (S = I off data):
    b(w) = c^T (S Lam + rho')^-1 S y_m,   x(w) = sig_A^2 c^T (S Lam + rho')^-1 S c   (outside modes: ridge only),
    Lam = sig_A^2 N E_w[c c^T], rho' = kappa'/P, y_m = (1, eps, 0, ...).
    U(w) = d/(2 s0)|w|^2 + 1/2 log(1+x) - sig_A^2 b^2 / (2(1+x)).
    training discrepancy = (kappa/P) [(S Lam + rho')^-1 S y_m]_{He1,He3};  population = [y_m - Lam (S Lam + rho')^-1 S y_m]_{He1,He3}.
Laplace per well (bulk, specialists; fraction fixed F_S; U(-w) = U(w)): w0 from the perp-averaged V(w0) (minimum, curvature);
perp mean m and covariance Sig from the Gaussian stationarity conditions E[grad_perp U] = 0, Sig^-1 = E[grad_perp^2 U] (Stein).
S choices: pop (S = I, off data), synth (fresh Gaussian samples of size P, quenched average), data (the actual training set).
Usage: tmp_mode_laplace.py check | pop | synth | data | all   [ITERS=100] [NSYN=4]"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

from psi_mode_laplace_steepwell import ROOT
from train_discrepancy_two_pop import training_set

WHAT = sys.argv[1] if len(sys.argv) > 1 else "all"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NSYN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
DEV, DT = "cuda:1", torch.float64
M, F_S, N_PERP, N_GRID, DAMP = 4000, 0.095, 64, 61, 0.3
FIX_M = os.environ.get("FIX_M", "0") == "1"
M_DAMP = float(os.environ.get("M_DAMP", "1.0"))
FIX_TRACE = os.environ.get("FIX_TRACE", "0") == "1"
F_S = float(os.environ.get("F_S", F_S))
_D = int(os.environ.get("D", "16"))
run = [r for r in glob.glob(os.path.join(ROOT, "invariant_beta*")) if json.load(open(os.path.join(r, "config.json")))["d"] == _D][0]
c = json.load(open(os.path.join(run, "config.json")))
d, P, N, eps, s0 = c["d"], c["P"], c["N"], c["eps"], c["s0"]
sig_a2, kappa = c["sa0"] / N, c["temperature"] / 2
chi = float(os.environ.get("CHI", str(c.get("chi", 1.0))))
P_ORIG = P
P = int(os.environ.get("P_OVERRIDE", P))
if os.environ.get("KAPPA_SCALE", "0") == "1":
    kappa = kappa * P / P_ORIG
DATA_SEED = int(os.environ.get("DATA_SEED", "500"))
rec = json.load(open(os.path.join(run, "pred_vs_true.json")))
y_np = np.asarray(rec[max(rec, key=int)]["y_true"], dtype=float)
X_np = training_set(P_ORIG, d, y_np, eps)[0]
iu = torch.triu_indices(d - 1, d - 1, offset=1, device=DEV)
NM = 4 + (d - 1) + iu.shape[1] + 2 * (d - 1)
MODES3 = os.environ.get("MODES3", "0") == "1"
if MODES3:
    _t = torch.combinations(torch.arange(d - 1, device=DEV), 3)
    _pi, _pj = torch.meshgrid(torch.arange(d - 1, device=DEV), torch.arange(d - 1, device=DEV), indexing="ij")
    _off = _pi != _pj
    i3, pi_, pj_ = _t.T, _pi[_off], _pj[_off]
    NM += _t.shape[0] + int(_off.sum()) + (d - 1)
ym = torch.zeros(NM, device=DEV, dtype=DT); ym[0], ym[1] = 1.0, eps
r2 = math.sqrt(2.0)


def Ek(k, D):
    return (2 / math.sqrt(math.pi)) * (-1) ** ((k - 1) // 2) * math.factorial(k - 1) / math.factorial((k - 1) // 2) / D ** (k / 2)


def modes(X):
    z, xp = X[:, 0], X[:, 1:]
    he = torch.stack([z, (z ** 3 - 3 * z) / math.sqrt(6), (z ** 5 - 10 * z ** 3 + 15 * z) / math.sqrt(120),
                      (z ** 7 - 21 * z ** 5 + 105 * z ** 3 - 105 * z) / math.sqrt(5040)], 1)
    out = [he, xp, z[:, None] * xp[:, iu[0]] * xp[:, iu[1]], z[:, None] * (xp ** 2 - 1) / r2, (z ** 2 - 1)[:, None] * xp / r2]
    if MODES3:
        out += [xp[:, i3[0]] * xp[:, i3[1]] * xp[:, i3[2]], xp[:, pi_] * (xp[:, pj_] ** 2 - 1) / r2,
                (xp ** 3 - 3 * xp) / math.sqrt(6)]
    return torch.cat(out, 1)


def coeffs(W):
    n2 = (W ** 2).sum(1); D = 1 + 2 * n2; w0, wp = W[:, 0], W[:, 1:]
    E1, E3, E5, E7 = Ek(1, D), Ek(3, D), Ek(5, D), Ek(7, D)
    he = torch.stack([E1 * w0, E3 * w0 ** 3 / math.sqrt(6), E5 * w0 ** 5 / math.sqrt(120), E7 * w0 ** 7 / math.sqrt(5040)], 1)
    out = [he, E1[:, None] * wp, (E3 * w0)[:, None] * wp[:, iu[0]] * wp[:, iu[1]], (E3 * w0)[:, None] * wp ** 2 / r2,
           (E3 * w0 ** 2)[:, None] * wp / r2]
    if MODES3:
        out += [E3[:, None] * wp[:, i3[0]] * wp[:, i3[1]] * wp[:, i3[2]], E3[:, None] * wp[:, pi_] * wp[:, pj_] ** 2 / r2,
                E3[:, None] * wp ** 3 / math.sqrt(6)]
    C = torch.cat(out, 1)
    return C, (2 / math.pi) * torch.asin(2 * n2 / D)


def check():
    g = torch.Generator(device="cpu").manual_seed(0)
    W = torch.randn(4, d, generator=g, dtype=DT).to(DEV) * 0.3; W[3, 0] = 1.4
    acc, acc2, n = 0, 0, 0
    for _ in range(40):
        X = torch.randn(50000, d, generator=g, dtype=DT).to(DEV)
        F = torch.erf(X @ W.T)
        acc = acc + F.T @ modes(X); acc2 = acc2 + (F ** 2).sum(0); n += len(X)
    C, p2 = coeffs(W)
    print(f"check ({n} MC points): max |c_MC - c_analytic| = {float((acc / n - C).abs().max()):.4f} "
          f"(max |c| {float(C.abs().max()):.3f}); <phi^2> error {float((acc2 / n - p2).abs().max()):.4f}", flush=True)


SELF = os.environ.get("SELF", "span")


class Theory:
    """SELF=span: self-overlap from the modes only. SELF=full: plus the outside-mode part from the population resolvent,
    sig_A^2 [<phi|[K + kappa_eff/P]^-1|phi> - c^T (Lam + kappa_eff/P)^-1 c] (arcsin kernel, kappa_eff from eq. 2.8)."""

    def __init__(self, S):
        self.S = S
        self.I = torch.eye(NM, device=DEV, dtype=DT)

    def setup(self, W):
        C, p2 = coeffs(W)
        self.Lam = sig_a2 * (N / len(W)) * C.T @ C
        self.kp = kappa + float(sig_a2 * (N / len(W)) * (p2 - (C ** 2).sum(1)).sum())
        self.rho = self.kp / (chi * P)
        self.A = torch.linalg.solve(self.S @ self.Lam + self.rho * self.I, self.S)
        self.v = self.A @ ym
        if SELF == "full":
            from tmp_population_theory import gram, kappa_eff_from
            s = sig_a2 * N / len(W)
            G = gram(W, W)
            self.keff, _ = kappa_eff_from(G, s, kappa, P)
            re = self.keff / P
            self.Wref, self.re = W.detach().clone(), re
            self.sh = torch.full((len(W),), math.sqrt(s), device=DEV, dtype=DT)
            self.L = torch.linalg.cholesky(re * torch.eye(len(W), device=DEV, dtype=DT) + s * G)
            self.Ainv = torch.linalg.inv(self.Lam + re * self.I)

    def setup_w(self, W, sw):
        """Weighted cloud: neuron i carries kernel weight sw_i (sum sw = sig_A^2 N)."""
        from tmp_population_theory import gram
        from scipy.optimize import brentq
        C, p2 = coeffs(W)
        self.Lam = (C * sw[:, None]).T @ C
        self.kp = kappa + float((sw * (p2 - (C ** 2).sum(1))).sum())
        self.rho = self.kp / (chi * P)
        self.A = torch.linalg.solve(self.S @ self.Lam + self.rho * self.I, self.S)
        self.v = self.A @ ym
        if SELF != "full":
            return
        sh = torch.sqrt(sw)
        G = sh[:, None] * gram(W, W) * sh[None, :]
        mu = torch.linalg.eigvalsh(G).clamp(min=0).cpu().numpy()
        self.keff = brentq(lambda k: k - kappa - (k / P) * np.sum(mu / (mu + k / P)), kappa, 1e4 * kappa)
        self.re = self.keff / P
        self.Wref, self.sh = W.detach().clone(), sh
        self.L = torch.linalg.cholesky(self.re * torch.eye(len(W), device=DEV, dtype=DT) + G)
        self.Ainv = torch.linalg.inv(self.Lam + self.re * self.I)

    def x_out(self, W, C):
        from tmp_population_theory import gram
        Z = torch.linalg.solve_triangular(self.L, self.sh[:, None] * gram(self.Wref, W), upper=False)
        n2 = (W ** 2).sum(1)
        Bww = ((2 / math.pi) * torch.asin(2 * n2 / (1 + 2 * n2)) - (Z ** 2).sum(0)) / self.re
        return sig_a2 * (Bww - ((C @ self.Ainv) * C).sum(1))

    def U(self, W):
        C, p2 = coeffs(W)
        x = sig_a2 * ((C @ self.A.T) * C).sum(1)
        if SELF == "full":
            x = x + self.x_out(W, C)
        b = C @ self.v
        return (d / (2 * s0)) * (W ** 2).sum(1) + 0.5 * torch.log1p(x) - 0.5 * sig_a2 * b ** 2 / (1 + x)

    def disc(self):
        tr = (kappa / (chi * P)) * torch.linalg.solve(self.S @ self.Lam + self.rho * self.I, self.S @ ym)
        pop = ym - self.Lam @ self.v
        return tr[:2].cpu().numpy(), pop[:2].cpu().numpy()


def _well_ranges():
    """Teacher-axis search windows for bulk / specialist wells (overridable via env)."""
    return {
        "b": (float(os.environ.get("BULK_LO", "0.0")), float(os.environ.get("BULK_HI", "0.7"))),
        "s": (float(os.environ.get("SPEC_LO", "0.8")), float(os.environ.get("SPEC_HI", "2.6"))),
    }


def _well_init():
    return {
        "b": (float(os.environ.get("MU_B0", "0.3")), float(os.environ.get("SIG_B0", "0.15"))),
        "s": (float(os.environ.get("MU_S0", "1.45")), float(os.environ.get("SIG_S0", "0.25"))),
    }


def solve(S, tag, f_s=None):
    f_s = F_S if f_s is None else f_s
    th = Theory(S)
    g = torch.Generator(device="cpu").manual_seed(0)
    Ms = int(round(f_s * M)); Mb = M - Ms
    Z = {"b": torch.randn(Mb, d, generator=g, dtype=DT).to(DEV), "s": torch.randn(Ms, d, generator=g, dtype=DT).to(DEV)}
    sg = {k: torch.where(torch.rand(len(Z[k]), generator=g) < 0.5, -1.0, 1.0).to(DT).to(DEV) for k in Z}
    Zp = torch.randn(N_PERP, d - 1, generator=g, dtype=DT).to(DEV)
    rng = _well_ranges()
    init = _well_init()
    st = {k: dict(mu=mu, sig=sd, m=torch.zeros(d - 1, device=DEV, dtype=DT), Sig=torch.eye(d - 1, device=DEV, dtype=DT) * s0 / d)
          for k, (mu, sd) in init.items()}

    def well(k):
        w = st[k]; L = torch.linalg.cholesky(w["Sig"])
        return torch.cat([(w["mu"] + w["sig"] * Z[k][:, 0])[:, None], w["m"] + Z[k][:, 1:] @ L.T], 1), L

    for it in range(ITERS):
        th.setup(torch.cat([well(k)[0] * sg[k][:, None] for k in ("b", "s")], 0))
        new = {}
        for k in ("b", "s"):
            w = st[k]; Ww, L = well(k)
            lo, hi = rng[k]
            grid = torch.linspace(lo, hi, N_GRID, device=DEV, dtype=DT)
            Wg = torch.cat([grid.repeat_interleave(N_PERP)[:, None], (w["m"] + Zp @ L.T).repeat(N_GRID, 1)], 1)
            with torch.no_grad():
                V = th.U(Wg).reshape(N_GRID, N_PERP).mean(1).cpu().numpy()
            gr = grid.cpu().numpy(); i = int(np.clip(np.argmin(V), 2, N_GRID - 3))
            cf = np.polyfit(gr[i - 2:i + 3], V[i - 2:i + 3], 2)
            mu = float(np.clip(-cf[1] / (2 * cf[0]), lo, hi)) if cf[0] > 0 else gr[i]
            sd = float(1 / math.sqrt(2 * cf[0])) if cf[0] > 0 else w["sig"]
            Wv = Ww.clone().requires_grad_(True)
            gp, = torch.autograd.grad(th.U(Wv).sum(), Wv)
            gp = gp[:, 1:]
            dev_ = Ww[:, 1:] - w["m"]
            H = torch.linalg.solve(w["Sig"], dev_.T @ gp / len(Ww)); H = 0.5 * (H + H.T)
            e, Vv = torch.linalg.eigh(H)
            Sig = Vv @ torch.diag(1 / e.clamp(2.0, 100.0)) @ Vv.T
            if FIX_TRACE:
                Sig = Sig * ((d - 1) * s0 / d) / torch.trace(Sig)
            m = w["m"] * 0 if FIX_M else w["m"] - M_DAMP * w["Sig"] @ gp.mean(0)
            new[k] = dict(mu=mu, sig=sd, m=m, Sig=Sig)
        for k in st:
            for q in st[k]:
                st[k][q] = (1 - DAMP) * st[k][q] + DAMP * new[k][q]
        if (it + 1) % 10 == 0:
            tr, pop = th.disc()
            print(f"  [{tag}] iter {it + 1}: bulk mu {st['b']['mu']:.3f}, |m_perp|^2 {float((st['b']['m'] ** 2).sum()):.4f}, specialists mu {st['s']['mu']:.2f}, "
                  f"training ({tr[0]:+.4f}, {tr[1]:+.4f}), population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    Wc = torch.cat([well(k)[0] * sg[k][:, None] for k in ("b", "s")], 0)
    th.setup(Wc)
    tr, pop = th.disc()
    with torch.no_grad():
        Cc, _ = coeffs(Wc)
        xs = sig_a2 * ((Cc @ th.A.T) * Cc).sum(1)
        xo = th.x_out(Wc, Cc) if SELF == "full" else torch.zeros_like(xs)
    print(f"  self-overlap mean: in-span {float(xs.mean()):.4f}, outside modes {float(xo.mean()):.4f} (on data 0.086)", flush=True)
    out = {}
    for k in ("b", "s"):
        w = st[k]; ev = torch.linalg.eigvalsh(w["Sig"]).cpu().numpy()
        out[k] = (w["mu"], w["sig"], float((w["m"] ** 2).sum() + torch.trace(w["Sig"])), float((w["m"] ** 2).sum()), ev.min(), ev.max())
    print(f"mode Laplace [{tag}]: bulk mu {out['b'][0]:.3f} sig {out['b'][1]:.3f}, |w_perp|^2 {out['b'][2]:.3f} "
          f"(|m_perp|^2 {out['b'][3]:.3f}, perp variances {out['b'][4]:.3f}..{out['b'][5]:.3f}); specialists mu {out['s'][0]:.2f}, "
          f"|w_perp|^2 {out['s'][2]:.2f}; kappa'/kappa {th.kp / kappa:.3f}; training ({tr[0]:+.4f}, {tr[1]:+.4f}); "
          f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    LAST["st"], LAST["th"] = st, th
    Fw = {}
    with torch.no_grad():
        for k in ("b", "s"):
            Ww, _ = well(k)
            EU = float(torch.cat([th.U(Ww[i:i + 4000]) for i in range(0, len(Ww), 4000)]).mean())
            ld = math.log(st[k]["sig"] ** 2) + float(torch.logdet(st[k]["Sig"]))
            Fw[k] = EU - 0.5 * (ld + d * math.log(2 * math.pi * math.e))
    LAST["F"] = Fw
    return tr, pop


def solve_fraction(S, tag, grid=(0.03, 0.06, 0.09, 0.13, 0.18, 0.25)):
    """Specialist fraction from the well free energies (Gaussian variational, Gibbs-Bogoliubov):
    root of g(f) = log(f/(1-f)) + F_spec(f) - F_bulk(f), each F at the kernel self-consistent with fraction f."""
    gs = []
    for f in grid:
        solve(S, f"{tag}, f={f}", f_s=f)
        F = LAST["F"]
        gs.append(math.log(f / (1 - f)) + F["s"] - F["b"])
        print(f"  f={f}: F_spec - F_bulk {F['s'] - F['b']:+.3f}, g(f) {gs[-1]:+.3f}", flush=True)
    gs = np.array(gs); fr = np.array(grid)
    j = np.where(np.sign(gs[:-1]) != np.sign(gs[1:]))[0]
    if len(j) == 0:
        print(f"no root of g(f) in {grid}", flush=True)
        return None
    j = j[0]
    lf = np.log(fr[j]) + (np.log(fr[j + 1]) - np.log(fr[j])) * gs[j] / (gs[j] - gs[j + 1])
    f_star = float(np.exp(lf))
    print(f"self-consistent specialist fraction {f_star:.3f} (particles 0.095, network 0.089)", flush=True)
    return f_star, solve(S, f"{tag}, self-consistent f={f_star:.3f}", f_s=f_star)


F_DAMP = float(os.environ.get("F_DAMP", "0.2"))


def solve_sc(S, tag, f0, f_damp=None):
    """solve() with a self-consistent specialist fraction. Fixed M/2 neurons per well; the fraction f enters as a kernel
    weight per well (setup_w). Each iteration f <- (1 - F_DAMP) f + F_DAMP Z_s / (Z_b + Z_s), with
    Z_k = int_{well k} exp(-V_k(w0)) dw0 * exp(1/2 log det Sig_k)  (V_k = U averaged over the well's w_perp Gaussian)."""
    fd = F_DAMP if f_damp is None else f_damp
    th = Theory(S)
    g = torch.Generator(device="cpu").manual_seed(0)
    Mh = M // 2
    Z = {k: torch.randn(Mh, d, generator=g, dtype=DT).to(DEV) for k in ("b", "s")}
    sg = {k: torch.where(torch.rand(Mh, generator=g) < 0.5, -1.0, 1.0).to(DT).to(DEV) for k in Z}
    Zp = torch.randn(N_PERP, d - 1, generator=g, dtype=DT).to(DEV)
    rng = _well_ranges()
    init = _well_init()
    st = {k: dict(mu=mu, sig=sd, m=torch.zeros(d - 1, device=DEV, dtype=DT), Sig=torch.eye(d - 1, device=DEV, dtype=DT) * s0 / d)
          for k, (mu, sd) in init.items()}
    f = f0

    def well(k):
        w = st[k]; L = torch.linalg.cholesky(w["Sig"])
        return torch.cat([(w["mu"] + w["sig"] * Z[k][:, 0])[:, None], w["m"] + Z[k][:, 1:] @ L.T], 1), L

    def setup():
        W = torch.cat([well(k)[0] * sg[k][:, None] for k in ("b", "s")], 0)
        tot = sig_a2 * N
        sw = torch.cat([torch.full((Mh,), tot * (1 - f) / Mh, device=DEV, dtype=DT), torch.full((Mh,), tot * f / Mh, device=DEV, dtype=DT)])
        th.setup_w(W, sw)

    for it in range(ITERS):
        setup()
        new, logZ = {}, {}
        for k in ("b", "s"):
            w = st[k]; Ww, L = well(k)
            lo, hi = rng[k]
            grid = torch.linspace(lo, hi, N_GRID, device=DEV, dtype=DT)
            Wg = torch.cat([grid.repeat_interleave(N_PERP)[:, None], (w["m"] + Zp @ L.T).repeat(N_GRID, 1)], 1)
            with torch.no_grad():
                V = th.U(Wg).reshape(N_GRID, N_PERP).mean(1).cpu().numpy()
            gr = grid.cpu().numpy()
            logZ[k] = float(np.log(np.trapz(np.exp(-(V - V.min())), gr)) - V.min() + 0.5 * float(torch.logdet(w["Sig"])))
            i = int(np.clip(np.argmin(V), 2, N_GRID - 3))
            cf = np.polyfit(gr[i - 2:i + 3], V[i - 2:i + 3], 2)
            mu = float(np.clip(-cf[1] / (2 * cf[0]), lo, hi)) if cf[0] > 0 else gr[i]
            sd = float(1 / math.sqrt(2 * cf[0])) if cf[0] > 0 else w["sig"]
            Wv = Ww.clone().requires_grad_(True)
            gp, = torch.autograd.grad(th.U(Wv).sum(), Wv)
            gp = gp[:, 1:]
            dev_ = Ww[:, 1:] - w["m"]
            H = torch.linalg.solve(w["Sig"], dev_.T @ gp / len(Ww)); H = 0.5 * (H + H.T)
            e, Vv = torch.linalg.eigh(H)
            Sig = Vv @ torch.diag(1 / e.clamp(2.0, 100.0)) @ Vv.T
            if FIX_TRACE:
                Sig = Sig * ((d - 1) * s0 / d) / torch.trace(Sig)
            m = w["m"] * 0 if FIX_M else w["m"] - M_DAMP * w["Sig"] @ gp.mean(0)
            new[k] = dict(mu=mu, sig=sd, m=m, Sig=Sig)
        for k in st:
            for q in st[k]:
                st[k][q] = (1 - DAMP) * st[k][q] + DAMP * new[k][q]
        f_new = 1.0 / (1.0 + math.exp(min(logZ["b"] - logZ["s"], 700.0)))
        f = (1 - fd) * f + fd * f_new
        if (it + 1) % 10 == 0 and fd > 0:
            tr, pop = th.disc()
            print(f"  [{tag}] iter {it + 1}: f {f:.4f} (target {f_new:.4f}), bulk mu {st['b']['mu']:.3f} sig {st['b']['sig']:.3f}, "
                  f"specialists mu {st['s']['mu']:.2f}, population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    setup()
    tr, pop = th.disc()
    print(f"self-consistent fraction [{tag}, start {f0}]: f = {f:.4f} (last target {f_new:.4f}); bulk mu {st['b']['mu']:.3f} "
          f"sig {st['b']['sig']:.3f}, specialists mu {st['s']['mu']:.2f} sig {st['s']['sig']:.2f}; training ({tr[0]:+.4f}, "
          f"{tr[1]:+.4f}); population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    LAST["st"], LAST["th"], LAST["f"], LAST["dlogZ"] = st, th, f, logZ["s"] - logZ["b"]
    return f, tr, pop


def solve_sc_root(S, tag, lo=0.005, hi=0.45, n_bisect=9):
    """Root of g(f) = log(Z_s/Z_b) - log(f/(1-f)), each g from a converged solve at fixed f (bisection in log f).
    g(lo) > 0 > g(hi) expected; prints the bracket check."""
    def g(f):
        solve_sc(S, f"{tag}, f={f:.4f}", f, f_damp=0.0)
        val = LAST["dlogZ"] - math.log(f / (1 - f))
        print(f"  g(f={f:.4f}) = {val:+.3f}  (log Z_s/Z_b {LAST['dlogZ']:+.3f})", flush=True)
        return val
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        print(f"no sign change on [{lo}, {hi}]: g = {glo:+.3f}, {ghi:+.3f}", flush=True)
        return None
    for _ in range(n_bisect):
        mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        gm = g(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi, ghi = mid, gm
    f_star = math.exp(0.5 * (math.log(lo) + math.log(hi)))
    out = solve_sc(S, f"{tag}, f*={f_star:.4f}", f_star, f_damp=0.0)
    print(f"SELF-CONSISTENT FRACTION [{tag}]: f* = {f_star:.4f} (bracket {lo:.4f}..{hi:.4f})", flush=True)
    return out


LAST = {}
NB, NS, IS_ITERS, IS_DAMP = 6000, 2000, 40, 0.3


def solve_exact(S, tag):
    """Exact single-neuron density exp(-U) inside each well (joint in w0 and w_perp, so the conditional of w_perp given
    w0 is exact), by importance weights on a fixed proposal built from the converged Gaussian wells (perp covariance
    inflated 1x, 2x, 4x for the tails; w0 sd 1.5x). Bulk |w0| <= 0.6, specialists |w0| > 0.6, fraction fixed F_S.
    The proposal is symmetrized under w -> -w (U(-w) = U(w)). Kernel rebuilt from the weighted sample, damped weights."""
    solve(S, tag + ", Gaussian start")
    st = LAST["st"]
    th = Theory(S)
    g = torch.Generator(device="cpu").manual_seed(5)
    lams = (1.0, 2.0, 4.0)
    Ws, lq, mask = [], [], []
    for k, n in (("b", NB), ("s", NS)):
        w = st[k]; L = torch.linalg.cholesky(w["Sig"]); s0w = 1.5 * w["sig"]
        comp = torch.randint(0, 3, (n,), generator=g)
        lam = torch.tensor(lams, dtype=DT)[comp].to(DEV)
        z = torch.randn(n, d, generator=g, dtype=DT).to(DEV)
        W = torch.cat([(w["mu"] + s0w * z[:, 0])[:, None], w["m"] + torch.sqrt(lam)[:, None] * (z[:, 1:] @ L.T)], 1)
        Sinv = torch.linalg.inv(w["Sig"]); ld = torch.logdet(w["Sig"])

        def logq(V):
            l0 = -0.5 * ((V[:, 0] - w["mu"]) / s0w) ** 2 - math.log(s0w)
            dv = V[:, 1:] - w["m"]; qf = ((dv @ Sinv) * dv).sum(1)
            lp = torch.logsumexp(torch.stack([-0.5 * qf / l - 0.5 * (d - 1) * math.log(l) for l in lams]), 0) - math.log(3)
            return l0 + lp - 0.5 * ld

        lq.append(torch.logaddexp(logq(W), logq(-W)) - math.log(2))
        inside = W[:, 0].abs() <= 0.6 if k == "b" else W[:, 0].abs() > 0.6
        Ws.append(W); mask.append(inside)
    W = torch.cat(Ws); lqa = torch.cat(lq); ins = torch.cat(mask)
    wb = torch.zeros(len(W), dtype=torch.bool, device=DEV); wb[:NB] = True
    frac = torch.where(wb, torch.tensor(1 - F_S, dtype=DT, device=DEV), torch.tensor(F_S, dtype=DT, device=DEV))

    def weights(U):
        lw = torch.where(ins, -U - lqa, torch.tensor(-1e30, dtype=DT, device=DEV))
        out = torch.zeros_like(lw)
        for sel in (wb, ~wb):
            out[sel] = torch.softmax(lw[sel], 0)
        return out

    om = weights((d / (2 * s0)) * (W ** 2).sum(1))
    for it in range(IS_ITERS):
        th.setup_w(W, sig_a2 * N * frac * om)
        with torch.no_grad():
            U = torch.cat([th.U(W[i:i + 4000]) for i in range(0, len(W), 4000)])
        om = (1 - IS_DAMP) * om + IS_DAMP * weights(U)
        if (it + 1) % 10 == 0:
            tr, pop = th.disc()
            ess = [float(1 / (om[sel] ** 2).sum()) for sel in (wb, ~wb)]
            r2 = (W[:, 1:] ** 2).sum(1)
            print(f"  [{tag}, exact density] iter {it + 1}: bulk |w_perp|^2 {float((om[wb] * r2[wb]).sum()):.3f}, "
                  f"effective sample size bulk {ess[0]:.0f}/{NB}, specialists {ess[1]:.0f}/{NS}; training ({tr[0]:+.4f}, {tr[1]:+.4f}), "
                  f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    th.setup_w(W, sig_a2 * N * frac * om)
    tr, pop = th.disc()
    with torch.no_grad():
        C, _ = coeffs(W)
        xs = sig_a2 * ((C @ th.A.T) * C).sum(1); xo = th.x_out(W, C)
    r2 = (W[:, 1:] ** 2).sum(1); a0 = W[:, 0].abs()
    mb = float((om[wb] * r2[wb]).sum()); vb = float((om[wb] * (r2[wb] - mb) ** 2).sum())
    i = torch.argsort(a0[~wb]); cs = torch.cumsum(om[~wb][i], 0); smed = float(a0[~wb][i][torch.searchsorted(cs, cs[-1] / 2)])
    print(f"exact-density mode theory [{tag}]: bulk mean |w0| {float((om[wb] * a0[wb]).sum()):.3f}, |w_perp|^2 {mb:.3f} "
          f"(variance {vb:.3f}); specialists median |w0| {smed:.2f}, |w_perp|^2 {float((om[~wb] * r2[~wb]).sum()):.2f}; "
          f"self-overlap mean {float((frac * om * (xs + xo)).sum()):.4f}; training ({tr[0]:+.4f}, {tr[1]:+.4f}); "
          f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    return tr, pop


R_ITERS, R_DAMP, N_DIR_R = 40, 0.3, 16


def solve_radial(S, tag):
    """Semi-analytic exact radial density. Direction of w_perp: angular shape of the converged Laplace covariance
    (bulk Sig for |w0| <= 0.6, specialist Sig above); u = L z / |L z|. Magnitude rho = |w_perp| exact on a grid:
    p(w0, rho) ∝ rho^(d-2) <exp(-U(w0, rho u))>_u (N_DIR_R fixed directions per region). Region fractions fixed
    (bulk 1 - F_S, specialists F_S). Kernel rebuilt from M samples of the grid density each iteration (damped)."""
    solve(S, tag + ", Gaussian start")
    st = LAST["st"]
    th = Theory(S)
    g = torch.Generator(device="cpu").manual_seed(7)
    g0 = torch.linspace(0.0, 2.6, 81, device=DEV, dtype=DT); rh = torch.linspace(0.05, 2.5, 48, device=DEV, dtype=DT)
    g0n, rhn = g0.cpu().numpy(), rh.cpu().numpy()
    Lr = {k: torch.linalg.cholesky(st[k]["Sig"]) for k in ("b", "s")}

    def acg(z, k):
        v = z @ Lr[k].T
        return v / v.norm(dim=1, keepdim=True)

    Ud = {k: acg(torch.randn(N_DIR_R, d - 1, generator=g, dtype=DT).to(DEV), k) for k in ("b", "s")}
    reg_b = g0n <= 0.6
    Wg = torch.zeros(len(g0), len(rh), N_DIR_R, d, device=DEV, dtype=DT)
    Wg[..., 0] = g0[:, None, None]
    for i in range(len(g0)):
        u = Ud["b" if reg_b[i] else "s"]
        Wg[i, :, :, 1:] = rh[:, None, None] * u[None, :, :]
    Wg = Wg.reshape(-1, d)
    U0 = torch.rand(M, generator=g, dtype=DT).numpy(); U1 = torch.rand(M, generator=g, dtype=DT).numpy()
    Zc = torch.randn(M, d - 1, generator=g, dtype=DT).to(DEV)
    sgn = torch.where(torch.arange(M, device=DEV) % 2 == 0, 1.0, -1.0).to(DT)
    dg, dr = g0n[1] - g0n[0], rhn[1] - rhn[0]

    def cloud(p):
        pw = p.sum(1); cw = np.cumsum(pw) / pw.sum()
        i0 = np.clip(np.searchsorted(cw, U0), 0, len(g0n) - 1)
        cr = np.cumsum(p[i0], 1); cr /= cr[:, -1:]
        ir = np.array([np.clip(np.searchsorted(cr[k], U1[k]), 0, len(rhn) - 1) for k in range(M)])
        w0 = torch.from_numpy(np.clip(g0n[i0] + dg * (U1 - 0.5), 0, None)).to(DEV)
        r = torch.from_numpy(np.clip(rhn[ir] + dr * (U0 - 0.5), 1e-3, None)).to(DEV)
        isb = torch.from_numpy(reg_b[i0]).to(DEV)
        u = torch.where(isb[:, None], acg(Zc, "b"), acg(Zc, "s"))
        return torch.cat([w0[:, None], r[:, None] * u], 1) * sgn[:, None]

    def density(logR):
        lp = logR + (d - 2) * np.log(rhn)[None, :]
        p = np.zeros_like(lp)
        for sel, f in ((reg_b, 1 - F_S), (~reg_b, F_S)):
            q = np.exp(lp[sel] - lp[sel].max()); p[sel] = f * q / q.sum()
        return p

    p = density(-(d / (2 * s0)) * (g0n[:, None] ** 2 + rhn[None, :] ** 2))
    for it in range(R_ITERS):
        th.setup(cloud(p))
        with torch.no_grad():
            U = torch.cat([th.U(Wg[i:i + 15000]) for i in range(0, len(Wg), 15000)]).reshape(len(g0), len(rh), N_DIR_R)
        logR = (torch.logsumexp(-U, 2) - math.log(N_DIR_R)).cpu().numpy()
        p = (1 - R_DAMP) * p + R_DAMP * density(logR)
        if (it + 1) % 10 == 0:
            tr, pop = th.disc()
            pb = p[reg_b]
            print(f"  [{tag}, radial] iter {it + 1}: bulk |w_perp|^2 {float((pb * rhn[None, :] ** 2).sum() / pb.sum()):.3f}, "
                  f"training ({tr[0]:+.4f}, {tr[1]:+.4f}), population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    Wc = cloud(p); th.setup(Wc); tr, pop = th.disc()
    with torch.no_grad():
        C, _ = coeffs(Wc)
        xs = sig_a2 * ((C @ th.A.T) * C).sum(1); xo = th.x_out(Wc, C)
    pb, ps = p[reg_b], p[~reg_b]
    mb = float((pb * rhn ** 2).sum() / pb.sum()); vb = float((pb * (rhn ** 2 - mb) ** 2).sum() / pb.sum())
    pws = ps.sum(1); cs = np.cumsum(pws); smed = g0n[~reg_b][np.searchsorted(cs, cs[-1] / 2)]
    print(f"radial exact-density mode theory [{tag}]: bulk mean |w0| {float((pb.sum(1) * g0n[reg_b]).sum() / pb.sum()):.3f}, "
          f"|w_perp|^2 {mb:.3f} (variance {vb:.3f}); specialists median |w0| {smed:.2f}, |w_perp|^2 "
          f"{float((ps * rhn ** 2).sum() / ps.sum()):.2f}; self-overlap mean {float((xs + xo).mean()):.4f}; "
          f"training ({tr[0]:+.4f}, {tr[1]:+.4f}); population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    return tr, pop


def main():
    print(f"d={d} P={P} N={N}; Laplace in {NM} modes; specialist fraction fixed at {F_S}")
    print("targets: network training (+0.0155, +0.0786), population (+0.097, +0.302); 8-sample particle mean training "
          "(+0.0158, +0.0945), population (+0.106, +0.303); network bulk |w0| 0.305, |w_perp|^2 1.11; specialists 1.47", flush=True)
    if WHAT in ("check", "all"):
        check()
    if WHAT in ("pop", "all"):
        solve(torch.eye(NM, device=DEV, dtype=DT), "off data, S = I")
    if WHAT in ("synth", "all"):
        res = []
        for sd in range(NSYN):
            Xs = torch.randn(P, d, generator=torch.Generator().manual_seed(1000 + sd), dtype=DT).to(DEV)
            Ps = modes(Xs)
            res.append(solve(Ps.T @ Ps / P, f"quenched, synthetic sample {sd}"))
        tr = np.mean([r[0] for r in res], 0); pop = np.mean([r[1] for r in res], 0)
        print(f"quenched average over {NSYN} synthetic samples: training ({tr[0]:+.4f}, {tr[1]:+.4f}), "
              f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    if WHAT in ("data", "all"):
        Pd = modes(torch.from_numpy(X_np).to(DEV))
        solve(Pd.T @ Pd / P, "actual training set")
    if WHAT == "synthdata":
        Xs = torch.from_numpy(np.random.default_rng(DATA_SEED).standard_normal((P, d))).to(DEV)
        Ps = modes(Xs)
        solve(Ps.T @ Ps / P, f"synthetic sample seed {DATA_SEED}, P={P}")
        solve(torch.eye(NM, device=DEV, dtype=DT), f"off data, S = I, P={P}")
    if WHAT == "seeds":
        for sd in [int(x) for x in os.environ.get("SEEDS", "101,102,103,104").split(",")]:
            Xs = torch.from_numpy(np.random.default_rng(sd).standard_normal((P, d))).to(DEV)
            Ps = modes(Xs)
            solve(Ps.T @ Ps / P, f"particle-run sample seed {sd}")
    if WHAT == "gauss":
        # S = I + dS, dS Gaussian with Cov(dS_ab, dS_cd) = (E[psi_a psi_b psi_c psi_d] - d_ab d_cd)/P, realised as
        # dS = sum_m g_m (psi_m psi_m^T - I) / sqrt(P M_POP) over population points psi_m, g_m ~ N(0,1) (no training set).
        M_POP, CH = int(float(os.environ.get("M_POP", "4e6"))), 200000
        res = []
        for sd in [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6,7,8").split(",")]:
            gen = torch.Generator(device=DEV).manual_seed(7000 + sd)
            dS = torch.zeros(NM, NM, device=DEV, dtype=DT); gsum = 0.0
            for _ in range(M_POP // CH):
                Pm = modes(torch.randn(CH, d, generator=gen, device=DEV, dtype=DT))
                g = torch.randn(CH, generator=gen, device=DEV, dtype=DT)
                dS += Pm.T @ (g[:, None] * Pm); gsum += float(g.sum())
            dS = (dS - gsum * torch.eye(NM, device=DEV, dtype=DT)) / math.sqrt(P * M_POP)
            S = torch.eye(NM, device=DEV, dtype=DT) + (dS + dS.T) / 2
            print(f"Gaussian S draw {sd}: min eigenvalue {float(torch.linalg.eigvalsh(S).min()):.3f}", flush=True)
            res.append(solve(S, f"Gaussian S draw {sd} (no dataset)"))
        tr = np.array([r[0] for r in res]); pop = np.array([r[1] for r in res]); n = len(res)
        print(f"average over {n} Gaussian S draws: training ({tr[:, 0].mean():+.4f}, {tr[:, 1].mean():+.4f}) "
              f"+- ({tr[:, 0].std(ddof=1) / n ** .5:.4f}, {tr[:, 1].std(ddof=1) / n ** .5:.4f}); population "
              f"({pop[:, 0].mean():+.4f}, {pop[:, 1].mean():+.4f}) +- ({pop[:, 0].std(ddof=1) / n ** .5:.4f}, "
              f"{pop[:, 1].std(ddof=1) / n ** .5:.4f}); particle 8-sample mean training (+0.0158, +0.0945), "
              f"population (+0.1057, +0.3030)", flush=True)
    if WHAT == "fd2":
        # Dataset-averaged 1/P correction without a training set: E[O(I + dS)] ~ O(I) + 1/2 E[d^2 O(dS, dS)],
        # Cov(dS_ab, dS_cd) = (E[psi_a psi_b psi_c psi_d] - d_ab d_cd)/P exactly (population points psi_m, weights g_m).
        # Second derivative by symmetric differences at amplitude EPS_FD: [O(I + e dS) + O(I - e dS) - 2 O(I)] / (2 e^2).
        M_POP, CH = int(float(os.environ.get("M_POP", "4e6"))), 200000
        EPS_FD = float(os.environ.get("EPS_FD", "0.15"))
        I_ = torch.eye(NM, device=DEV, dtype=DT)
        o0 = solve(I_, f"P={P} off data S = I")
        cors = []
        for sd in [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6").split(",")]:
            gen = torch.Generator(device=DEV).manual_seed(7000 + sd)
            dS = torch.zeros(NM, NM, device=DEV, dtype=DT); gsum = 0.0
            for _ in range(M_POP // CH):
                Pm = modes(torch.randn(CH, d, generator=gen, device=DEV, dtype=DT))
                g = torch.randn(CH, generator=gen, device=DEV, dtype=DT)
                dS += Pm.T @ (g[:, None] * Pm); gsum += float(g.sum())
            dS = (dS - gsum * I_) / math.sqrt(P * M_POP); dS = (dS + dS.T) / 2
            ev = torch.linalg.eigvalsh(dS)
            op = solve(I_ + EPS_FD * dS, f"P={P} S = I + {EPS_FD} dS draw {sd}")
            om = solve(I_ - EPS_FD * dS, f"P={P} S = I - {EPS_FD} dS draw {sd}")
            c = [(op[j] + om[j] - 2 * o0[j]) / (2 * EPS_FD ** 2) for j in (0, 1)]
            odd = [(op[j] - om[j]) / (2 * EPS_FD) for j in (0, 1)]
            cors.append(c)
            print(f"FD2 draw {sd} (dS eigenvalues {float(ev.min()):.2f}..{float(ev.max()):.2f}): second-order correction training "
                  f"({c[0][0]:+.4f}, {c[0][1]:+.4f}) population ({c[1][0]:+.4f}, {c[1][1]:+.4f}); first-order (zero on average) "
                  f"population ({odd[1][0]:+.4f}, {odd[1][1]:+.4f})", flush=True)
            n = len(cors); ct = np.array([x[0] for x in cors]); cp = np.array([x[1] for x in cors])
            se = lambda a: a.std(0, ddof=1) / n ** .5 if n > 1 else np.zeros(2)
            print(f"FD2 running ({n} draws): S = I + 1/P correction: training ({o0[0][0] + ct.mean(0)[0]:+.4f}, {o0[0][1] + ct.mean(0)[1]:+.4f}) "
                  f"+- ({se(ct)[0]:.4f}, {se(ct)[1]:.4f}); population ({o0[1][0] + cp.mean(0)[0]:+.4f}, {o0[1][1] + cp.mean(0)[1]:+.4f}) "
                  f"+- ({se(cp)[0]:.4f}, {se(cp)[1]:.4f})", flush=True)
    if WHAT == "sc_frac":
        for f0 in [float(x) for x in os.environ.get("F0", "0.03,0.2").split(",")]:
            solve_sc(torch.eye(NM, device=DEV, dtype=DT), f"d={d} off data", f0)
    if WHAT == "sc_root":
        solve_sc_root(torch.eye(NM, device=DEV, dtype=DT), f"d={d} off data",
                      lo=float(os.environ.get("F_LO", "0.005")), hi=float(os.environ.get("F_HI", "0.45")),
                      n_bisect=int(os.environ.get("N_BISECT", "9")))
    if WHAT == "sc_scan":
        for f in [float(x) for x in os.environ.get("F_GRID", "0.02,0.04,0.06,0.08,0.1,0.13,0.18,0.25").split(",")]:
            solve_sc(torch.eye(NM, device=DEV, dtype=DT), f"d={d} off data, f={f}", f, f_damp=0.0)
            print(f"SCAN d={d} f={f}: g = {LAST['dlogZ'] - math.log(f / (1 - f)):+.3f} (log Z_s/Z_b {LAST['dlogZ']:+.3f})", flush=True)
    if WHAT == "pscan":
        res = []
        for sd in [int(x) for x in os.environ.get("SEEDS", "101,102").split(",")]:
            Xs = torch.from_numpy(np.random.default_rng(sd).standard_normal((P, d))).to(DEV)
            Ps = modes(Xs)
            res.append(solve(Ps.T @ Ps / P, f"P={P} kappa={kappa:.5f} sample seed {sd}"))
        off = solve(torch.eye(NM, device=DEV, dtype=DT), f"P={P} kappa={kappa:.5f} off data S = I")
        tr = np.mean([r[0] for r in res], 0); pop = np.mean([r[1] for r in res], 0)
        print(f"PSCAN P={P} kappa/P={kappa / P:.3e}: on-data mean training ({tr[0]:+.4f}, {tr[1]:+.4f}) population "
              f"({pop[0]:+.4f}, {pop[1]:+.4f}); off data training ({off[0][0]:+.4f}, {off[0][1]:+.4f}) population "
              f"({off[1][0]:+.4f}, {off[1][1]:+.4f})", flush=True)
    if WHAT == "frac_seeds":
        for sd in [int(x) for x in os.environ.get("SEEDS", "101,102,103,104").split(",")]:
            Xs = torch.from_numpy(np.random.default_rng(sd).standard_normal((P, d))).to(DEV)
            Ps = modes(Xs)
            solve_fraction(Ps.T @ Ps / P, f"particle-run sample seed {sd}")
    if WHAT == "frac_data":
        Pd = modes(torch.from_numpy(X_np).to(DEV))
        solve_fraction(Pd.T @ Pd / P, "actual training set")
    if WHAT == "frac_synth":
        res = []
        for sd in range(NSYN):
            Xs = torch.randn(P, d, generator=torch.Generator().manual_seed(1000 + sd), dtype=DT).to(DEV)
            Ps = modes(Xs)
            out = solve_fraction(Ps.T @ Ps / P, f"quenched, synthetic sample {sd}")
            if out is not None:
                res.append(out)
        if res:
            fs = [r[0] for r in res]; tr = np.mean([r[1][0] for r in res], 0); pop = np.mean([r[1][1] for r in res], 0)
            print(f"quenched average over {len(res)} synthetic samples: specialist fraction {np.mean(fs):.3f} +- {np.std(fs, ddof=1) if len(fs) > 1 else 0:.3f}; "
                  f"training ({tr[0]:+.4f}, {tr[1]:+.4f}), population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    if WHAT == "radial_data":
        Pd = modes(torch.from_numpy(X_np).to(DEV))
        solve_radial(Pd.T @ Pd / P, "actual training set")
    if WHAT == "radial_synth":
        res = []
        for sd in range(NSYN):
            Xs = torch.randn(P, d, generator=torch.Generator().manual_seed(1000 + sd), dtype=DT).to(DEV)
            Ps = modes(Xs)
            res.append(solve_radial(Ps.T @ Ps / P, f"quenched, synthetic sample {sd}"))
        tr = np.mean([r[0] for r in res], 0); pop = np.mean([r[1] for r in res], 0)
        print(f"radial exact density, quenched average over {NSYN} synthetic samples: training ({tr[0]:+.4f}, {tr[1]:+.4f}), "
              f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)
    if WHAT == "exact_data":
        Pd = modes(torch.from_numpy(X_np).to(DEV))
        solve_exact(Pd.T @ Pd / P, "actual training set")
    if WHAT == "exact_synth":
        res = []
        for sd in range(NSYN):
            Xs = torch.randn(P, d, generator=torch.Generator().manual_seed(1000 + sd), dtype=DT).to(DEV)
            Ps = modes(Xs)
            res.append(solve_exact(Ps.T @ Ps / P, f"quenched, synthetic sample {sd}"))
        tr = np.mean([r[0] for r in res], 0); pop = np.mean([r[1] for r in res], 0)
        print(f"exact density, quenched average over {NSYN} synthetic samples: training ({tr[0]:+.4f}, {tr[1]:+.4f}), "
              f"population ({pop[0]:+.4f}, {pop[1]:+.4f})", flush=True)


if __name__ == "__main__":
    main()
