"""
FCN3 theory with the perpendicular directions locked to their prior (no q adaptation).

Same free energy and solver as fcn3_fixed.BlazingFreeEnergy (the default theory), with two things held fixed:

  * read-in layer: every neuron's perpendicular norm q = |w_perp|^2 is fixed at its prior value q0 = sigma_w^2 (d-1)/d.
    The read-in distribution is a law over w0 alone (one q column of the (w0, log q) grid), with prior
    exp(-beta w0^2), and it is tilted only through the He-tower features c_n(w0, q0). There is no tilt on the
    perpendicular part P_perp of the read-in activation (the statistic is dropped).
  * hidden layer: every hidden neuron's perpendicular variance q_h is fixed at its prior value
    qbar0 = sigma_B^2 T0, T0 = E_prior[P_perp(w0, q0)]: no Gamma spread in q_h and no response of qbar to the read-in
    distribution (E7 is replaced by qbar = qbar0). M (E8) then enters nothing but the proposal lattice.

What is removed relative to the full theory: the q-dependence of c_n(w0, q) along each read-in neuron's q, the
read-in tilt r_B M (qbar - <q_h>)/qbar^2 P_perp, the read-in's prior spread in q, and the hidden-field's q_h
distribution and its tilt by exp(g~/2 (t.e)^2) through e(h, q_h). The He-tower part (J, G, H, t) is unchanged.

    from fcn3_locked_perp import solve_locked
    r = solve_locked(Params(...))          # same keys as fcn3_fixed.solve_fcn3
"""
import math
from dataclasses import replace

import numpy as np
import torch
from scipy.special import erfinv
from scipy.stats import qmc

from fcn3_fixed import SQ2PI, BlazingFreeEnergy, HiddenMC
from fcn3_matrix_solver import DT, Numerics, trapz_weights


class LockedHiddenMC(HiddenMC):
    """Hidden-field samples with q_h = qbar exactly (log rho = 0): the proposal and the prior are densities in z only."""

    def set_proposal(self, comps, M):
        S = self.S
        zs = []
        for k, c in enumerate(comps):
            sob = qmc.Sobol(d=S, scramble=True, seed=self.seed + 101 * k)
            u = np.clip(sob.random_base2(self.n_log2), 1e-12, 1 - 1e-12)
            g = torch.tensor(np.sqrt(2) * erfinv(2 * u - 1), dtype=DT, device=self.dev)
            zs.append(c["mu"] + g @ c["C"].T)
        self.z = torch.cat(zs)
        self.lr = torch.zeros(len(self.z), dtype=DT, device=self.dev)
        lw = math.log(1.0 / len(comps))
        lmix = []
        for c in comps:
            zz = torch.linalg.solve_triangular(c["C"], (self.z - c["mu"]).T, upper=False).T
            lmix.append(-0.5 * (zz * zz).sum(1) - torch.log(torch.diagonal(c["C"])).sum() - S * math.log(SQ2PI) + lw)
        self.lcomp = torch.stack(lmix)
        self.lmix = torch.logsumexp(self.lcomp, 0)
        self.lprior_z = -0.5 * (self.z * self.z).sum(1) - S * math.log(SQ2PI)
        self.comps = comps

    def log_prior_rho(self, M):
        return torch.zeros_like(self.lr)


class LockedPerpFreeEnergy(BlazingFreeEnergy):
    def __init__(self, p, nu=None, **kw):
        nu = replace(nu or Numerics(), nq=1, lq_below=0.0, lq_above=0.0)     # one q column, at q0
        super().__init__(p, nu=nu, **kw)
        ri = self.ri
        assert ri.Q.shape[1] == 1 and abs(float(ri.Q[0, 0]) / ri.q0 - 1) < 1e-12
        # read-in prior on the single column: exp(-beta w0^2) with trapezoid weights in w0 (the q weight of a
        # one-point grid is 0 in trapz_weights, and the q part of the prior is a constant)
        ri.logm = torch.log(trapz_weights(ri.w))[:, None]
        self.lp0 = -ri.base + ri.logm
        self.lp0 = self.lp0 - torch.logsumexp(self.lp0.reshape(-1), 0)
        pw0 = torch.exp(self.lp0.reshape(-1))
        self.J0 = (self.Cf * pw0) @ self.Cf.T
        self.T0 = float(pw0 @ self.Pp)                    # prior E[P_perp(w0, q0)]: qbar0 = sigma_B^2 T0
        # statistics r^2k only (no P_perp tilt)
        r2 = (ri.W ** 2 / (1 + 2 * ri.W ** 2 + 2 * ri.Q)).reshape(-1)
        R = torch.stack([r2 ** k for k in range(1, self.K + 1)])
        self.sk = R @ pw0
        self.Tst = R / self.sk[:, None]
        self.nth = self.K
        self.mc = LockedHiddenMC(p, self.dev, n_log2=self.mc.n_log2, seed=self.mc.seed)

    def moments(self, phi):
        lw = self.lp0_a - 0.5 * (self.theta_ref @ self.Tst_a) - 0.5 * (phi @ self.S_w)
        lw = lw - torch.logsumexp(lw, 0)
        pw = torch.exp(lw)
        mbar = self.Tst_a @ pw
        D = -(pw * (lw - self.lp0_a)).sum()
        J = torch.einsum("k,kij->ij", self.sk * mbar, self.E)
        T = torch.tensor(self.T0, dtype=DT, device=self.dev)       # qbar locked at its prior value
        return D, J, T, pw

    def set_samples(self, n_log2, M, fresh=False):
        J, T = self.last["J"], self.last["T"]
        self.mc = LockedHiddenMC(self.p, self.dev, n_log2=n_log2, seed=self.mc.seed)
        self.mc_ready = False
        if fresh:
            self.far_comps = []
        self.build_proposal(J, T, M, self.t)
        with torch.no_grad():
            from fcn3_fixed import scaled_cholesky
            self.t, _ = self.solve_t(scaled_cholesky(self.p.sigB ** 2 * J), self.p.sigB ** 2 * T, M, self.t)
        self.build_proposal(J, T, M, self.t)
        self.refit(M)
        self.nprop += 1


def solve_locked(p, nmax=11, device=None, verbose=False, seed=0, theta0=None):
    """Locked-perpendicular theory at p: n_max = 3 from the prior, then `nmax` from that read-in law (as
    fcn3_fixed.solve_fcn3). theta0: optional n_max-length read-in exponents to start the final stage from instead.
    Returns the result dict (with the solver under "_F" for callers that need the grid or the samples)."""
    if theta0 is None:
        F3 = LockedPerpFreeEnergy(replace(p, nmax=3), device=device, verbose=verbose, seed=seed)
        o3 = F3.solve()
        if nmax == 3:
            o3["_F"] = F3
            return o3
        F = LockedPerpFreeEnergy(replace(p, nmax=nmax), device=device, verbose=verbose, seed=seed)
        th = torch.zeros(F.nth, dtype=DT, device=F.dev)
        th[:3] = F3.theta_ref[:3].to(F.dev)
        o = F.solve(theta0=th, M0=o3["M"])
        o["nmax3"] = {k: v for k, v in o3.items() if k not in ("t", "J", "H", "theta")}
        del F3
    else:
        F = LockedPerpFreeEnergy(replace(p, nmax=nmax), device=device, verbose=verbose, seed=seed)
        o = F.solve(theta0=torch.as_tensor(theta0, dtype=DT, device=F.dev))
    o["nmax"], o["_F"] = nmax, F
    return o


def project_readin(F, w_t, nlp_t, rank_tol=1e-9):
    """Read-in exponents theta (for F.solve(theta0=...)) of the locked law whose r^2k moments equal those of a target
    density on |w0| (w_t, -log density nlp_t; non-finite = pruned, given a soft floor 5 nats above the highest finite
    value). Used as a start: e.g. the full theory's read-in law. Directions of the statistics with variance below
    rank_tol x the largest under the target are left out (the r^2k are nearly collinear)."""
    w = F.ri.w.cpu().numpy()
    nlp_t = np.asarray(nlp_t, float)
    fin = np.isfinite(nlp_t)
    nl = np.where(fin, nlp_t, nlp_t[fin].max() + 5.0)
    lt = torch.tensor(np.interp(w, w_t, -nl), dtype=DT, device=F.dev) + torch.log(trapz_weights(F.ri.w))
    pt = torch.softmax(lt, 0)
    T = F.Tst
    Dv = T - (T @ pt)[:, None]
    C = (Dv * pt) @ Dv.T
    ds = torch.sqrt(torch.diagonal(C))
    ev, U = torch.linalg.eigh(C / torch.outer(ds, ds))
    keep = ev > rank_tol * ev.max()
    Wm = (U[:, keep] / torch.sqrt(ev[keep])).T / ds
    S = Wm @ Dv                                           # whitened, centred under the target: target E[S] = 0
    lp0 = F.lp0.reshape(-1)
    I = torch.eye(S.shape[0], dtype=DT, device=F.dev)
    phi = torch.zeros(S.shape[0], dtype=DT, device=F.dev)
    f = lambda ph: float(torch.logsumexp(lp0 - 0.5 * ph @ S, 0))      # convex; minimum at E_phi[S] = 0
    for _ in range(200):
        p = torch.softmax(lp0 - 0.5 * phi @ S, 0)
        ES = S @ p
        g = -0.5 * ES
        if float(g.abs().max()) < 1e-10:
            break
        Hm = 0.25 * ((S - ES[:, None]) * p) @ (S - ES[:, None]).T
        step = -torch.linalg.solve(Hm + 1e-12 * I, g)
        a, f0 = 1.0, f(phi)
        while f(phi + a * step) > f0 + 1e-4 * a * float(g @ step) and a > 1e-10:
            a *= 0.5
        phi = phi + a * step
    return (Wm.T @ phi).cpu().numpy()


def readin_nlp(F):
    """(w0 grid, -log p(w0) on the full line) of the solved read-in law, evaluated on the whole w0 grid (the solver
    prunes points 60 nats below the peak)."""
    with torch.no_grad():
        lw = F.lp0.reshape(-1) - 0.5 * (F.theta_ref @ F.Tst)
        lw = lw - torch.logsumexp(lw, 0)
        nlp = -(lw - torch.log(trapz_weights(F.ri.w)) + math.log(0.5))
    return F.ri.w.cpu().numpy(), nlp.cpu().numpy()


if __name__ == "__main__":
    import sys
    import time
    from fcn3_matrix_solver import Params
    from fcn3_fixed import solve_fcn3
    P = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
    p = Params(d=150, P=P, N1=1500, N2=1500, chi=1.0, kappa=0.015, eps=-0.3 * math.sqrt(6), sigA=math.sqrt(8 / 1500),
               sigB=1.25540121081682, sigw=math.sqrt(0.5), nmax=11)
    t0 = time.time()
    r = solve_locked(p, verbose=True)
    F = r["_F"]
    print(f"locked: L1 {r['L1']:.4f} L3 {r['L3']:.4f} outer mass {r['readin_mass_w0_gt_1']:.3e} "
          f"converged {r['converged']} ({time.time() - t0:.1f}s); qbar0 = {p.sigB ** 2 * F.T0:.5f}")
    o = solve_fcn3(p)
    print(f"full:   L1 {o['L1']:.4f} L3 {o['L3']:.4f} Eq_h/qbar {o['Eq'] / (p.sigB ** 2 * o['T']):.4f} "
          f"qbar {p.sigB ** 2 * o['T']:.5f}")
