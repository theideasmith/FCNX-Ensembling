"""
Free-energy solver for (E1)-(E8) with FIXED quadrature over the full distributions.

    W(J, T) = N1 * min_theta D(theta; J, T)                                     (read-in: -N1 KL*)
              + min_t [ N2 log Z_h(L(J), qbar = sigma_B^2 T, M; t) + chi (kappa/(2P)|t|^2 - y.t) ]

Read-in: the fixed (w0, log q) grid of fcn3_matrix_solver.ReadIn (sinh-spaced w0 to 6, q from q0 e^-16 to
q0 e^1.5); it covers both read-in minima and never moves.

Hidden field: fixed importance samples. Whitened z (h = L z, L L^T = G) and rho = q_h / qbar are drawn ONCE per
outer iteration from a mixture: the prior, plus a Gaussian on each far basin of the hidden-field distribution
(and its mirror image under h -> -h), each with a lognormal in rho. Weights w_i = prior(z_i, rho_i) / mixture.
With the samples fixed,
  * every quantity is a smooth deterministic function of (J, T, M, t);
  * log Z_h(t) = log sum_i w_i exp(g~/2 (t.e_i)^2) - log sum_i w_i is a log-sum-exp of convex quadratics, so the
    readout problem is exactly convex (Newton + Armijo on the potential converges);
  * dW/dx is computed by automatic differentiation through L(J) (scaled Cholesky), erf and the Gamma weight, so
    it is the exact derivative of the discretised W. The read-in part uses the envelope theorem (exact dual).
The proposal is re-centred between outer iterations from the current state (lattice scan along the drive
direction), and its effective sample size is reported.

The equilibrium is the global maximum of W; stationary points are (E4)-(E7), M from (E8) in the outer loop.

DEFAULT: solve_fcn3(p) at the bottom of this file (BlazingFreeEnergy, n_max = 3 then 11; seconds per solve).
The other classes are the development path and are kept for reference (FixedFreeEnergy: J-parametrised, valid
only at n_max = 3; Theta/Moment/Whitened/FastFreeEnergy: superseded by BlazingFreeEnergy).
"""
import math
import time

import numpy as np
import torch
from scipy.special import erfinv
from scipy.stats import qmc

from fcn3_free_energy import FreeEnergy
from fcn3_matrix_solver import DT, Groups, Numerics, Params, basin_labels, hermite_norm, solve_t_convex

SQ2PI = math.sqrt(2 * math.pi)


def scaled_cholesky(G):
    d = torch.sqrt(torch.diagonal(G))
    R = G / torch.outer(d, d)
    return d[:, None] * torch.linalg.cholesky(R)


def _tiny_gram(A, B):
    """A^T B for tall A, B (n x S, small S) as a broadcast sum: cuBLAS fp64 is ~100x slow on such products."""
    return (A[:, :, None] * B[:, None, :]).sum(0)


class HiddenMC:
    def __init__(self, p: Params, dev, n_log2=15, nx=64, seed=0):
        self.p, self.dev = p, dev
        self.S = p.nS
        self.n_log2, self.seed = n_log2, seed
        from numpy.polynomial.hermite_e import hermegauss
        x, w = hermegauss(nx)
        x = torch.tensor(x, dtype=DT, device=dev)
        wx = torch.tensor(w / w.sum(), dtype=DT, device=dev)
        self.Phi = hermite_norm(p.ks, x)                       # (S, nx)
        self.WPhi = (self.Phi * wx).T.contiguous()              # (nx, S)
        self.wx = wx

    def set_proposal(self, comps, M):
        """comps: list of dicts with mu (S,) and a lower-triangular factor C (S, S) in z, and lmu, lsd for log rho.
        Draws the fixed samples (scrambled Sobol, one equal block per component)."""
        S = self.S
        zs, lrs = [], []
        for k, c in enumerate(comps):
            sob = qmc.Sobol(d=S + 1, scramble=True, seed=self.seed + 101 * k)
            u = np.clip(sob.random_base2(self.n_log2), 1e-12, 1 - 1e-12)
            g = torch.tensor(np.sqrt(2) * erfinv(2 * u - 1),
                             dtype=DT, device=self.dev)
            zs.append(c["mu"] + g[:, :S] @ c["C"].T)
            lrs.append(c["lmu"] + c["lsd"] * g[:, S])
        self.z = torch.cat(zs)
        self.lr = torch.cat(lrs)
        # log mixture density (in z and log rho)
        lmix = []
        # the proposal is the mixture with the ACTUAL sample allocation (equal blocks per component)
        wts = torch.full((len(comps),), 1.0 / len(comps), dtype=DT, device=self.dev)
        for k, c in enumerate(comps):
            zz = torch.linalg.solve_triangular(c["C"], (self.z - c["mu"]).T, upper=False).T
            l = -0.5 * (zz * zz).sum(1) - torch.log(torch.diagonal(c["C"])).sum() - S * math.log(SQ2PI)
            l = l - 0.5 * ((self.lr - c["lmu"]) / c["lsd"]) ** 2 - math.log(c["lsd"] * SQ2PI)
            lmix.append(l + torch.log(wts[k]))
        self.lcomp = torch.stack(lmix)                 # (components, samples): log weight x density of each
        self.lmix = torch.logsumexp(self.lcomp, 0)
        self.lprior_z = -0.5 * (self.z * self.z).sum(1) - S * math.log(SQ2PI)
        self.comps = comps

    def log_prior_rho(self, M):
        """log of the Gamma(M/2, mean 1) density of rho, per d(log rho)."""
        k = M / 2
        rho = torch.exp(self.lr)
        return k * math.log(k) - math.lgamma(k) + k * self.lr - k * rho

    def logw(self, M):
        return self.lprior_z + self.log_prior_rho(M) - self.lmix

    def features(self, L, qbar):
        """e (n, S), h (n, S), q (n,) for the fixed samples; differentiable in L and qbar."""
        # broadcast form: the backward of z @ L.T is an (S x n)(n x S) product, which cuBLAS fp64 runs ~100x slow
        h = (self.z[:, None, :] * L[None, :, :]).sum(-1)
        q = qbar * torch.exp(self.lr)
        arg = (h @ self.Phi) / torch.sqrt(1 + 2 * q)[:, None]
        e = torch.erf(arg) @ self.WPhi
        return e, h, q


class FixedFreeEnergy(FreeEnergy):
    def __init__(self, p, nu=None, device=None, verbose=True, n_log2=15, seed=0):
        super().__init__(p, nu, device, verbose)
        self.mc = HiddenMC(p, self.dev, n_log2=n_log2, seed=seed)
        self.mc_ready = False

    # ------------------------------------------------------------------ proposal from the current state
    def build_proposal(self, J, T, M, t):
        """Lattice scan of the hidden-field distribution along its drive direction: local minima of
        -log p(s) (q_h at its conditional mode) become mixture components."""
        p, mid = self.p, self.mid
        G = p.sigB ** 2 * J
        L = scaled_cholesky(G)
        qbar = p.sigB ** 2 * T
        mid.groups = Groups()
        mid.setup(L, t, qbar, M, recount=True)
        c = mid.cells(t)
        A = mid.action_grid(c)                                   # (ns, nqh)
        lab = basin_labels(A)
        roots = torch.unique(lab).tolist()
        Lc = c["L"]
        tot = torch.logsumexp(Lc.reshape(-1), 0)
        S = self.S
        I = torch.eye(S, dtype=DT, device=self.dev)
        comps = [dict(mu=torch.zeros(S, dtype=DT, device=self.dev), C=I.clone(), lmu=0.0, lsd=2.0 * math.sqrt(2 / M))]
        u = mid.u
        far = []
        for r in roots:
            i_s, i_q = divmod(r, A.shape[1])
            mass = float(torch.logsumexp(torch.where(lab == r, Lc, -torch.inf).reshape(-1), 0) - tot)
            s_b = float(mid.s[i_s])
            self.log(f"    hidden-field basin: s={float(mid.s[i_s]):.2f}, log q/qbar="
                     f"{float(torch.log(mid.qh[i_q] / qbar)):.3f}, log mass {mass:.1f}")
            if s_b < 1.0 or mass < -60:
                continue
            # width along u from the curvature of the s-profile at the minimum
            prof = -torch.logsumexp(Lc - mid.logws[:, None], 1)
            j = min(max(i_s, 1), len(prof) - 2)
            curv = float(prof[j + 1] - 2 * prof[j] + prof[j - 1]) / float(mid.s[1] - mid.s[0]) ** 2
            w_s = 1.5 / math.sqrt(max(curv, 1e-2))
            lq = float(torch.log(mid.qh[i_q] / qbar))
            # width w_s along u, prior width transverse: covariance I + (w^2 - 1) u u^T (an isotropic width w in
            # every direction wastes nearly all samples when the hidden field has several dimensions)
            ws = max(w_s, 1.0)
            Cf = torch.linalg.cholesky(I + (ws * ws - 1.0) * torch.outer(u, u))
            for sign in (1.0, -1.0):
                far.append(dict(mu=sign * s_b * u, C=Cf.clone(), lmu=lq, lsd=2.0 * math.sqrt(2 / M)))
        # far components found earlier stay in the mixture (unless re-found within one width): the proposal only
        # has to cover the distribution, and dropping a component that still carries mass biases the integrals
        for c_old in getattr(self, "far_comps", []):
            if all(float((c_old["mu"] - c["mu"]).norm()) > float(c["C"][0, 0]) for c in far):
                far.append(c_old)
        self.far_comps = far
        near = self.fit_near(L, qbar, M, t, far) if self.mc_ready else None
        if near is not None:
            comps.append(near)
        comps += far
        self.log(f"    proposal: prior{' + fitted near basin' if near is not None else ''} + {len(far)} far components")
        self.mc.set_proposal(comps, M)
        self.mc_ready = True

    def fit_near(self, L, qbar, M, t, far, inflate=1.2):
        """Gaussian fitted to the near basin of the current hidden-field distribution (second moments of the
        weighted samples closer to the origin than half way to any far basin; mean 0 by the symmetry h -> -h),
        widened by `inflate`. Lowers the variance of the bulk integrals, which set G - H."""
        with torch.no_grad():
            e, h, q = self.mc.features(L, qbar)
            lt = self.mc.logw(M) + 0.5 * self.p.gt * (e @ t) ** 2
            w = torch.exp(lt - torch.logsumexp(lt, 0))
            z, lr = self.mc.z, self.mc.lr
            mask = torch.ones(len(z), dtype=torch.bool, device=self.dev)
            for c in far:
                mu = c["mu"]
                mask &= (z @ mu).abs() < 0.5 * float(mu @ mu)
            wn = torch.where(mask, w, torch.zeros_like(w))
            wn = wn / wn.sum()
            C = _tiny_gram(z * wn[:, None], z)
            m = float(wn @ lr)
            sd = math.sqrt(float(wn @ (lr - m) ** 2))
            try:
                Lc = torch.linalg.cholesky(C)
            except torch.linalg.LinAlgError:
                return None
        return dict(mu=torch.zeros_like(t), C=inflate * Lc, lmu=m, lsd=inflate * sd)

    # ------------------------------------------------------------------ hidden part on the fixed samples
    def _hidden_val(self, L, qbar, M, t):
        mc, p = self.mc, self.p
        e, h, q = mc.features(L, qbar)
        lw = mc.logw(M)
        tb = e @ t
        lt = lw + 0.5 * p.gt * tb * tb
        logZ = torch.logsumexp(lt, 0) - torch.logsumexp(lw, 0)
        return logZ, e, h, q, lt

    def solve_t(self, L, qbar, M, t0, tol=1e-10, maxit=100):
        p = self.p
        y = self.y
        I = torch.eye(self.S, dtype=DT, device=self.dev)
        c0 = p.N2 / p.chi
        with torch.no_grad():
            e, h, q = self.mc.features(L, qbar)
            lw = self.mc.logw(M)

            def phi(t):
                tb = e @ t
                lz = torch.logsumexp(lw + 0.5 * p.gt * tb * tb, 0) - torch.logsumexp(lw, 0)
                return 0.5 * p.lam * float(t @ t) + c0 * float(lz) - float(y @ t), tb

            t = t0.clone()
            f, tb = phi(t)
            for it in range(maxit):
                lt = lw + 0.5 * p.gt * tb * tb
                w = torch.exp(lt - torch.logsumexp(lt, 0))
                K = (e * w[:, None]).T @ e
                g = p.lam * t + p.sigA ** 2 * (K @ t) - y
                if float(g.abs().max()) < tol:
                    break
                v = tb[:, None] * e
                mv = w @ v
                Cov = (v * w[:, None]).T @ v - torch.outer(mv, mv)
                Hm = p.lam * I + p.sigA ** 2 * (K + p.gt * Cov)
                step = -torch.linalg.solve(Hm, g)
                a, slope = 1.0, float(g @ step)
                while True:
                    f1, tb1 = phi(t + a * step)
                    if f1 <= f + 1e-4 * a * slope or a < 1e-12:
                        break
                    a *= 0.5
                t, f, tb = t + a * step, f1, tb1
            ess = float(1 / (w * w).sum())
        return t, dict(K=K, w=w, ess=ess, e=e, h=h, q=q, grad=float(g.abs().max()))

    def _value_grad(self, x, M):
        """W and dW/dx (exact for the discretised W). None if the read-in moments are infeasible."""
        p = self.p
        x = x.clone().requires_grad_(True)
        J, T = self.decode(x)
        Jd, Td = J.detach(), float(T.detach())
        try:
            torch.linalg.cholesky(Jd)
        except torch.linalg.LinAlgError:
            return None, None
        if math.isfinite(p.N1):
            Dmin, Lam, lam, pw, gres = self.kl_star(Jd, Td)
            if not (gres < 1e-6 and math.isfinite(Dmin)):
                return None, None
        else:
            Dmin, Lam, lam, pw, gres = 0.0, torch.zeros_like(Jd), 0.0, None, 0.0
        G = p.sigB ** 2 * J
        L = scaled_cholesky(G)
        qbar = p.sigB ** 2 * T
        with torch.no_grad():
            t0 = self.t if self.t is not None else self.y / p.lam
            t, st = self.solve_t(L.detach(), float(qbar.detach()), M, t0)
        self.t = t
        logZ, e, h, q, lt = self._hidden_val(L, qbar, M, t)
        phi_t = p.N2 * logZ + p.chi * (0.5 * p.lam * (t @ t) - self.y @ t)
        Nr = p.N1 if math.isfinite(p.N1) else 0.0
        obj = phi_t + 0.5 * Nr * ((J * Lam).sum() + T * lam)
        obj.backward()
        W = Nr * Dmin + float(phi_t)
        self.nev += 1
        w = st["w"]
        Hm = (h.detach() * w[:, None]).T @ h.detach()
        Eq = float(w @ q.detach())
        self.last = dict(J=Jd, T=Td, M=M, Lam=Lam, lam=lam, pw=pw, dual_res=gres, Dmin=Dmin, t=t, K=st["K"],
                         H=Hm, Eq=Eq, ess=st["ess"], tgrad=st["grad"], qbar=float(qbar.detach()))
        return W, x.grad.detach()

    # ------------------------------------------------------------------ driver
    def solve_fixed(self, x0=None, M0=None, outer=10, mtol=1e-6, gtol=1e-6):
        t0 = time.time()
        p = self.p
        if x0 is None:
            x0 = self.encode(self.J0, self.T0)
        M = M0
        if M is None:
            if math.isfinite(p.N1):
                self.kl_star(self.J0, self.T0)
            self.last = dict(pw=torch.exp(self.lp0.reshape(-1)), J=self.J0, T=self.T0)
            M = self.M_from_readin()
        x = x0
        J, T = self.decode(x)
        t = self.y / p.lam
        # an initial t from the lattice solver (any reasonable t; the proposal only needs the basins)
        L = scaled_cholesky(p.sigB ** 2 * J.detach())
        t, _, _ = solve_t_convex(self.mid, p, L, p.sigB ** 2 * float(T), M, t)
        self.t = t
        hist = []
        for k in range(outer):
            J, T = self.decode(x)
            self.build_proposal(J.detach(), float(T.detach()), M, self.t)
            if k == 0:                       # second pass: fit the near component on the first samples
                with torch.no_grad():
                    Lk = scaled_cholesky(p.sigB ** 2 * J.detach())
                    self.t, _ = self.solve_t(Lk, p.sigB ** 2 * float(T), M, self.t)
                self.build_proposal(J.detach(), float(T.detach()), M, self.t)
            if math.isfinite(p.N1):
                x, W, info = self.maximise(x, M, gtol=gtol)
            else:
                W, g = self._value_grad(x, M)
                info = type("I", (), dict(nit=0, jac=-g.cpu().numpy()))()
            if math.isfinite(p.N1):
                Mn = self.M_from_readin()
            else:
                Mn = M
            L_ = self.last
            fb = (self.y - p.lam * L_["t"]).cpu().numpy()
            self.log(f"  outer {k}: W={W:.4f}  |grad|={np.abs(info.jac).max():.1e}  mu1={fb[0]:.6f} mu3={fb[1] / p.eps:.6f}"
                     f"  ESS={L_['ess']:.0f}/{len(self.mc.z)}  M {M:.4f} -> {Mn:.4f}  ({self.nev} evals, "
                     f"{time.time() - t0:.1f}s)")
            hist.append((W, fb.copy()))
            stop = abs(Mn / M - 1) < mtol and k > 0 and abs(hist[-1][1][1] - hist[-2][1][1]) < 1e-7
            M = Mn
            if stop or not math.isfinite(p.N1):
                break
        L_ = self.last
        fb = (self.y - p.lam * L_["t"]).cpu().numpy()
        out = dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, ess=L_["ess"], nsamples=len(self.mc.z),
                   J=L_["J"].cpu().numpy().tolist(), T=L_["T"], t=L_["t"].cpu().numpy().tolist(),
                   H=L_["H"].cpu().numpy().tolist(), Eq=L_["Eq"], x=x.detach().cpu().numpy().tolist(),
                   walltime_s=time.time() - t0)
        if L_["pw"] is not None:
            pw = L_["pw"].reshape(self.ri.W.shape)
            out["readin_mass_w0_gt_1"] = float(pw[self.ri.W > 1.0].sum())
            out["d_w0sq"] = float(self.p.d * (pw * self.ri.W ** 2).sum())
        return out


# ======================================================================================================================
# Read-in distribution parametrised by its exponents (any n_max)
# ======================================================================================================================
from fcn3_matrix_solver import cn_const  # noqa: E402


class ThetaFreeEnergy(FixedFreeEnergy):
    """The read-in coefficients are c_n(w) = a_n r^n with r = w0 / sqrt(1 + 2|w|^2), so J_nm = a_n a_m E[r^(n+m)]
    depends only on n + m: J is fixed by the n_max moments E[r^2k], k = 1..n_max, and the reachable J form an
    n_max-dimensional slice of the n(n+1)/2-dimensional matrix space (for n_max >= 5 a strict slice, so moves in J
    leave it). Here the read-in distribution is parametrised by its exponents,
        p_theta(w) = p0(w) exp(-1/2 sum_k theta_k r^2k / s_k - 1/2 theta_perp P_perp / T0) / Z(theta),
    (s_k = prior moments, for scale) which is the form (E5) has, and every theta is feasible:
        W(theta) = N1 (log Z(theta) + theta . mbar(theta) / 2) + Phi(J(mbar), T(mbar)),   [first term = -KL(p_theta|p0)]
    with the gradient by autograd through the read-in grid and the fixed hidden-field samples."""

    def __init__(self, p, **kw):
        super().__init__(p, **kw)
        ri = self.ri
        S = self.S
        ks = list(p.ks)
        r2 = (ri.W ** 2 / (1 + 2 * ri.W ** 2 + 2 * ri.Q)).reshape(-1)
        self.K = p.nmax                                         # moments r^2k, k = 1..nmax
        R = torch.stack([r2 ** k for k in range(1, self.K + 1)])            # (K, npts)
        pw0 = torch.exp(self.lp0.reshape(-1))
        self.sk = R @ pw0
        self.Tst = torch.cat([R / self.sk[:, None], (self.Pp / self.T0)[None]], 0)
        a = torch.tensor([cn_const(n) for n in ks], dtype=DT, device=self.dev)
        E = torch.zeros(self.K, S, S, dtype=DT, device=self.dev)
        for i, n in enumerate(ks):
            for j, m in enumerate(ks):
                E[(n + m) // 2 - 1, i, j] = a[i] * a[j]
        self.E = E
        self.nth = self.K + 1

    def moments(self, th):
        lw = self.lp0.reshape(-1) - 0.5 * (th @ self.Tst)
        lz = torch.logsumexp(lw, 0)
        pw = torch.exp(lw - lz)
        mbar = self.Tst @ pw
        D = lz + 0.5 * (th @ mbar)                              # -KL(p_theta | p0)
        J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
        T = self.T0 * mbar[-1]
        return D, J, T, pw

    def _value_grad(self, th, M):
        p = self.p
        th = th.clone().requires_grad_(True)
        D, J, T, pw = self.moments(th)
        Jd, Td = J.detach(), float(T.detach())
        G = p.sigB ** 2 * J
        try:
            L = scaled_cholesky(G)
        except torch.linalg.LinAlgError:
            return None, None
        qbar = p.sigB ** 2 * T
        with torch.no_grad():
            t0 = self.t if self.t is not None else self.y / p.lam
            t, st = self.solve_t(L.detach(), float(qbar.detach()), M, t0)
        self.t = t
        logZ, e, h, q, lt = self._hidden_val(L, qbar, M, t)
        phi_t = p.N2 * logZ + p.chi * (0.5 * p.lam * (t @ t) - self.y @ t)
        obj = p.N1 * D + phi_t
        obj.backward()
        self.nev += 1
        w = st["w"]
        self.last = dict(J=Jd, T=Td, M=M, pw=pw.detach(), Dmin=float(D), t=t, K=st["K"],
                         H=_tiny_gram(h.detach() * w[:, None], h.detach()), Eq=float(w @ q.detach()), ess=st["ess"],
                         tgrad=st["grad"], qbar=float(qbar.detach()), theta=th.detach().clone())
        return float(obj), th.grad.detach()

    def _hess(self, th, g, M):
        """-Hessian columns by central differences of the exact gradient, step 1e-4 / sd_theta(T_i)."""
        with torch.no_grad():
            _, _, _, pw = self.moments(th)
            m = self.Tst @ pw
            sd = torch.sqrt(((self.Tst - m[:, None]) ** 2) @ pw).clamp_min(1e-300)
        n = len(th)
        Hm = torch.zeros(n, n, dtype=DT, device=self.dev)
        t_save = self.t
        for i in range(n):
            hi = 1e-4 / float(sd[i])
            e = torch.zeros_like(th)
            e[i] = hi
            _, gp = self._value_grad(th + e, M)
            self.t = t_save
            _, gm = self._value_grad(th - e, M)
            self.t = t_save
            Hm[:, i] = (gp - gm) / (2 * hi)
        return 0.5 * (Hm + Hm.T)

    def ascend(self, th, M, tol=1e-8, maxit=60):
        """Newton ascent on W(theta): eigenvalues of the Hessian made negative (|lambda|), backtracking on W.
        Converged when the Newton decrement g.(-H)^-1 g < tol (W units)."""
        W, g = self._value_grad(th, M)
        for it in range(maxit):
            Hm = self._hess(th, g, M)
            ev, U = torch.linalg.eigh(Hm)
            nneg = int((ev > 0).sum())
            lam = ev.abs().clamp_min(1e-12 * float(ev.abs().max()))
            step = U @ ((U.T @ g) / lam)                        # ascent direction (|H|)^-1 g
            dec = float(g @ step)
            if dec < tol:
                break
            a = 1.0
            while a > 1e-6:
                W1, g1 = self._value_grad(th + a * step, M)
                if W1 is not None and W1 >= W + 1e-4 * a * dec - 1e-9 * abs(W):
                    break
                a *= 0.5
            else:
                self.log(f"    ascent: line search failed (decrement {dec:.2e})")
                break
            th, W, g = th + a * step, W1, g1
            self.log(f"    ascent {it}: W={W:.6f}  decrement {dec:.2e}  step {a:.3g}  "
                     f"positive Hessian eigenvalues {nneg}")
        W, g = self._value_grad(th, M)
        Hm = self._hess(th, g, M)
        ev = torch.linalg.eigvalsh(Hm)
        return th, W, g, ev

    def solve_theta(self, th0=None, M0=None, outer=8, mtol=1e-6):
        t0 = time.time()
        p = self.p
        th = torch.zeros(self.nth, dtype=DT, device=self.dev) if th0 is None else th0.clone()
        with torch.no_grad():
            D, J, T, pw = self.moments(th)
        self.last = dict(pw=pw, J=J, T=float(T))
        M = self.M_from_readin() if M0 is None else M0
        L = scaled_cholesky(p.sigB ** 2 * J)
        self.t, _, _ = solve_t_convex(self.mid, p, L, p.sigB ** 2 * float(T), M, self.y / p.lam)
        for k in range(outer):
            with torch.no_grad():
                D, J, T, pw = self.moments(th)
            self.build_proposal(J, float(T), M, self.t)
            with torch.no_grad():
                Lk = scaled_cholesky(p.sigB ** 2 * J)
                self.t, _ = self.solve_t(Lk, p.sigB ** 2 * float(T), M, self.t)
            self.build_proposal(J, float(T), M, self.t)
            th, W, g, ev = self.ascend(th, M)
            Mn = self.M_from_readin()
            L_ = self.last
            fb = (self.y - p.lam * L_["t"]).cpu().numpy()
            self.log(f"  outer {k}: W={W:.6f} mu1={fb[0]:.6f} mu3={fb[1] / p.eps:.6f} ESS={L_['ess']:.0f}/"
                     f"{len(self.mc.z)}  M {M:.4f} -> {Mn:.4f}  Hessian eigenvalues max {float(ev.max()):.2e}  "
                     f"({self.nev} evals, {time.time() - t0:.1f}s)")
            done = abs(Mn / M - 1) < mtol
            M = Mn
            if done:
                break
        L_ = self.last
        fb = (self.y - p.lam * L_["t"]).cpu().numpy()
        pw = L_["pw"].reshape(self.ri.W.shape)
        return dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, ess=L_["ess"], nsamples=len(self.mc.z),
                    J=L_["J"].cpu().numpy().tolist(), T=L_["T"], t=L_["t"].cpu().numpy().tolist(),
                    H=L_["H"].cpu().numpy().tolist(), Eq=L_["Eq"], theta=th.cpu().numpy().tolist(),
                    hess_eig=ev.cpu().numpy().tolist(), readin_mass_w0_gt_1=float(pw[self.ri.W > 1.0].sum()),
                    d_w0sq=float(p.d * (pw * self.ri.W ** 2).sum()), walltime_s=time.time() - t0)


class MomentFreeEnergy(ThetaFreeEnergy):
    """W as a function of the independent read-in moments: x = log(E[r^2k] / s_k), k = 1..n_max, and log(T / T0)
    (prior: x = 0). J = sum_k s_k e^{x_k} E_k is then always on the reachable slice. The read-in part is
    N1 min_theta [log E0 exp(-theta.Tst/2) + theta.mbar/2] (= -KL*, convex in theta, Newton); its gradient in mbar
    is N1 theta/2 (envelope). The hidden part is differentiated by autograd as in FixedFreeEnergy."""

    def __init__(self, p, **kw):
        super().__init__(p, **kw)
        self.th_dual = None

    def dual(self, mbar, tol=1e-11, maxit=200):
        lp0 = self.lp0.reshape(-1)
        Tst = self.Tst
        n = len(mbar)
        I = torch.eye(n, dtype=DT, device=self.dev)
        for attempt in range(2):
            th = torch.zeros(n, dtype=DT, device=self.dev) if (self.th_dual is None or attempt) else self.th_dual.clone()

            def D(th):
                lw = lp0 - 0.5 * (th @ Tst)
                lz = torch.logsumexp(lw, 0)
                return float(lz + 0.5 * th @ mbar), lw - lz

            f, lw = D(th)
            for it in range(maxit):
                pw = torch.exp(lw)
                ET = Tst @ pw
                g = 0.5 * (mbar - ET)
                res = float((g.abs() / mbar.abs()).max())
                if res < tol:
                    break
                Dv = Tst - ET[:, None]
                Hm = 0.25 * (Dv * pw) @ Dv.T
                step = -torch.linalg.lstsq(Hm + 1e-14 * float(torch.diagonal(Hm).max()) * I, g[:, None],
                                           rcond=1e-14).solution[:, 0]
                a, slope = 1.0, float(g @ step)
                while True:
                    f1, lw1 = D(th + a * step)
                    if f1 <= f + 1e-4 * a * slope or a < 1e-12:
                        break
                    a *= 0.5
                th, f, lw = th + a * step, f1, lw1
            if res < 1e-8:
                break
        self.th_dual = th.clone() if res < 1e-8 else None
        return f, th, torch.exp(lw), res

    def _value_grad(self, x, M):
        p = self.p
        x = x.clone().requires_grad_(True)
        mbar = torch.exp(x)
        with torch.no_grad():
            Dmin, th, pw, res = self.dual(mbar.detach())
        if not (res < 1e-8 and math.isfinite(Dmin)):
            return None, None
        J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
        T = self.T0 * mbar[-1]
        Jd, Td = J.detach(), float(T.detach())
        try:
            L = scaled_cholesky(p.sigB ** 2 * J)
        except torch.linalg.LinAlgError:
            return None, None
        qbar = p.sigB ** 2 * T
        with torch.no_grad():
            t0 = self.t if self.t is not None else self.y / p.lam
            t, st = self.solve_t(L.detach(), float(qbar.detach()), M, t0)
        self.t = t
        logZ, e, h, q, lt = self._hidden_val(L, qbar, M, t)
        phi_t = p.N2 * logZ + p.chi * (0.5 * p.lam * (t @ t) - self.y @ t)
        obj = phi_t + p.N1 * 0.5 * (th @ mbar)
        obj.backward()
        self.nev += 1
        w = st["w"]
        self.last = dict(J=Jd, T=Td, M=M, pw=pw, Dmin=Dmin, dual_res=res, t=t, K=st["K"],
                         H=(h.detach() * w[:, None]).T @ h.detach(), Eq=float(w @ q.detach()), ess=st["ess"],
                         tgrad=st["grad"], qbar=float(qbar.detach()), theta=th)
        return p.N1 * Dmin + float(phi_t), x.grad.detach()

    def newton(self, x, M, tol=1e-7, maxit=40):
        """Newton on grad W = 0 with a central-difference Hessian of the exact gradient; eigenvalues made negative
        (ascent), backtracking on W, steps capped at 1 in log-moment units. Converged when the Newton decrement
        g.(|H|)^-1 g < tol."""
        W, g = self._value_grad(x, M)
        if W is None:
            raise RuntimeError("starting point is infeasible")
        n = len(x)
        for it in range(maxit):
            Hm = torch.zeros(n, n, dtype=DT, device=self.dev)
            t_save = self.t
            for i in range(n):
                e = torch.zeros_like(x)
                e[i] = 1e-5
                Wp, gp = self._value_grad(x + e, M)
                self.t = t_save
                Wm, gm = self._value_grad(x - e, M)
                self.t = t_save
                if Wp is None or Wm is None:
                    raise RuntimeError("Hessian probe left the feasible set")
                Hm[:, i] = (gp - gm) / 2e-5
            Hm = 0.5 * (Hm + Hm.T)
            ev, U = torch.linalg.eigh(Hm)
            lam = ev.abs().clamp_min(1e-10 * float(ev.abs().max()))
            step = U @ ((U.T @ g) / lam)
            dec = float(g @ step)
            if dec < tol:
                break
            mx = float(step.abs().max())
            if mx > 1.0:
                step = step / mx
            a = 1.0
            while a > 1e-8:
                W1, g1 = self._value_grad(x + a * step, M)
                if W1 is not None and W1 >= W + 1e-4 * a * float(g @ step) - 1e-10 * abs(W):
                    break
                a *= 0.5
            else:
                self.log(f"    newton: line search failed (decrement {dec:.2e})")
                break
            x, W, g = x + a * step, W1, g1
            self.log(f"    newton {it}: W={W:.6f}  decrement {dec:.2e}  step {a:.3g}  positive eigenvalues "
                     f"{int((ev > 0).sum())}  |g|={float(g.abs().max()):.1e}")
        return x, W, g, ev

    def solve_moments(self, x0=None, M0=None, outer=8, mtol=1e-6, method="newton"):
        t0 = time.time()
        p = self.p
        x = torch.zeros(self.nth, dtype=DT, device=self.dev) if x0 is None else x0.clone()
        mbar = torch.exp(x)
        Dmin, th, pw, res = self.dual(mbar)
        J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
        T = float(self.T0 * mbar[-1])
        self.last = dict(pw=pw, J=J, T=T)
        M = self.M_from_readin() if M0 is None else M0
        self.t, _, _ = solve_t_convex(self.mid, p, scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * T, M,
                                      self.y / p.lam)
        ev = torch.zeros(1)
        for k in range(outer):
            mbar = torch.exp(x)
            J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
            T = float(self.T0 * mbar[-1])
            self.build_proposal(J, T, M, self.t)
            with torch.no_grad():
                self.t, _ = self.solve_t(scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * T, M, self.t)
            self.build_proposal(J, T, M, self.t)
            if method == "newton":
                x, W, g, ev = self.newton(x, M)
            else:
                x, W, info = self.maximise(x, M, gtol=1e-7 * p.N1)
            Mn = self.M_from_readin()
            L_ = self.last
            fb = (self.y - p.lam * L_["t"]).cpu().numpy()
            self.log(f"  outer {k}: W={W:.6f} mu1={fb[0]:.6f} mu3={fb[1] / p.eps:.6f} ESS={L_['ess']:.0f}/"
                     f"{len(self.mc.z)}  M {M:.4f} -> {Mn:.4f}  ({self.nev} evals, {time.time() - t0:.1f}s)")
            done = abs(Mn / M - 1) < mtol
            M = Mn
            if done:
                break
        L_ = self.last
        fb = (self.y - p.lam * L_["t"]).cpu().numpy()
        pw = L_["pw"].reshape(self.ri.W.shape)
        return dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, ess=L_["ess"], nsamples=len(self.mc.z),
                    J=L_["J"].cpu().numpy().tolist(), T=L_["T"], t=L_["t"].cpu().numpy().tolist(),
                    H=L_["H"].cpu().numpy().tolist(), Eq=L_["Eq"], x=x.cpu().numpy().tolist(),
                    hess_eig=ev.cpu().numpy().tolist(), readin_mass_w0_gt_1=float(pw[self.ri.W > 1.0].sum()),
                    d_w0sq=float(p.d * (pw * self.ri.W ** 2).sum()), walltime_s=time.time() - t0)


class WhitenedFreeEnergy(ThetaFreeEnergy):
    """W over the exponents of the read-in distribution, with no inner dual solve:
        log p(w) = log p0(w) - 1/2 theta_ref . Tst(w) - 1/2 phi . S(w) - log Z,
    where S are the statistics Tst (r^2k / s_k, k = 1..n_max, and P_perp / T0) centred and whitened under the
    distribution at theta_ref (eigen-decomposition of their covariance; directions with variance below
    `rank_tol` x the largest are dropped: the grid cannot resolve them). In these coordinates the curvature of
    -KL is close to -identity, so Newton is well conditioned at every n_max; the raw r^2k are nearly collinear
    (curvature spread ~1e12 at n_max = 11). Before every Newton step theta_ref absorbs phi and S is re-whitened.
    -KL(p | p0) = -sum_w p (log p - log p0) is evaluated directly, and every phi is feasible."""

    rank_tol = 1e-13

    def rebase(self, phi=None):
        with torch.no_grad():
            if phi is not None:
                self.theta_ref = self.theta_ref + self.Umap @ phi
            lw = self.lp0.reshape(-1) - 0.5 * (self.theta_ref @ self.Tst)
            pw = torch.exp(lw - torch.logsumexp(lw, 0))
            mu = self.Tst @ pw
            Dv = self.Tst - mu[:, None]
            C = ((Dv * pw) @ Dv.T).cpu()
            dsc = torch.sqrt(torch.diagonal(C)).clamp_min(1e-300)
            ev, U = torch.linalg.eigh(C / torch.outer(dsc, dsc))       # Jacobi-scaled, on the CPU
            keep = ev > self.rank_tol * float(ev.max())
            # whitened statistic S = Wm (Tst - mu), with Wm = diag(ev)^-1/2 U^T diag(dsc)^-1
            Wm = (U[:, keep] / torch.sqrt(ev[keep])).T / dsc
            self.Wm = Wm.to(self.dev)
            self.S_w = self.Wm @ Dv
            self.Umap = self.Wm.T                       # phi . S = (Wm^T phi) . Tst + const
            self.nphi = int(keep.sum())

    def moments(self, phi):
        lw = self.lp0.reshape(-1) - 0.5 * (self.theta_ref @ self.Tst) - 0.5 * (phi @ self.S_w)
        lw = lw - torch.logsumexp(lw, 0)
        pw = torch.exp(lw)
        mbar = self.Tst @ pw
        D = -(pw * (lw - self.lp0.reshape(-1))).sum()           # -KL(p | p0)
        J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
        T = self.T0 * mbar[-1]
        return D, J, T, pw

    def newton(self, M, tol=1e-8, maxit=40, h=1e-4):
        p = self.p
        phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
        W, g = self._value_grad(phi, M)
        ev = torch.zeros(1)
        for it in range(maxit):
            n = len(phi)
            Hm = torch.zeros(n, n, dtype=DT, device=self.dev)
            t_save = self.t
            for i in range(n):
                e = torch.zeros_like(phi)
                e[i] = h
                _, gp = self._value_grad(phi + e, M)
                self.t = t_save
                _, gm = self._value_grad(phi - e, M)
                self.t = t_save
                Hm[:, i] = (gp - gm) / (2 * h)
            Hm = 0.5 * (Hm + Hm.T)
            ev, U = torch.linalg.eigh(Hm)
            lam = ev.abs().clamp_min(1e-10 * float(ev.abs().max()))
            step = U @ ((U.T @ g) / lam)
            dec = float(g @ step)
            if dec < tol:
                break
            mx = float(step.abs().max())
            if mx > 2.0:                                # at most 2 standard deviations of any statistic
                step = step * (2.0 / mx)
            a = 1.0
            while a > 1e-8:
                W1, g1 = self._value_grad(phi + a * step, M)
                if W1 is not None and W1 >= W + 1e-4 * a * float(g @ step) - 1e-10 * abs(W):
                    break
                a *= 0.5
            else:
                self.log(f"    newton: line search failed (decrement {dec:.2e})")
                break
            self.rebase(phi + a * step)                 # theta_ref absorbs the step; re-whiten
            phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
            W, g = self._value_grad(phi, M)
            self.log(f"    newton {it}: W={W:.6f}  decrement {dec:.2e}  step {a:.3g}  rank {self.nphi}  "
                     f"positive eigenvalues {int((ev > 0).sum())}")
        return W, g, ev

    def solve_whitened(self, theta0=None, M0=None, outer=8, mtol=1e-6):
        t0 = time.time()
        p = self.p
        self.theta_ref = torch.zeros(self.nth, dtype=DT, device=self.dev) if theta0 is None else theta0.clone()
        self.rebase()
        phi0 = torch.zeros(self.nphi, dtype=DT, device=self.dev)
        with torch.no_grad():
            D, J, T, pw = self.moments(phi0)
        self.last = dict(pw=pw, J=J, T=float(T))
        M = self.M_from_readin() if M0 is None else M0
        self.t, _, _ = solve_t_convex(self.mid, p, scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * float(T), M,
                                      self.y / p.lam)
        W, ev = None, torch.zeros(1)
        for k in range(outer):
            with torch.no_grad():
                D, J, T, pw = self.moments(torch.zeros(self.nphi, dtype=DT, device=self.dev))
            self.build_proposal(J, float(T), M, self.t)
            with torch.no_grad():
                self.t, _ = self.solve_t(scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * float(T), M, self.t)
            self.build_proposal(J, float(T), M, self.t)
            self.rebase()
            W, g, ev = self.newton(M)
            Mn = self.M_from_readin()
            L_ = self.last
            fb = (self.y - p.lam * L_["t"]).cpu().numpy()
            self.log(f"  outer {k}: W={W:.6f} mu1={fb[0]:.6f} mu3={fb[1] / p.eps:.6f} ESS={L_['ess']:.0f}/"
                     f"{len(self.mc.z)}  M {M:.4f} -> {Mn:.4f}  ({self.nev} evals, {time.time() - t0:.1f}s)")
            done = abs(Mn / M - 1) < mtol
            M = Mn
            if done:
                break
        L_ = self.last
        fb = (self.y - p.lam * L_["t"]).cpu().numpy()
        pw = L_["pw"].reshape(self.ri.W.shape)
        return dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, ess=L_["ess"], nsamples=len(self.mc.z),
                    J=L_["J"].cpu().numpy().tolist(), T=L_["T"], t=L_["t"].cpu().numpy().tolist(),
                    H=L_["H"].cpu().numpy().tolist(), Eq=L_["Eq"], theta=self.theta_ref.cpu().numpy().tolist(),
                    hess_eig=ev.cpu().numpy().tolist(), readin_mass_w0_gt_1=float(pw[self.ri.W > 1.0].sum()),
                    d_w0sq=float(p.d * (pw * self.ri.W ** 2).sum()), walltime_s=time.time() - t0)


class FastFreeEnergy(WhitenedFreeEnergy):
    """WhitenedFreeEnergy with the cost taken out:
      * exact Hessian: W(phi) = min_t f(phi, t), so d2W = f_pp - f_pt f_tt^-1 f_tp, from one vectorised autograd
        Hessian of f in (phi, t) (no finite differences, no extra t solves);
      * the t solve stops when the Newton step stops changing t (it used to run to its iteration cap);
      * the proposal scan (which only has to locate hidden-field basins; the importance weights make the integrals
        exact whatever the proposal) uses a coarse lattice."""

    def __init__(self, p, **kw):
        super().__init__(p, **kw)
        from fcn3_matrix_solver import Middle
        coarse = Numerics(zeta_levels=((1e-2, 4), (1e-4, 2), (0.0, 1)), nqh=80, ds=0.3, smax=45)
        self.mid = Middle(p, coarse, self.dev)

    def solve_t(self, L, qbar, M, t0, tol=1e-13, maxit=60):
        p, y = self.p, self.y
        I = torch.eye(self.S, dtype=DT, device=self.dev)
        c0 = p.N2 / p.chi
        with torch.no_grad():
            e, h, q = self.mc.features(L, qbar)
            lw = self.mc.logw(M)
            lw0 = torch.logsumexp(lw, 0)

            def phi(t):
                tb = e @ t
                return 0.5 * p.lam * (t @ t) + c0 * (torch.logsumexp(lw + 0.5 * p.gt * tb * tb, 0) - lw0) - y @ t, tb

            t = t0.clone()
            f, tb = phi(t)
            for it in range(maxit):
                lt = lw + 0.5 * p.gt * tb * tb
                w = torch.exp(lt - torch.logsumexp(lt, 0))
                ew = e * w[:, None]
                K = _tiny_gram(ew, e)
                g = p.lam * t + p.sigA ** 2 * (K @ t) - y
                v = tb[:, None] * e
                mv = (v * w[:, None]).sum(0)
                Cov = _tiny_gram(v * w[:, None], v) - torch.outer(mv, mv)
                Hm = p.lam * I + p.sigA ** 2 * (K + p.gt * Cov)
                step = -torch.linalg.solve(Hm, g)
                if float((step.abs() / t.abs().clamp_min(1e-300)).max()) < tol:
                    break
                f1, tb1 = phi(t + step)
                a = 1.0
                while float(f1) > float(f) + 1e-4 * a * float(g @ step) + 1e-14 * abs(float(f)) and a > 1e-10:
                    a *= 0.5
                    f1, tb1 = phi(t + a * step)
                t, f, tb = t + a * step, f1, tb1
            lt = lw + 0.5 * p.gt * tb * tb
            w = torch.exp(lt - torch.logsumexp(lt, 0))
            K = _tiny_gram(e * w[:, None], e)
            g = p.lam * t + p.sigA ** 2 * (K @ t) - y
            ess = float(1 / (w * w).sum())
        return t, dict(K=K, w=w, ess=ess, e=e, h=h, q=q, grad=float(g.abs().max()), its=it)

    def f_obj(self, z, M):
        """f(phi, t) = N1 (-KL) + N2 log Z_h(t) + chi (lam/2 |t|^2 - y.t); W(phi) = min_t f."""
        p = self.p
        n = self.nphi
        phi, t = z[:n], z[n:]
        D, J, T, pw = self.moments(phi)
        L = scaled_cholesky(p.sigB ** 2 * J)
        logZ, e, h, q, lt = self._hidden_val(L, p.sigB ** 2 * T, M, t)
        return p.N1 * D + p.N2 * logZ + p.chi * (0.5 * p.lam * (t @ t) - self.y @ t)

    def hessian(self, phi, M):
        z = torch.cat([phi, self.t]).detach()
        Hz = torch.autograd.functional.hessian(lambda zz: self.f_obj(zz, M), z, vectorize=True)
        n = self.nphi
        Hpp, Hpt, Htt = Hz[:n, :n], Hz[:n, n:], Hz[n:, n:]
        return Hpp - Hpt @ torch.linalg.solve(Htt, Hpt.T)

    def newton(self, M, tol=1e-9, maxit=80, h=None):
        phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
        W, g = self._value_grad(phi, M)
        ev = torch.zeros(1)
        for it in range(maxit):
            Hm = self.hessian(phi, M)
            Hm = 0.5 * (Hm + Hm.T)
            ev, U = torch.linalg.eigh(Hm)
            lam = ev.abs().clamp_min(1e-10 * float(ev.abs().max()))
            step = U @ ((U.T @ g) / lam)
            dec = float(g @ step)
            if dec < tol:
                break
            mx = float(step.abs().max())
            if mx > 2.0:
                step = step * (2.0 / mx)
            a, t_save = 1.0, self.t
            while a > 1e-8:
                self.t = t_save
                W1, g1 = self._value_grad(phi + a * step, M)
                if W1 is not None and W1 >= W + 1e-4 * a * float(g @ step) - 1e-12 * abs(W):
                    break
                a *= 0.5
            else:
                self.t = t_save
                self.log(f"    newton: line search failed (decrement {dec:.2e})")
                break
            self.rebase(phi + a * step)
            phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
            W, g = self._value_grad(phi, M)
            self.log(f"    newton {it}: W={W:.6f}  decrement {dec:.2e}  step {a:.3g}  rank {self.nphi}")
        self.newton_its = it
        return W, g, ev



class BlazingFreeEnergy(FastFreeEnergy):
    """FastFreeEnergy with fewer and cheaper Newton steps.
      * Read-in grid pruned to points within `prune_nats` of the peak, re-selected from the full grid at every
        rebase. Guard: if an accepted step puts more than e^-30 of the mass outside the previous active set (so the
        step was evaluated on too few points), the step is undone, the prune widened, and the step redone.
      * Two fidelities: stage 1 with 2^small hidden-field samples per proposal component and M held fixed during
        each Newton run (M from (E8) is updated between runs: far from the solution (E8) gives meaningless M), with
        the proposal rebuilt whenever its effective sample size falls below 5%; stage 2 rebuilds the proposal once
        at the stage-1 state with 2^big samples and finishes with M updated after every Newton step.
      * Exact Hessian: autograd.functional for small sample sets, torch.func (forward-over-reverse) for large."""

    prune_nats = 60.0

    def rebase(self, phi=None):
        with torch.no_grad():
            if phi is not None:
                self.theta_ref = self.theta_ref + self.Umap @ phi
            lp0 = self.lp0.reshape(-1)
            lw = lp0 - 0.5 * (self.theta_ref @ self.Tst)
            lw = lw - torch.logsumexp(lw, 0)
            self.step_excluded = -math.inf
            if getattr(self, "mask", None) is not None:
                out = lw[~self.mask]
                self.step_excluded = float(torch.logsumexp(out, 0)) if out.numel() else -math.inf
            self.mask = lw > float(lw.max()) - self.prune_nats
            self.idx = torch.nonzero(self.mask)[:, 0]
            self.Tst_a = self.Tst[:, self.idx].contiguous()
            self.lp0_a = lp0[self.idx]
            pw = torch.exp(lw[self.idx])
            pw = pw / pw.sum()
            mu = self.Tst_a @ pw
            Dv = self.Tst_a - mu[:, None]
            C = ((Dv * pw) @ Dv.T).cpu()
            dsc = torch.sqrt(torch.diagonal(C)).clamp_min(1e-300)
            ev, U = torch.linalg.eigh(C / torch.outer(dsc, dsc))
            keep = ev > self.rank_tol * float(ev.max())
            Wm = (U[:, keep] / torch.sqrt(ev[keep])).T / dsc
            self.Wm = Wm.to(self.dev)
            self.S_w = (self.Wm @ Dv).contiguous()
            self.Umap = self.Wm.T
            self.nphi = int(keep.sum())

    def moments(self, phi):
        lw = self.lp0_a - 0.5 * (self.theta_ref @ self.Tst_a) - 0.5 * (phi @ self.S_w)
        lw = lw - torch.logsumexp(lw, 0)
        pw = torch.exp(lw)
        mbar = self.Tst_a @ pw
        D = -(pw * (lw - self.lp0_a)).sum()
        J = torch.einsum("k,kij->ij", self.sk * mbar[:-1], self.E)
        T = self.T0 * mbar[-1]
        return D, J, T, pw

    def pw_full(self, pw_a):
        out = torch.zeros(self.Tst.shape[1], dtype=DT, device=self.dev)
        out[self.idx] = pw_a
        return out

    def _value_grad(self, phi, M):
        W, g = super()._value_grad(phi, M)
        if W is not None:
            self.last["pw"] = self.pw_full(self.last["pw"])
        return W, g

    def hessian(self, phi, M):
        if len(self.mc.z) < 50000:
            return FastFreeEnergy.hessian(self, phi, M)
        from torch.func import hessian as fhess
        z = torch.cat([phi, self.t]).detach()
        Hz = fhess(lambda zz: self.f_obj(zz, M))(z)
        n = self.nphi
        Hpp, Hpt, Htt = Hz[:n, :n], Hz[:n, n:], Hz[n:, n:]
        return Hpp - Hpt @ torch.linalg.solve(Htt, Hpt.T)

    def _M_now(self):
        with torch.no_grad():
            D, J, T, pw = self.moments(torch.zeros(self.nphi, dtype=DT, device=self.dev))
        self.last = dict(getattr(self, "last", {}), pw=self.pw_full(pw), J=J, T=float(T))
        return self.M_from_readin()

    def newton_M(self, M, tol, maxit=100, fold=False, ess_min=0.0):
        """Newton ascent in phi (re-whitened after every step). fold: M re-evaluated from (E8) after every accepted
        step. Returns (W, g, Hessian eigenvalues, M, status) with status 'converged', 'ess' (effective sample size
        fell below ess_min x samples: rebuild the proposal) or 'stalled'."""
        phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
        W, g = self._value_grad(phi, M)
        ev, status = torch.zeros(1), "stalled"
        for it in range(maxit):
            Hm = self.hessian(phi, M)
            Hm = 0.5 * (Hm + Hm.T)
            ev, U = torch.linalg.eigh(Hm)
            lam = ev.abs().clamp_min(1e-10 * float(ev.abs().max()))
            step = U @ ((U.T @ g) / lam)
            dec = float(g @ step)
            if dec < tol:
                status = "converged"
                break
            mx = float(step.abs().max())
            if mx > 2.0:
                step = step * (2.0 / mx)
            a, t_save = 1.0, self.t
            while a > 1e-8:
                self.t = t_save
                W1, g1 = self._value_grad(phi + a * step, M)
                if W1 is not None and W1 >= W + 1e-4 * a * float(g @ step) - 1e-12 * abs(W):
                    break
                a *= 0.5
            else:
                self.t = t_save
                self.log(f"    newton: line search failed (decrement {dec:.2e})")
                break
            th_prev, U_old = self.theta_ref.clone(), self.Umap.clone()
            self.rebase(phi + a * step)
            self.nsteps += 1
            if self.step_excluded > -30.0:
                # the step moved mass onto pruned points: undo it and redo it from the old point on the full grid
                # (the prune returns to its width at the next step)
                ex = self.step_excluded
                self.theta_ref = th_prev
                keep_nats, self.prune_nats = self.prune_nats, math.inf
                self.mask = None
                self.rebase()
                self.prune_nats = keep_nats
                self.nguard += 1
                self.log(f"    prune guard: step put e^{ex:.1f} on pruned points; redone on the full grid")
            elif fold:
                M = self._M_now()
            phi = torch.zeros(self.nphi, dtype=DT, device=self.dev)
            if (not fold and self.step_excluded <= -30.0 and U_old.shape == self.Umap.shape
                    and U_old.shape[0] == U_old.shape[1]):     # exact only when no direction is dropped
                # same distribution, new coordinates: carry W and the gradient over instead of re-evaluating
                # (phi . S = (Umap phi) . Tst + const, so g_phi = Umap^T g_theta)
                g_raw = torch.linalg.solve(U_old.T, g1)
                W, g = W1, self.Umap.T @ g_raw
            else:
                W, g = self._value_grad(phi, M)
            self.log(f"    newton {it}: W={W:.6f}  decrement {dec:.2e}  step {a:.3g}  M={M:.4f}  "
                     f"active read-in points {len(self.idx)}  ESS {self.last['ess']:.0f}")
            if self.last["ess"] < ess_min * len(self.mc.z):
                status = "ess"
                break
        return W, g, ev, M, status

    def refit(self, M, rounds=5, target=0.2, inflate=1.2):
        """Adapt the proposal to the current hidden-field distribution: each component except the prior gets the
        mean, full covariance and log-rho spread of the importance-weighted samples it is responsible for (weighted
        EM step), widened by `inflate`, and the samples are redrawn. Stops when ESS > target x samples."""
        p = self.p
        J, T = self.last["J"], self.last["T"]
        L = scaled_cholesky(p.sigB ** 2 * J)
        qbar = p.sigB ** 2 * T
        S = self.S
        I = torch.eye(S, dtype=DT, device=self.dev)
        for r in range(rounds):
            with torch.no_grad():
                e, h, q = self.mc.features(L, qbar)
                lt = self.mc.logw(M) + 0.5 * p.gt * (e @ self.t) ** 2
                w = torch.exp(lt - torch.logsumexp(lt, 0))
                ess = float(1 / (w * w).sum())
                if ess > target * len(w):
                    break
                comps = self.mc.comps
                resp = torch.softmax(self.mc.lcomp[1:], 0)          # responsibilities, prior component excluded
                z, lr = self.mc.z, self.mc.lr
                new = [comps[0]]
                for k, c in enumerate(comps[1:]):
                    wk = w * resp[k]
                    if float(wk.sum()) < 1e-8:
                        new.append(c)
                        continue
                    wk = wk / wk.sum()
                    mu = _tiny_gram(wk[:, None], z)[0]
                    dz = z - mu
                    Cov = _tiny_gram(dz * wk[:, None], dz) + 1e-6 * I
                    lm = float(wk @ lr)
                    ls = math.sqrt(max(float(wk @ (lr - lm) ** 2), 1e-6))
                    try:
                        Cc = torch.linalg.cholesky(Cov)
                    except torch.linalg.LinAlgError:
                        new.append(c)
                        continue
                    new.append(dict(mu=mu, C=inflate * Cc, lmu=lm, lsd=inflate * ls))
                self.mc.set_proposal(new, M)
                self.t, _ = self.solve_t(L, qbar, M, self.t)
            self.log(f"    proposal refit {r}: ESS before {ess:.0f}/{len(w)}")
        self.far_comps = [c for c in self.mc.comps[2:]] if len(self.mc.comps) > 2 else []

    def set_samples(self, n_log2, M, fresh=False):
        """Draw a new fixed sample set (2^n_log2 per component) and its proposal at the current state."""
        J, T = self.last["J"], self.last["T"]
        self.mc = HiddenMC(self.p, self.dev, n_log2=n_log2, seed=self.mc.seed)
        self.mc_ready = False
        if fresh:
            self.far_comps = []
        self.build_proposal(J, T, M, self.t)
        with torch.no_grad():
            self.t, _ = self.solve_t(scaled_cholesky(self.p.sigB ** 2 * J), self.p.sigB ** 2 * T, M, self.t)
        self.build_proposal(J, T, M, self.t)
        self.refit(M)
        self.nprop += 1

    def solve(self, theta0=None, M0=None, small=11, big=15, tol_small=1e-2, tol=1e-9, mtol=1e-3):
        t0 = time.time()
        p = self.p
        self.prune_nats = type(self).prune_nats
        self.mask, self.nsteps, self.nguard, self.nprop = None, 0, 0, 0
        self.theta_ref = torch.zeros(self.nth, dtype=DT, device=self.dev) if theta0 is None else theta0.clone()
        self.rebase()
        self.last = {}
        M = self._M_now()
        if M0 is not None:
            M = M0
        J, T = self.last["J"], self.last["T"]
        self.t, _, _ = solve_t_convex(self.mid, p, scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * T, M,
                                      self.y / p.lam)
        # stage 1: few samples, full read-in grid (steps are large), M fixed during each Newton run
        if small < big:
            self.prune_nats = math.inf
            self.mask = None
            self.set_samples(small, M)
            for rep in range(30):
                self.rebase()
                W, g, ev, M1, st = self.newton_M(M, tol_small, ess_min=0.05)
                if st == "ess":
                    self._M_now()
                    self.set_samples(small, M)
                    continue
                Mn = self._M_now()
                done = abs(Mn / M - 1) < mtol
                M = Mn
                if done:
                    break
            self.log(f"  stage 1 (2^{small}): W={W:.4f} M={M:.4f} ({self.nsteps} Newton steps, "
                     f"{time.time() - t0:.2f}s)")
        # stage 2: full sample set, pruned read-in grid, fresh proposal built once at the stage-1 state, M updated
        # after every step
        self.prune_nats = type(self).prune_nats
        self.mask = None
        self.rebase()
        self._M_now()
        self.set_samples(big, M, fresh=True)
        for rep in range(5):
            self.rebase()
            W, g, ev, M, st = self.newton_M(M, tol, fold=True, ess_min=0.02)
            if st != "ess":
                break
            self._M_now()
            self.set_samples(big, M)
        L_ = self.last
        fb = (self.y - p.lam * L_["t"]).cpu().numpy()
        self.log(f"  stage 2 (2^{big}): W={W:.6f} mu1={fb[0]:.6f} mu3={fb[1] / p.eps:.6f} M={M:.4f} "
                 f"ESS={L_['ess']:.0f}/{len(self.mc.z)} ({self.nsteps} Newton steps, {time.time() - t0:.2f}s)")
        pw = L_["pw"].reshape(self.ri.W.shape)
        return dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, ess=L_["ess"], nsamples=len(self.mc.z),
                    J=L_["J"].cpu().numpy().tolist(), T=L_["T"], t=L_["t"].cpu().numpy().tolist(),
                    H=L_["H"].cpu().numpy().tolist(), Eq=L_["Eq"], theta=self.theta_ref.cpu().numpy().tolist(),
                    hess_eig=ev.cpu().numpy().tolist(), readin_mass_w0_gt_1=float(pw[self.ri.W > 1.0].sum()),
                    d_w0sq=float(p.d * (pw * self.ri.W ** 2).sum()), newton_steps=self.nsteps,
                    prune_guards=self.nguard, proposals=self.nprop, converged=(st == "converged"),
                    walltime_s=time.time() - t0)


# ======================================================================================================================
# Default entry point
# ======================================================================================================================
def solve_fcn3(p, nmax=11, device=None, verbose=False, seed=0):
    """Equilibrium of (E1)-(E8) at parameters p (fcn3_matrix_solver.Params): the global maximum of the free energy W,
    over the full distributions of both layers, with BlazingFreeEnergy (the fast default solver).
    Solves at n_max = 3 from the prior, then at `nmax` starting from that read-in distribution (its exponents for
    r^2, r^4, r^6 and P_perp carry over; the higher ones start at 0). A few seconds per solve on a GPU.
    Returns the n_max = `nmax` result dict, with the n_max = 3 result under "nmax3"."""
    from dataclasses import replace
    if not math.isfinite(p.N1):
        # N1 = inf: the read-in distribution is the prior (no read-in degrees of freedom); only the readout t is
        # solved, on the fixed hidden-field samples (FixedFreeEnergy handles this case directly)
        F = FixedFreeEnergy(replace(p, nmax=nmax), device=device, verbose=verbose, seed=seed)
        o = F.solve_fixed()
        o["nmax"], o["converged"] = nmax, True
        return o
    p3 = replace(p, nmax=3)
    F3 = BlazingFreeEnergy(p3, device=device, verbose=verbose, seed=seed)
    o3 = F3.solve()
    if nmax == 3:
        return o3
    pn = replace(p, nmax=nmax)
    F = BlazingFreeEnergy(pn, device=device, verbose=verbose, seed=seed)
    th = torch.zeros(F.nth, dtype=DT, device=F.dev)
    th[:3] = F3.theta_ref[:3].to(F.dev)
    th[-1] = F3.theta_ref[-1].to(F.dev)
    o = F.solve(theta0=th, M0=o3["M"])
    o["nmax"] = nmax
    o["nmax3"] = {k: v for k, v in o3.items() if k not in ("t", "J", "H", "theta")}
    return o
