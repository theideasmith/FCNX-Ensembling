"""
Free energy of the FCN3 mean-field theory (E1)-(E8) and a solver that maximises it.

    W(J, T) = -N1 KL*(J, T)
              + min_t [ N2 log Z_h(G = sigma_B^2 J, qbar = sigma_B^2 T, M; t) + chi (kappa/(2P) |t|^2 - y.t) ]

  J = E_w[c c^T] (s x s), T = E_w[P_perp(w)] (so qbar_h = sigma_B^2 T, eq. E7).
  KL*(J, T) = min over read-in distributions q with those moments of KL(q || prior), by convex duality
            = -min_{Lam, lam} [ log E_prior exp(-1/2 (c^T Lam c + lam P_perp)) + 1/2 tr(Lam J) + 1/2 lam T ].
  log Z_h   = log E_prior[exp((g~/2)(t.e)^2)] over the hidden field (prior N(0, G) x Gamma(M/2, qbar)).

Stationarity:  dW/dJ = (N1/2)(Lam - r_B G^-1 (G - H) G^-1),  dW/dT = (N1/2)(lam - r_B M (qbar - <q_h>)/qbar^2),
which is (E5); the moment constraints are (E6)-(E7); the t minimum is (E4). M is held fixed in W and updated
from (E8) in an outer loop (the theory drops the variation of M).

The equilibrium is the global maximum of W (Gibbs variational principle). Each local maximum is a branch.
"""
import math
import time

import numpy as np
import torch
from scipy.optimize import minimize

from fcn3_matrix_solver import DT, Groups, Middle, Numerics, Params, ReadIn, readin_features, solve_t_convex


class FreeEnergy:
    def __init__(self, p: Params, nu: Numerics = None, device=None, verbose=True):
        self.p, self.nu = p, nu or Numerics()
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ri = ReadIn(p, self.nu, self.dev)
        self.mid = Middle(p, self.nu, self.dev)
        self.verbose = verbose
        S = self.S = p.nS
        self.iu = torch.triu_indices(S, S, device=self.dev)
        self.I = torch.eye(S, dtype=DT, device=self.dev)
        self.y = torch.tensor(p.y(), dtype=DT, device=self.dev)
        ri = self.ri
        self.lp0 = -ri.base + ri.logm
        self.lp0 = self.lp0 - torch.logsumexp(self.lp0.reshape(-1), 0)       # normalised read-in prior on the grid
        self.Cf = ri.C.reshape(S, -1)                                         # (S, npts)
        self.Pp = ri.Pperp.reshape(-1)
        w0 = torch.exp(self.lp0.reshape(-1))
        self.J0 = (self.Cf * w0) @ self.Cf.T
        self.T0 = float(w0 @ self.Pp)
        self.theta = None
        self.t = None
        self.nev = 0
        self.recount = True

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    # ------------------------------------------------------------------ read-in dual (convex)
    def _stats(self, scale):
        """Sufficient statistics, normalised: c_a c_b m_ab / sqrt(scale_a scale_b) (a <= b), and P_perp/T0."""
        a, b = self.iu
        mult = torch.where(a == b, 1.0, 2.0).to(DT)
        C = self.Cf / torch.sqrt(scale)[:, None]
        Tst = torch.cat([(C[a] * C[b]) * mult[:, None], (self.Pp / self.T0)[None]], 0)    # (k, npts)
        return Tst

    def kl_star(self, J, T, tol=1e-10, maxit=300):
        out = self._kl_star(J, T, tol, maxit)
        if out[4] > 1e-9 and self.theta is not None:
            self.theta = None                                  # warm start was bad: restart from theta = 0
            out = self._kl_star(J, T, tol, maxit)
        if out[4] > 1e-8:
            self.theta = None
        return out

    def _kl_star(self, J, T, tol=1e-10, maxit=300):
        """min_theta D(theta) = log E0 exp(-theta.Tst/2) + theta.Tbar/2 (convex). Returns -KL* = D_min,
        Lam (s x s), lam, and the read-in weights."""
        scale = torch.diagonal(J).clone()
        Tst = self._stats(scale)
        a, b = self.iu
        mult = torch.where(a == b, 1.0, 2.0).to(DT)
        Tbar = torch.cat([(J[a, b] / torch.sqrt(scale[a] * scale[b])) * mult, torch.tensor([T / self.T0], dtype=DT,
                                                                                           device=self.dev)])
        lp0 = self.lp0.reshape(-1)
        th = torch.zeros(len(Tbar), dtype=DT, device=self.dev) if self.theta is None else self._theta_rescale(scale)

        def D(th):
            lw = lp0 - 0.5 * (th @ Tst)
            lz = torch.logsumexp(lw, 0)
            return float(lz + 0.5 * th @ Tbar), lw - lz

        f, lw = D(th)
        for it in range(maxit):
            pw = torch.exp(lw)
            ET = Tst @ pw
            g = 0.5 * (Tbar - ET)
            if float(g.abs().max()) < tol:
                break
            Dv = Tst - ET[:, None]
            Hm = 0.25 * (Dv * pw) @ Dv.T
            step = -torch.linalg.lstsq(Hm + 1e-14 * torch.eye(len(g), dtype=DT, device=self.dev), g[:, None],
                                       rcond=1e-13).solution[:, 0]
            al, slope = 1.0, float(g @ step)
            while True:
                f1, lw1 = D(th + al * step)
                if f1 <= f + 1e-4 * al * slope or al < 1e-12:
                    break
                al *= 0.5
            th, f, lw = th + al * step, f1, lw1
        self.theta, self.theta_scale = th.clone(), scale.clone()
        # back to un-normalised multipliers: exponent -1/2 (c^T Lam c + lam P_perp)
        Lam = torch.zeros(self.S, self.S, dtype=DT, device=self.dev)
        Lam[a, b] = th[:-1] / torch.sqrt(scale[a] * scale[b])
        Lam = Lam + Lam.T - torch.diag(torch.diagonal(Lam))
        lam = float(th[-1] / self.T0)
        return f, Lam, lam, torch.exp(lw), float(g.abs().max())

    def _theta_rescale(self, scale):
        a, b = self.iu
        th = self.theta.clone()
        th[:-1] = th[:-1] * torch.sqrt(scale[a] * scale[b] / (self.theta_scale[a] * self.theta_scale[b]))
        return th

    # ------------------------------------------------------------------ hidden field (convex in t)
    def hidden(self, J, T, M):
        p = self.p
        G = p.sigB ** 2 * J
        qbar = p.sigB ** 2 * T
        d = torch.sqrt(torch.diagonal(G))
        L = d[:, None] * torch.linalg.cholesky(G / torch.outer(d, d))
        t0 = self.t if self.t is not None else self.y / p.lam
        for _ in range(2):
            t, st, its = solve_t_convex(self.mid, p, L, qbar, M, t0, recount=self.recount)
            t0 = t
        self.recount = False                 # transverse rule frozen until the next outer (M) update
        self.t = t
        # normalised log Z_h: same cells with and without the second term of (11)
        mid = self.mid
        c1 = mid.cells(t)
        c0 = mid.cells(torch.zeros_like(t))
        logZh = float(torch.logsumexp(c1["L"].reshape(-1), 0) - torch.logsumexp(c0["L"].reshape(-1), 0))
        phi_t = p.N2 * logZh + p.chi * (0.5 * p.lam * float(t @ t) - float(self.y @ t))
        return dict(t=t, st=st, logZh=logZh, phi_t=phi_t, qbar=qbar, G=G, L=L)

    # ------------------------------------------------------------------ W and its gradient
    def W(self, J, T, M):
        p = self.p
        Dmin, Lam, lam, pw, gres = self.kl_star(J, T)
        h = self.hidden(J, T, M)
        st = h["st"]
        Vh = st["Vh"]                                            # G^-1 (G - H) G^-1 (Stein)
        qbar = h["qbar"]
        W = p.N1 * Dmin + h["phi_t"] if math.isfinite(p.N1) else h["phi_t"]
        gJ = 0.5 * p.N1 * (Lam - p.rB * Vh)
        gT = 0.5 * p.N1 * (lam - p.rB * M * (qbar - float(st["Eq"])) / qbar ** 2)
        self.nev += 1
        self.last = dict(J=J, T=T, M=M, Lam=Lam, lam=lam, pw=pw, h=h, Dmin=Dmin, dual_res=gres)
        return W, gJ, gT

    # ------------------------------------------------------------------ parametrisation
    def encode(self, J, T):
        dg = torch.sqrt(torch.diagonal(J))
        R = J / torch.outer(dg, dg)
        il = torch.triu_indices(self.S, self.S, offset=1, device=self.dev)
        return torch.cat([torch.log(torch.diagonal(J)), torch.atanh(R[il[0], il[1]].clamp(-1 + 1e-15, 1 - 1e-15)),
                          torch.tensor([math.log(T)], dtype=DT, device=self.dev)])

    def decode(self, x):
        S = self.S
        il = torch.triu_indices(S, S, offset=1, device=self.dev)
        dg = torch.exp(0.5 * x[:S])
        R = torch.eye(S, dtype=DT, device=self.dev)
        R = R.index_put((il[0], il[1]), torch.tanh(x[S:-1]))
        R = R.index_put((il[1], il[0]), torch.tanh(x[S:-1]))
        return R * torch.outer(dg, dg), torch.exp(x[-1])

    def _value_grad(self, x, M):
        """W and dW/dx at x, or (None, None) if the moments are infeasible."""
        x = x.clone().requires_grad_(True)
        J, T = self.decode(x)
        try:
            torch.linalg.cholesky(J.detach())
            Wv, gJ, gT = self.W(J.detach(), float(T.detach()), M)
        except (torch.linalg.LinAlgError, RuntimeError):
            return None, None
        if not (self.last["dual_res"] < 1e-6 and math.isfinite(Wv)):
            return None, None
        ((J * gJ).sum() + T * gT).backward()
        return Wv, x.grad.detach()

    def maximise(self, x0, M, gtol=1e-6, maxiter=400, radius=0.3):
        """Local maximum of W at fixed M: trust-region quasi-Newton ascent in x. A step is taken only if the new
        point is feasible (the read-in dual converges) and W increases; otherwise the radius shrinks."""
        x = x0.clone()
        Wv, g = self._value_grad(x, M)
        if Wv is None:
            raise RuntimeError("starting point is infeasible")
        n = len(x)
        B = torch.eye(n, dtype=DT, device=self.dev) * float(g.abs().max()) / radius      # approx. -Hessian
        it = 0
        for it in range(maxiter):
            if float(g.abs().max()) < gtol * max(1.0, abs(Wv)) ** 0 and float(g.abs().max()) < gtol:
                break
            step = torch.linalg.solve(B, g)
            mx = float(step.abs().max())
            if mx > radius:
                step = step * (radius / mx)
            W1, g1 = self._value_grad(x + step, M)
            if W1 is None or W1 <= Wv:
                radius *= 0.3
                if radius < 1e-10:
                    break
                continue
            pred = float(g @ step - 0.5 * step @ B @ step)
            rho = (W1 - Wv) / max(pred, 1e-300)
            if rho > 0.75 and float(step.abs().max()) > 0.99 * radius:
                radius *= 2.0
            elif rho < 0.25:
                radius *= 0.5
            # BFGS update of B (approximates -Hessian): yk = -(g1 - g)
            yk, sk = -(g1 - g), step
            sy = float(sk @ yk)
            if sy > 1e-12 * float(sk @ sk) * float(yk.norm()):
                Bs = B @ sk
                B = B - torch.outer(Bs, Bs) / float(sk @ Bs) + torch.outer(yk, yk) / sy
            x, Wv, g = x + step, W1, g1
        class R:
            pass
        r = R()
        r.nit, r.jac = it, -g.cpu().numpy()
        return x, Wv, r

    def _fd_col(self, x, g, j, M):
        """d grad / d x_j: central difference, falling back to one-sided or a smaller step near infeasibility."""
        for h in (1e-5, 1e-6, 1e-7):
            e = torch.zeros_like(x)
            e[j] = h
            _, gp = self._value_grad(x + e, M)
            _, gm = self._value_grad(x - e, M)
            if gp is not None and gm is not None:
                return (gp - gm) / (2 * h)
            if gp is not None:
                return (gp - g) / h
            if gm is not None:
                return (g - gm) / h
        raise RuntimeError("cannot difference the gradient here")

    def stationary(self, x0, M, tol=1e-7, maxit=40, radius=1.0):
        """Stationary point of W at fixed M: Newton on grad W = 0 in the smooth J coordinates (this is
        (E5)-(E7)). The gradient is accurate (exact dual multipliers, Stein form of G^-1(G-H)G^-1); W itself
        carries quadrature noise of order N2 * 1e-4 and is used only to compare branches. The Hessian
        (finite differences of the gradient) is returned to classify the point (maximum vs saddle)."""
        x = x0.clone()
        W, g = self._value_grad(x, M)
        if W is None:
            raise RuntimeError("starting point is infeasible")
        n = len(x)
        Hm = None
        for it in range(maxit):
            gn = float(g.abs().max())
            if gn < tol * max(1.0, float(self.p.N1)):
                break
            if Hm is None:
                Hm = torch.zeros(n, n, dtype=DT, device=self.dev)
                for j in range(n):
                    Hm[:, j] = self._fd_col(x, g, j, M)
                Hm = 0.5 * (Hm + Hm.T)
            step = -torch.linalg.solve(Hm, g)
            mx = float(step.abs().max())
            if mx > radius:
                step = step * (radius / mx)
            a = 1.0
            while a > 1e-4:
                W1, g1 = self._value_grad(x + a * step, M)
                if W1 is not None and float(g1.abs().max()) < gn:
                    break
                a *= 0.5
            else:
                Hm = None                          # refresh the Hessian and retry from the same point
                radius *= 0.5
                if radius < 1e-6:
                    break
                continue
            sk, yk = a * step, g1 - g
            Hm = Hm + torch.outer(yk - Hm @ sk, sk) / float(sk @ sk)          # Broyden update
            Hm = 0.5 * (Hm + Hm.T)
            x, W, g = x + sk, W1, g1
        # Hessian at the solution for classification
        H = torch.zeros(n, n, dtype=DT, device=self.dev)
        for j in range(n):
            H[:, j] = self._fd_col(x, g, j, M)
        H = 0.5 * (H + H.T)
        self._value_grad(x, M)
        return x, W, g, torch.linalg.eigvalsh(H).cpu().numpy()

    def solve_stationary(self, x0=None, M0=None, outer=8, mtol=1e-6):
        t0 = time.time()
        if x0 is None:
            x0 = self.encode(self.J0, self.T0)
        M = M0
        if M is None:
            self.kl_star(self.J0, self.T0)
            self.last = dict(pw=torch.exp(self.lp0.reshape(-1)), J=self.J0, T=self.T0)
            M = self.M_from_readin()
        x = x0
        for k in range(outer):
            self.recount = True
            x, W, g, ev = self.stationary(x, M)
            Mn = self.M_from_readin()
            self.log(f"  outer {k}: W={W:.4f}  |grad|={float(g.abs().max()):.1e}  Hessian eig {np.round(ev, 1).tolist()}"
                     f"  M {M:.4f} -> {Mn:.4f}  ({self.nev} evals, {time.time() - t0:.1f}s)")
            M_old, M = M, Mn
            if abs(Mn / M_old - 1) < mtol:
                break
        r = self.results(x, W, M, time.time() - t0)
        r["grad_max"] = float(g.abs().max())
        r["hessian_eigs"] = ev.tolist()
        r["kind"] = "maximum" if (ev < 0).all() else "saddle"
        return r

    def M_from_readin(self):
        """(E8) from the current read-in distribution."""
        ri, p = self.ri, self.p
        pw = self.last["pw"].reshape(ri.W.shape)
        EJ2, J_at = ri.pair_trace(pw)
        J, T = self.last["J"], self.last["T"]
        qbar = p.sigB ** 2 * T
        return qbar ** 2 / (p.sigB ** 4 * float(EJ2 - (J_at ** 2).sum()))

    def solve(self, x0=None, M0=None, outer=8, mtol=1e-7):
        """Local maximum from x0 (default: the prior), with M updated by (E8) until it stops changing."""
        t0 = time.time()
        if x0 is None:
            x0 = self.encode(self.J0, self.T0)
        M = M0
        if M is None:
            # prior M
            self.kl_star(self.J0, self.T0)
            self.last = dict(pw=torch.exp(self.lp0.reshape(-1)), J=self.J0, T=self.T0)
            M = self.M_from_readin()
        x = x0
        for k in range(outer):
            self.recount = True
            x, W, res = self.maximise(x, M)
            J, T = self.decode(x)
            self.W(J, float(T), M)
            Mn = self.M_from_readin()
            self.log(f"  outer {k}: W={W:.6f}  |grad|={np.abs(res.jac).max():.1e}  M {M:.4f} -> {Mn:.4f}  "
                     f"({res.nit} its, {self.nev} evals, {time.time() - t0:.1f}s)")
            if abs(Mn / M - 1) < mtol:
                M = Mn
                break
            M = Mn
        return self.results(x, W, M, time.time() - t0)

    def results(self, x, W, M, wall):
        p, L = self.p, self.last
        h = L["h"]
        t = h["t"]
        fb = (self.y - p.lam * t).cpu().numpy()
        ri = self.ri
        pw = L["pw"].reshape(ri.W.shape)
        far = float(pw[ri.W > 1.0].sum())
        Hm = h["L"] @ h["st"]["Ezz"] @ h["L"].T
        return dict(W=W, L1=float(fb[0]), L3=float(fb[1] / p.eps), M=M, qbar=float(h["qbar"]),
                    readin_mass_w0_gt_1=far, d_w0sq=float(p.d * (pw * ri.W ** 2).sum()), H=Hm.cpu().numpy().tolist(),
                    J=L["J"].cpu().numpy().tolist(), T=float(L["T"]), t=t.cpu().numpy().tolist(), x=x.cpu().numpy(),
                    dual_res=L["dual_res"], walltime_s=wall)
