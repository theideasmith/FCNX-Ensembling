"""
Fast solver for (E1)-(E8) of notes/fcn3_matrix_eos.tex, over the full distributions (same equations and grids as
fcn3_matrix_solver.FCN3Solver, which serves as the reference).

All unknowns are solved together by one Newton iteration:
    x    log J_nn, atanh of the correlations of J, log qbar_h, log M          (E6)-(E8)
    tau  kappa/P * t  (= y - fbar)                                            (E4)
    ell  logits of the far-basin masses                                       (see fcn3_matrix_solver)
with residual
    R_x   = encode(state from the read-in distribution) - x
    R_tau = sigma_A^2 K t + kappa/P t - y
    R_ell = log P(B_g) - log P(B_0) - ell_g      (full-distribution masses of the frozen basins)
The Jacobian is computed by finite differences once per continuation step and then updated by Broyden's method.

Within a step the quadrature is frozen (basins, the hidden-field (s, q_h) cells that hold mass, the transverse
zeta rule and frame), so R is smooth. Only cells holding mass are integrated; before and after each step one pass
over the full grids rebuilds the frozen quadrature and checks that the cells left out hold less than
e^-30 of the mass, so nothing is cut off.

Continuation in chi uses a secant predictor in log chi and an adaptive step.
"""
import math
import time
from dataclasses import asdict

import numpy as np
import torch

from fcn3_matrix_solver import DT, Groups, Middle, Numerics, NoConvergence, Params, ReadIn, make_groups


class FastSolver:
    def __init__(self, p: Params, nu: Numerics = None, device=None, verbose=True, newton_tol=1e-9,
                 cell_floor=40.0, excluded_max=-30.0, include_floor=-120.0):
        self.p, self.nu = p, nu or Numerics()
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.readin = ReadIn(p, self.nu, self.dev)
        self.mid = Middle(p, self.nu, self.dev)
        self.verbose = verbose
        self.newton_tol, self.cell_floor, self.excluded_max = newton_tol, cell_floor, excluded_max
        self.include_floor = include_floor   # far basins down to this log mass ratio get a logit
        self.max_step = math.log(math.sqrt(2.0))
        self.S = p.nS
        self.I = torch.eye(self.S, dtype=DT, device=self.dev)
        self.y = torch.tensor(p.y(), dtype=DT, device=self.dev)
        self.il = torch.triu_indices(self.S, self.S, offset=1, device=self.dev)
        ri0 = self.readin.solve(self.readin.base)
        self.state0 = self._state_from(ri0)
        self.nx = self.S + len(self.il[0]) + 2
        self.npass = 0
        self.frame = None

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    # -------------------------------------------------------------------------------------------- state maps
    def _state_from(self, ri):
        p = self.p
        qbar = float(p.sigB ** 2 * (ri["EJxx"] - torch.trace(ri["J"])))
        trG2 = float(p.sigB ** 4 * (ri["EJ2"] - (ri["J_atoms"] ** 2).sum()))
        return dict(J=ri["J"], qbar=qbar, M=qbar ** 2 / trG2)

    def encode(self, s):
        J = s["J"]
        dg = torch.sqrt(torch.diag(J))
        R = J / torch.outer(dg, dg)
        r = torch.atanh(R[self.il[0], self.il[1]].clamp(-1 + 1e-15, 1 - 1e-15))
        return torch.cat([torch.log(torch.diag(J)), r,
                          torch.tensor([math.log(s["qbar"]), math.log(s["M"])], dtype=DT, device=self.dev)])

    def decode(self, x):
        S = self.S
        dg = torch.exp(0.5 * x[:S])
        R = self.I.clone()
        R[self.il[0], self.il[1]] = torch.tanh(x[S:-2])
        R = R + R.T - self.I
        return dict(J=R * torch.outer(dg, dg), qbar=math.exp(float(x[-2])), M=math.exp(float(x[-1])))

    # -------------------------------------------------------------------------------------------- hidden field
    def _mid_setup(self, L, t, qbar, M, full):
        """Hidden-field geometry. full=True picks the transverse rule and frame (SVD); otherwise the frozen frame
        is re-orthogonalised against the current drive direction u (smooth in the unknowns)."""
        mid, p = self.mid, self.p
        if full:
            mid.setup(L, t, qbar, M, recount=True)
            self.frame_Qp = mid.Qp.clone()
            return
        v = L.T @ t
        u = v / v.norm()
        Qp = self.frame_Qp - u[:, None] * (u @ self.frame_Qp)[None, :]
        Q, Rr = torch.linalg.qr(Qp)
        Qp = Q * torch.sign(torch.diagonal(Rr))[None, :]
        mid.u, mid.Qp, mid.L = u, Qp, L
        mid.a = (L @ u) @ mid.Phi
        mid.A = mid.zeta @ ((L @ Qp).T @ mid.Phi)
        k = M / 2
        nu = self.nu
        lqh = math.log(qbar) + torch.linspace(-nu.lqh_below, nu.lqh_above, nu.nqh, dtype=DT, device=self.dev)
        mid.qh = torch.exp(lqh)
        lw = torch.log(_trapz(lqh)) + k * (lqh - math.log(qbar)) - k * mid.qh / qbar
        mid.logwq = lw - torch.logsumexp(lw, 0)
        mid.rq = 1 / torch.sqrt(1 + 2 * mid.qh)
        mid.qbar, mid.M = qbar, M

    def _cells_list(self, t, si, qi):
        """Per active cell: log mass and conditional means over zeta. Reductions over zeta are written as
        elementwise products and sums: as batched matmuls with 6x6 outputs and ~1e4-long reductions cuBLAS is
        ~100x slower. The Stein Hessian term is reduced over zeta per x0 node before contracting with phi phi^T."""
        mid, p = self.mid, self.p
        S, nz = self.S, len(mid.zeta)
        s = mid.s[si]
        rq = mid.rq[qi]
        nc = len(si)
        per = max(1, self.nu.chunk // (nz * self.nu.nx))
        out = {k: [] for k in ("L", "BB", "tB2", "t2BB", "e1", "zz2", "Vt")}
        tphi = (t @ mid.Phi) * mid.wx                                            # (nx,)
        PP = (mid.Phi[:, None, :] * mid.Phi[None, :, :]).reshape(S * S, -1).T   # (nx, S*S)
        PhiT = mid.Phi.T.contiguous()
        for i in range(0, nc, per):
            sl = slice(i, i + per)
            arg = (s[sl, None, None] * mid.a + mid.A[None]) * rq[sl, None, None]  # (n, nz, nx)
            ep = (2 / math.sqrt(math.pi)) * torch.exp(-arg * arg)               # erf'(a)
            G1 = ((ep * tphi) @ PhiT) * rq[sl, None, None]                       # grad_h (t.e), (n, nz, S)
            m2 = -2 * arg * ep                                                   # erf''(a)
            arg.erf_()
            B = arg @ mid.WPhi                                                   # e_n, (n, nz, S)
            tB = B @ t
            lw = mid.logwz + 0.5 * p.gt * tB * tB
            m = lw.amax(1, keepdim=True)
            e = torch.exp(lw - m)
            Z = e.sum(1)
            f = e / Z[:, None]
            out["L"].append(m[:, 0] + torch.log(Z) + mid.logwq[qi[sl]] - 0.5 * s[sl] ** 2 + mid.logws[si[sl]])
            fB = f[..., None] * B
            out["BB"].append((fB[..., :, None] * B[..., None, :]).sum(1))
            out["tB2"].append((fB * tB[..., None]).sum(1))
            out["t2BB"].append(((fB * (tB * tB)[..., None])[..., :, None] * B[..., None, :]).sum(1))
            out["e1"].append((f[..., None] * mid.zeta).sum(1))
            out["zz2"].append((f[..., None] * mid.zz).sum(1).reshape(-1, S - 1, S - 1))
            # Stein term: E_zeta[(1 + g~ (t.e)^2) grad grad^T + (t.e) Hess], Hess = rq^2 sum_x tphi_x erf''_x phi phi^T
            wg = f * (1 + p.gt * tB * tB)
            gg = ((wg[..., None] * G1)[..., :, None] * G1[..., None, :]).sum(1)
            hx = ((f * tB)[..., None] * m2).sum(1) * tphi                       # (n, nx)
            Hs = (hx @ PP).reshape(-1, S, S) * (rq[sl] ** 2)[:, None, None]
            out["Vt"].append(gg + Hs)
        return {k: torch.cat(v) for k, v in out.items()}

    def _mid_stats(self, c, si, qi, gi, ell_h):
        """Moments over the active cells (gi: group of each cell, ell_h: logits of the far groups)."""
        mid = self.mid
        L = c["L"]
        m = int(gi.max()) if len(gi) else 0
        lse = [torch.logsumexp(torch.where(gi == k, L, -torch.inf), 0) for k in range(m + 1)]
        delta = [lse[k] - lse[0] for k in range(1, m + 1)]
        if m:
            logits = torch.cat([torch.zeros(1, dtype=DT, device=self.dev), ell_h])
            logpi = logits - torch.logsumexp(logits, 0)
            Lr = L.clone()
            for k in range(m + 1):
                Lr = torch.where(gi == k, L - lse[k] + logpi[k], Lr)
        else:
            Lr = L
        pi = torch.exp(Lr - torch.logsumexp(Lr, 0))
        K = torch.einsum("c,cij->ij", pi, c["BB"])
        u, Qp, s = mid.u, mid.Qp, mid.s[si]
        cross = Qp @ torch.einsum("c,ci->i", pi * s, c["e1"])
        Ezz = (pi @ (s * s)) * torch.outer(u, u) + torch.outer(u, cross) + torch.outer(cross, u) \
            + Qp @ torch.einsum("c,cij->ij", pi, c["zz2"]) @ Qp.T
        Eq = pi @ mid.qh[qi]
        lp0 = mid.logws - 0.5 * mid.s ** 2
        p0 = torch.exp(lp0 - torch.logsumexp(lp0, 0))
        Ezz0 = (p0 @ (mid.s ** 2)) * torch.outer(u, u) + Qp @ mid.Zz2 @ Qp.T
        Eq0 = torch.exp(mid.logwq) @ mid.qh
        Vh = -self.p.gt * torch.einsum("c,cij->ij", pi, c["Vt"])
        return dict(K=K, Ezz=Ezz - Ezz0 + self.I, Eq=Eq - Eq0 + mid.qbar, Vh=Vh, delta=delta, pi=pi, Lr=Lr)

    # -------------------------------------------------------------------------------------------- read-in, pruned
    def _readin_frame(self, A):
        """Freeze the read-in basins and the grid points that hold mass (within cell_floor of each basin's
        largest point), plus their atom blocks for E k(w,w')^2."""
        ri = self.readin
        rg = make_groups(A, ri.seed, self.include_floor)
        ri.groups = rg
        Lp = -A + ri.logm
        gfull = rg.group_index() if rg.m else torch.zeros_like(rg.labels if rg.labels is not None else Lp,
                                                                   dtype=torch.long)
        keep = torch.zeros_like(Lp, dtype=torch.bool)
        for k in range(rg.m + 1):
            mk = gfull == k
            mx = torch.where(mk, Lp, -torch.inf).max()
            keep |= mk & (Lp > mx - self.cell_floor)
        idx = torch.nonzero(keep.reshape(-1), as_tuple=True)[0]
        n0, n1 = Lp.shape
        bw, bq = self.nu.pair_block
        blk = (idx // n1) // bw * ((n1 + bq - 1) // bq) + (idx % n1) // bq
        ub, binv = torch.unique(blk, return_inverse=True)
        f = lambda X: X.reshape(-1)[idx]
        self.rframe = dict(idx=idx, C=ri.C.reshape(self.S, -1)[:, idx], base=f(ri.base), knorm=f(ri.knorm),
                           Pperp=f(ri.Pperp), W=f(ri.W), Q=f(ri.Q), logm=f(ri.logm), g=f(gfull),
                           binv=binv, nb=len(ub), m=rg.m, keep=keep)
        return rg, Lp

    def _readin_solve(self, Linv, V, mcoef, ell_r):
        rf, p = self.rframe, self.p
        ct = Linv @ rf["C"]
        A = rf["base"] + 0.5 * p.rB * ((ct * (V @ ct)).sum(0)) + 0.5 * p.rB * mcoef * rf["Pperp"]
        Lp = -A + rf["logm"]
        m = rf["m"]
        if m:
            g = rf["g"]
            lse = [torch.logsumexp(torch.where(g == k, Lp, -torch.inf), 0) for k in range(m + 1)]
            delta = [lse[k] - lse[0] for k in range(1, m + 1)]
            logits = torch.cat([torch.zeros(1, dtype=DT, device=self.dev), ell_r])
            logpi = logits - torch.logsumexp(logits, 0)
            Lr = Lp.clone()
            for k in range(m + 1):
                Lr = torch.where(g == k, Lp - lse[k] + logpi[k], Lr)
        else:
            delta, Lr = [], Lp
        pw = torch.exp(Lr - torch.logsumexp(Lr, 0))
        C = rf["C"]
        J = (C[:, None, :] * C[None, :, :] * pw).sum(-1)       # not a matmul: cuBLAS is ~100x slower here
        EJxx = pw @ rf["knorm"]
        # atoms for E k^2: block sums of the kept points
        nb = rf["nb"]
        mb = torch.zeros(nb, dtype=DT, device=self.dev).index_add_(0, rf["binv"], pw)
        wb = torch.zeros(nb, dtype=DT, device=self.dev).index_add_(0, rf["binv"], pw * rf["W"]) / mb.clamp_min(1e-300)
        qb = torch.zeros(nb, dtype=DT, device=self.dev).index_add_(0, rf["binv"], pw * rf["Q"]) / mb.clamp_min(1e-300)
        if rf.get("akeep") is None:                         # atoms with mass, chosen once per frame
            rf["akeep"] = torch.nonzero(mb > 1e-14 * mb.max(), as_tuple=True)[0]
        ak = rf["akeep"]
        mk = mb[ak]
        EJ2, J_at = _pair_trace(mk / mk.sum(), wb[ak], qb[ak].clamp_min(1e-300), self.p.ks, self.readin.u,
                                self.readin.wu)
        return dict(J=J, EJxx=EJxx, EJ2=EJ2, J_atoms=J_at, delta=delta, pw=pw,
                    w0sq=pw @ (rf["W"] ** 2), qmean=pw @ rf["Q"])

    # -------------------------------------------------------------------------------------------- residual
    def residual(self, u):
        """R(u) at the frozen quadrature. Returns None if G is not positive definite."""
        p, S, nx = self.p, self.S, self.nx
        fr = self.frame
        x, tau, ell = u[:nx], u[nx:nx + S], u[nx + S:]
        s = self.decode(x)
        try:
            L, Linv = scaled_cholesky(p.sigB ** 2 * s["J"])
        except torch.linalg.LinAlgError:
            return None
        t = tau / p.lam
        self._mid_setup(L, t, s["qbar"], s["M"], full=False)
        c = self._cells_list(t, fr["si"], fr["qi"])
        st = self._mid_stats(c, fr["si"], fr["qi"], fr["gi"], ell[fr["mr"]:])
        mcoef = s["M"] * (s["qbar"] - float(st["Eq"])) / s["qbar"] ** 2
        V = st["Vh"]                                          # G^-1 (G - H) G^-1 by Stein's lemma, in h
        ri = self._readin_solve(self.I, V, mcoef, ell[:fr["mr"]])
        A = (self.I, V, mcoef)
        self.npass += 1
        new = self._state_from(ri)
        Rx = self.encode(new) - x
        Rt = p.sigA ** 2 * (st["K"] @ t) + tau - self.y
        dl = list(ri["delta"]) + list(st["delta"])
        Rl = (torch.stack(dl) - ell) if len(dl) else torch.zeros(0, dtype=DT, device=self.dev)
        self.last = dict(state=s, t=t, L=L, mid=st, ri=ri, A=A, mcoef=mcoef, new=new)
        return torch.cat([Rx, Rt, Rl])

    # -------------------------------------------------------------------------------------------- frozen quadrature
    def full_pass(self, u, mr_groups=None, mh_groups=None):
        """Evaluate both distributions on the full grids at u (with the given groups/logits, if any)."""
        p, S, nx = self.p, self.S, self.nx
        x, tau = u[:nx], u[nx:nx + S]
        s = self.decode(x)
        L, self.Linv_full = scaled_cholesky(p.sigB ** 2 * s["J"])
        t = tau / p.lam
        self._mid_setup(L, t, s["qbar"], s["M"], full=True)
        mid = self.mid
        if len(mid.counts) > 1:
            # basins and excluded mass only: one node on each spectator direction (the mass per cell barely
            # depends on them), then restore the full rule
            keep = (mid.zeta, mid.wz, mid.logwz, mid.zz, mid.Zz2, mid.A)
            z, w = mid._product_rule([mid.counts[0]] + [1] * (len(mid.counts) - 1))
            mid._set_zeta(z, w)
            mid.zz = (mid.zeta[:, :, None] * mid.zeta[:, None, :]).reshape(len(mid.zeta), -1)
            mid.Zz2 = (mid.wz[:, None] * mid.zz).sum(0).reshape(p.nS - 1, p.nS - 1)
            mid.A = mid.zeta @ ((L @ mid.Qp).T @ mid.Phi)
            c = mid.cells(t)
            mid.zeta, mid.wz, mid.logwz, mid.zz, mid.Zz2, mid.A = keep
        else:
            c = mid.cells(t)
        return s, L, t, c

    def build_frame(self, u, ell_prev):
        """Basins, active cells and logits at u. Returns the new unknown vector (with logits)."""
        p, S, nx = self.p, self.S, self.nx
        s, L, t, c = self.full_pass(u)
        mid = self.mid
        # hidden-field basins on the full (s, q_h) grid
        Lh = c["L"]
        hg = make_groups(mid.action_grid(c), mid.seed, self.include_floor)
        gfull = hg.group_index()
        # active cells: in every basin, cells within cell_floor of that basin's largest cell
        keep = torch.zeros_like(Lh, dtype=torch.bool)
        for k in range(hg.m + 1):
            mk = gfull == k
            mx = torch.where(mk, Lh, -torch.inf).max()
            keep |= mk & (Lh > mx - self.cell_floor)
        si, qi = torch.nonzero(keep, as_tuple=True)
        gi = gfull[si, qi]
        # read-in basins at the same point: one pass of the read-in action with the full hidden distribution
        mid.groups = Groups()
        st = mid.stats(c)
        mcoef = s["M"] * (s["qbar"] - float(st["Eq"])) / s["qbar"] ** 2
        A = self.readin.action(self.I, st["Vh"], mcoef)
        rg, _ = self._readin_frame(A)
        rg_delta = rg.apply(-A + self.readin.logm)[1]
        hg_delta = hg.apply(Lh)[1]
        self.frame = dict(si=si, qi=qi, gi=gi, mr=rg.m, mh=hg.m, ncell=int(keep.sum()), ntot=keep.numel())
        m = rg.m + hg.m
        if ell_prev is not None and len(ell_prev) == m:
            ell = np.array(ell_prev, float)
        else:
            ell = np.clip(np.array(list(rg_delta) + list(hg_delta), float), self.include_floor, 0.0) \
                if m else np.zeros(0)
        return torch.cat([u[:nx + S], torch.tensor(ell, dtype=DT, device=self.dev)])

    def check_frame(self, u):
        """At a solution: log of the mass outside the active cells, and whether new far basins appeared."""
        p, S, nx = self.p, self.S, self.nx
        fr = self.frame
        s, L, t, c = self.full_pass(u)
        mid = self.mid
        Lh = c["L"]
        lab_new = make_groups(mid.action_grid(c), mid.seed, self.nu.basin_floor)
        # weights actually used in the solve, extended to every cell of the full grid by group membership
        hg = make_groups(mid.action_grid(c), mid.seed, -math.inf)
        keep = torch.zeros_like(Lh, dtype=torch.bool)
        keep[fr["si"], fr["qi"]] = True
        out = float(torch.logsumexp(torch.where(keep, -torch.inf, Lh).reshape(-1), 0)
                    - torch.logsumexp(Lh.reshape(-1), 0))
        Linv, V, mcoef = self.last["A"]
        A = self.readin.action(Linv, V, mcoef)
        rg_new = make_groups(A, self.readin.seed, self.nu.basin_floor)
        Lp = -A + self.readin.logm
        keep_r = self.rframe["keep"]
        out_r = float(torch.logsumexp(torch.where(keep_r, -torch.inf, Lp).reshape(-1), 0)
                      - torch.logsumexp(Lp.reshape(-1), 0))
        return max(out, out_r), lab_new.m, rg_new.m

    # -------------------------------------------------------------------------------------------- Newton
    def _fd_jacobian(self, u, R0):
        n = len(u)
        Jm = torch.zeros(len(R0), n, dtype=DT, device=self.dev)
        for j in range(n):
            h = 1e-6 * (1 + abs(float(u[j])))
            e = u.clone()
            e[j] += h
            Rj = self.residual(e)
            if Rj is None:
                e[j] -= 2 * h
                Rj = self.residual(e)
                h = -h
            Jm[:, j] = (Rj - R0) / h
        return Jm

    def newton(self, u, Jm=None, maxit=12):
        R = self.residual(u)
        if R is None:
            raise NoConvergence("G not positive definite at the start")
        fresh = False
        if Jm is None or Jm.shape != (len(R), len(u)):
            Jm, fresh = self._fd_jacobian(u, R), True
        hist = []
        for it in range(maxit):
            r = float(R.abs().max())
            if r < self.newton_tol:
                return u, R, Jm, it
            hist.append(r)
            if not fresh and len(hist) > 3 and r > 1e-3 * hist[-4]:
                Jm, fresh = self._fd_jacobian(u, R), True
                hist = [r]
                continue
            if len(hist) > 6 and r > 0.1 * hist[-6] and r > 1e-6:
                if not fresh:
                    Jm, fresh = self._fd_jacobian(u, R), True
                    hist = [r]
                else:
                    self.u_fail = u
                    raise NoConvergence(f"Newton stalled (|R|={r:.1e})")
            step = -torch.linalg.lstsq(Jm, R[:, None], rcond=1e-14).solution[:, 0]
            a = 1.0
            while a > 0.04:
                R1 = self.residual(u + a * step)
                if R1 is not None and float(R1.abs().max()) < (1 - 1e-4 * a) * r:
                    break
                a *= 0.5
            else:
                if fresh:
                    self.u_fail = u
                    raise NoConvergence(f"line search failed (|R|={r:.1e})")
                Jm, fresh = self._fd_jacobian(u, R), True
                continue
            sv = a * step
            Jm = Jm + torch.outer(R1 - R - Jm @ sv, sv) / (sv @ sv)               # Broyden update
            u, R, fresh = u + sv, R1, False
        self.u_fail = u
        raise NoConvergence(f"Newton did not converge (|R|={float(R.abs().max()):.1e})")

    def solve_at(self, u_pred, ell_prev, Jm):
        """Solve at the current chi from the predicted unknowns. Rebuilds the frozen quadrature if the check
        after the solve finds mass outside the active cells or new far basins."""
        u = self.build_frame(u_pred, ell_prev)
        retried = False
        for attempt in range(5):
            try:
                u, R, Jm, its = self.newton(u, Jm)
            except NoConvergence as e:
                if retried or getattr(self, "u_fail", None) is None:
                    raise
                retried = True
                uf = self.u_fail
                self.u_fail = None
                self.log(f"    {e}; rebuilding the quadrature at the last iterate")
                try:
                    u = self.build_frame(uf, None)
                except torch.linalg.LinAlgError:
                    raise e
                Jm = None
                continue
            out, mh_new, mr_new = self.check_frame(u)
            ell = u[self.nx + self.S:]
            negligible = False
            if out < self.excluded_max and mh_new <= self.frame["mh"] and mr_new <= self.frame["mr"] \
                    and not negligible:
                return u, R, Jm, its
            self.log(f"    rebuilding quadrature (mass outside active cells e^{out:.1f}, far basins "
                     f"read-in {self.frame['mr']}->{mr_new}, hidden {self.frame['mh']}->{mh_new}, "
                     f"negligible logit {negligible})")
            keep_ell = None if negligible else ell.cpu().numpy()
            u = self.build_frame(u, keep_ell if keep_ell is not None and len(keep_ell) ==
                                 self.frame["mr"] + self.frame["mh"] else None)
            Jm = None
        raise NoConvergence("quadrature did not settle")

    def solve(self, chi_start=None, init=None, refine=True, refine_tol=1e-5, max_refine=3):
        """Solve, then check the transverse zeta rule: re-solve (warm) with every node count doubled until mu_1,
        mu_3 and the far-basin masses change by less than refine_tol. The returned results are those of the
        finest rule; results["zeta_refinement"] lists the changes."""
        t_all = time.time()
        r = self._solve(chi_start, init)
        if not refine:
            return r
        from dataclasses import replace
        hist, cur = [], self
        for _ in range(max_refine):
            lv = tuple((thr, 2 * n) for thr, n in cur.nu.zeta_levels)
            S2 = FastSolver(self.p, replace(cur.nu, zeta_levels=lv), self.dev, self.verbose, self.newton_tol,
                            self.cell_floor, self.excluded_max, self.include_floor)
            r2 = S2._solve(None, cur.state())
            ch = max(abs(r2["L1"] - r["L1"]), abs((r2["L3"] or 0) - (r["L3"] or 0)),
                     max([abs(a - b) for a, b in zip(r2["readin_far_mass"], r["readin_far_mass"])] or [0.0]))
            hist.append(dict(counts=S2.mid.counts, change=ch))
            self.log(f"zeta rule {S2.mid.counts}: change {ch:.1e}")
            cur, r = S2, r2
            if ch < refine_tol:
                break
        self.__dict__.update({k: v for k, v in cur.__dict__.items()})
        r["zeta_refinement"] = hist
        r["walltime_s"] = time.time() - t_all
        return r

    def _solve(self, chi_start=None, init=None):
        """Continuation in chi from chi/64, or (init = another FastSolver's .state()) a direct solve from a nearby
        solution."""
        t0 = time.time()
        p = self.p
        if init is not None:
            u0 = init["u"].to(self.dev)
            n0 = self.nx + self.S
            try:
                u, R, Jm, its = self.solve_at(u0[:n0], u0[n0:].cpu().numpy() if len(u0) > n0 else None, None)
                self.u = u
                self.walltime = time.time() - t0
                self.log(f"warm start: Newton {its} its, |R|={float(R.abs().max()):.1e} ({self.walltime:.1f}s)")
                return self.results()
            except (NoConvergence, torch.linalg.LinAlgError) as e:
                self.log(f"warm start failed ({e}); continuation in chi")
        chi_t = p.chi
        lc, lt = math.log(chi_start or chi_t / 64), math.log(chi_t)
        p.chi = math.exp(lc)
        # initial unknowns: prior read-in state, readout from the prior hidden field
        x0 = self.encode(self.state0)
        u = torch.cat([x0, torch.zeros(self.S, dtype=DT, device=self.dev)])
        s0 = self.state0
        L0, _ = scaled_cholesky(p.sigB ** 2 * s0["J"])
        self._mid_setup(L0, self.y, s0["qbar"], s0["M"], full=True)
        self.mid.groups = Groups()
        K0 = self.mid.stats(self.mid.cells(torch.zeros(self.S, dtype=DT, device=self.dev)))["K"]
        tau0 = p.lam * torch.linalg.solve(p.sigA ** 2 * K0 + p.lam * self.I, self.y)
        u = torch.cat([x0, tau0])
        hist = []
        self.since_fail = 2
        Jm, ell_prev = None, None
        step = self.max_step
        while True:
            # secant predictor in log chi (same basin count)
            if len(hist) >= 2 and len(hist[-1][1]) == len(hist[-2][1]):
                (l1, u1), (l2, u2) = hist[-2], hist[-1]
                u_pred = u2 + (u2 - u1) * (lc - l2) / (l2 - l1)
            else:
                u_pred = u
            try:
                try:
                    scaled_cholesky(self.p.sigB ** 2 * self.decode(u_pred[:self.nx])["J"])
                except torch.linalg.LinAlgError:
                    u_pred = u                                    # predictor left the PD cone: use last solution
                u_new, R, Jm, its = self.solve_at(u_pred[:self.nx + self.S], u_pred[self.nx + self.S:]
                                                  .cpu().numpy() if len(u_pred) > self.nx + self.S else None,
                                                  Jm if self.nx + self.S > 15 else None)
                u = u_new
                hist.append((lc, u.clone()))
                self.since_fail += 1
                ell_prev = u[self.nx + self.S:].cpu().numpy()
                self.log(f"chi={p.chi:.4g}: Newton {its} its, |R|={float(R.abs().max()):.1e}, "
                         f"cells {self.frame['ncell']}/{self.frame['ntot']}, far logits "
                         f"{np.round(ell_prev, 3).tolist()}  ({time.time() - t0:.1f}s, {self.npass} passes)")
                if lc >= lt - 1e-12:
                    break
                # next step from how much the solution moved in this one
                if len(hist) >= 2 and len(hist[-1][1]) == len(hist[-2][1]):
                    du = (hist[-1][1] - hist[-2][1]).abs()
                    nxs = self.nx + self.S
                    move = max(float(du[:nxs].max()) / 2.0, float(du[nxs:].max()) / 2.0 if len(du) > nxs else 0.0)
                    used = hist[-1][0] - hist[-2][0]
                    grow = 1.0 if self.since_fail < 2 else 2.0
                    step = min(self.max_step, used * min(grow, 1.0 / max(move, 1e-3)))
                else:
                    step = min(step * 1.5, self.max_step)
            except (NoConvergence, torch.linalg.LinAlgError) as e:
                self.log(f"chi={p.chi:.4g}: {e}; halving the step")
                if not hist:
                    # the very first point is already too strongly coupled: start further down in chi
                    self.first_fail = getattr(self, "first_fail", 0) + 1
                    if self.first_fail > 4:
                        p.chi = chi_t
                        raise
                    lc -= math.log(8.0)
                    p.chi = math.exp(lc)
                    self.log(f"  restarting the continuation at chi={p.chi:.3g}")
                    continue
                lc = hist[-1][0]
                u = hist[-1][1]
                step /= 2
                self.since_fail = -1
                Jm = None
                if step < 1e-3:
                    p.chi = chi_t
                    raise
            lc = min(lc + step, lt)
            p.chi = math.exp(lc)
        p.chi = chi_t
        self.u = u
        self.walltime = time.time() - t0
        return self.results()

    def state(self):
        return dict(u=self.u.clone())

    def results(self):
        p, lt = self.p, self.last
        st, ri, L = lt["mid"], lt["ri"], lt["L"]
        G = p.sigB ** 2 * lt["state"]["J"]
        H = L @ st["Ezz"] @ L.T
        K, t = st["K"], lt["t"]
        fbar = p.sigA ** 2 * (K @ t)
        npy = lambda a: a.detach().cpu().numpy().tolist()
        ell = self.u[self.nx + self.S:].cpu().numpy().tolist()
        mr = self.frame["mr"]
        frac = lambda ds: [math.exp(d - float(np.logaddexp.reduce([0.0] + list(ds)))) for d in ds]
        return dict(params=asdict(p), converged=True, walltime_s=self.walltime, middle_passes=self.npass,
                    d_w0sq=float(p.d * ri["w0sq"]), qperp_over_prior=float(ri["qmean"] / self.readin.q0),
                    J=npy(ri["J"]), H=npy(H), K=npy(K), G=npy(G), t=npy(t), fbar=npy(fbar),
                    L1=float(fbar[0]), L3=float(fbar[1] / p.eps) if p.nS > 1 else None,
                    qh_over_qbar=float(st["Eq"] / lt["state"]["qbar"]), qbar=lt["state"]["qbar"],
                    M=lt["state"]["M"], m_perp=lt["mcoef"],
                    readin_far_mass=frac(ell[:mr]), hidden_far_mass=frac(ell[mr:]), far_logits=ell)


def scaled_cholesky(G):
    """L with L L^T = G, and L^-1, computed from the correlation matrix: G = D R D, D = diag(sqrt(G_nn)).
    The diagonal of G spans ~15 decades at n_max = 11; a plain Cholesky factorisation then loses all relative
    precision in the smallest directions, and L^-1 c (entries ~1e8) turns that into O(1) noise in the read-in
    action. R is well conditioned, and L^-1 = R_chol^-1 D^-1 applies D^-1 elementwise."""
    d = torch.sqrt(torch.diagonal(G))
    R = G / torch.outer(d, d)
    Lr = torch.linalg.cholesky(R)
    I = torch.eye(len(d), dtype=G.dtype, device=G.device)
    Lr_inv = torch.linalg.solve_triangular(Lr, I, upper=False)
    return d[:, None] * Lr, Lr_inv / d[None, :]


def _pair_trace(m, w0, q, ks, u, wu):
    """E_{w,w'} k(w,w')^2 over atoms (mass m, position (w0, q)) and the matching ||J||_F^2."""
    from fcn3_matrix_solver import readin_features
    Ca = readin_features(w0, q, ks)
    J_at = (Ca[:, None, :] * Ca[None, :, :] * m).sum(-1)
    T = 1 + 2 * (w0 * w0 + q)
    sq = torch.sqrt(q)
    num = 2 * (torch.outer(w0, w0)[..., None] + torch.outer(sq, sq)[..., None] * u)
    z = num / torch.sqrt(torch.outer(T, T))[..., None]
    k2 = ((2 / math.pi) * torch.asin(z.clamp(-1, 1))) ** 2 @ wu
    return m @ k2 @ m, J_at


def _trapz(x):
    dx = x[1:] - x[:-1]
    w = torch.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


if __name__ == "__main__":
    import sys
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    S = FastSolver(Params(nmax=nmax))
    r = S.solve()
    print({k: r[k] for k in ("walltime_s", "middle_passes", "L1", "L3", "d_w0sq", "qbar", "M", "qh_over_qbar",
                             "readin_far_mass", "hidden_far_mass", "far_logits")})
