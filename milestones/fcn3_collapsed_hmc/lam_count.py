#!/usr/bin/env python3
"""
Equilibrium outer-well occupancy of the read-in layer by partly disconnecting one tagged read-in neuron.

Why: a read-in neuron's well is carried jointly by its incoming weights (w0, q) and by its outgoing column B[:, j]
(outer neurons share one outgoing pattern, the "team pattern"). Switching needs both to change together, which no
local move does, so the number n of outer neurons never changes in a chain.

Method. Neuron j's feature enters the network as lam * erf(w_j . x) (its weights keep their ordinary prior). At
lam = 0 the neuron is invisible to the data and changes freely; at lam = 1 it is the real network. Fixed-lam windows are
sampled exactly (HMC with the lam-scaled energy, middle and read-in jump sweeps; the read-in sweep skips neuron j while
lam < 1). Every few iterations each window records the energy at every lam of the grid, which gives, by MBAR over the
windows, the free energy along lam for each identity of neuron j separately:
    team member  (X = out):  zeta_j = cos(B[:, j], team pattern) > ZETA_THR
    ordinary     (X = in):   otherwise
Within one identity a window mixes quickly; across identities only at small lam. Tying the two at a small anchor lam0,
    P_1(out) / P_1(in) = [P_lam0(out) / P_lam0(in)] * exp(-[F_out(1) - F_out(lam0)] + [F_in(1) - F_in(lam0)]),
and with m the number of OTHER outer neurons (read-in neurons are exchangeable)
    P(n = m+1) / P(n = m) = R(m) (N1 - m)/(m + 1),   R(m) = P_1(out) / P_1(in).

Subcommands
    window   run one or more fixed-lam windows from a start state; saves records per window
    seed     ramp lam from a window's saved state to make start states for further windows
    analyze  identity-resolved MBAR over all windows of a run folder -> R(m), P(m+1)/P(m)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from fcn3_sampler import FCN3, make_data

HERE = Path(__file__).resolve().parent
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ZETA_THR = 0.15
K_WALL = 5000.0          # soft wall on the team side: V = K_WALL/2 (ZETA_THR - zeta)^2 for zeta < ZETA_THR
R2_THR = 0.18


# ------------------------------------------------------------------------------------------------ coupling of neuron j
class Lam:
    """Mixin: neuron j's feature scaled by lam. Equivalent to scaling column j of B by lam in the data term while its
    prior stays 0.5 |ZB[:, j]|^2."""

    def set_lam(self, j, lam):
        self.lj, self.lam_ = int(j), float(lam)
        fs = getattr(self, "fsweep", None)
        if fs is not None:
            fs.lam.copy_(torch.tensor([float(j), float(lam)], dtype=torch.float64))

    def set_conf(self, side, thr, others, k=0.0):
        """Confine zeta_j = cos(s_j B[:, j], u), u = sum_{i in others} s_i B[:, i] the CURRENT team pattern of the
        other outer neurons (s = sign w0), to zeta > thr (side +1) or zeta <= thr (side -1); side 0: off. The other
        outer neurons are left out of the read-in jump sweep (HMC still moves them)."""
        self.conf_side, self.conf_thr, self.conf_k = int(side), float(thr), float(k)
        self.others = torch.as_tensor(others, dtype=torch.long, device=self.dev)
        fs = getattr(self, "fsweep", None)
        if fs is not None:
            fs.conf.copy_(torch.tensor([float(self.lj), float(side), float(thr), 1.0, float(k)], dtype=torch.float64))
            fs.sO.zero_()
            if side != 0:
                fs.sO[self.others] = 1.0

    def zeta(self, ZW, ZB):
        s = torch.sign(ZW[:, 0].double())
        u = ZB[:, self.others].double() @ s[self.others]
        b = ZB[:, self.lj].double() * s[self.lj]
        return float(b @ u) / (float(b.norm()) * float(u.norm()))

    def in_region(self, ZW, ZB):
        side = getattr(self, "conf_side", 0)
        if side in (0, 2):                     # none, or the soft wall (no hard rejection)
            return True
        z = self.zeta(ZW, ZB)
        return z > self.conf_thr if side > 0 else z <= self.conf_thr

    def hmc_step(self, ZW, ZB, u=None, gW=None, gB=None, step=0.05, n_leap=24):
        """HMC on the confined target: a trajectory that ends outside the region is rejected (exact: the restricted
        density is zero there)."""
        out = super().hmc_step(ZW, ZB, u, gW, gB, step, n_leap)
        zW, zB = out[0], out[1]
        if zW is not ZW and not self.in_region(zW, zB):
            if hasattr(self, "_cache"):
                self._cache = None
            return (ZW, ZB, u if u is not None else out[2], gW, gB, 0.0, out[6])
        return out

    def flip_signs(self, ZW, ZB):
        """Read-in sign flips only (a middle-neuron flip would change the meaning of the fixed u0)."""
        s1 = (torch.randint(0, 2, (self.N1,), device=self.dev, generator=self.gen) * 2 - 1).to(self.dt)
        return ZW * s1[:, None], ZB * s1[None, :]

    def _scaled(self, ZB, lam=None):
        lam = self.lam_ if lam is None else lam
        Z = ZB.clone()
        Z[:, self.lj] *= lam
        return Z

    def _prior_fix(self, ZB, lam):
        return 0.5 * (1 - lam * lam) * float((ZB[:, self.lj].double() ** 2).sum())

    def wall(self, ZW, ZB, grad=True):
        """Soft wall (side 2): V = k/2 (thr - zeta)^2 for zeta < thr; returns (V, dV/dZB or None)."""
        if getattr(self, "conf_side", 0) != 2:
            return 0.0, None
        s = torch.sign(ZW[:, 0].double())
        O, j = self.others, self.lj
        b = ZB[:, j].double() * s[j]
        u = ZB[:, O].double() @ s[O]
        nb, nu = b.norm(), u.norm()
        zeta = (b @ u) / (nb * nu)
        gap = self.conf_thr - float(zeta)
        if gap <= 0:
            return 0.0, (torch.zeros_like(ZB) if grad else None)
        V = 0.5 * self.conf_k * gap * gap
        if not grad:
            return V, None
        dV = -self.conf_k * gap                                    # dV/dzeta
        dzb = u / (nb * nu) - zeta * b / nb ** 2
        dzu = b / (nb * nu) - zeta * u / nu ** 2
        g = torch.zeros(ZB.shape, dtype=torch.float64, device=ZB.device)
        g[:, j] = dV * s[j] * dzb
        g[:, O] = dV * dzu[:, None] * s[O][None, :]
        return V, g.to(ZB.dtype)

    # reference path
    def U(self, ZW, ZB, grad=True):
        lam = self.lam_
        u, gW, gB = super().U(ZW, self._scaled(ZB), grad)
        u = u + self._prior_fix(ZB, lam)
        V, gV = self.wall(ZW, ZB, grad and gB is not None)
        u = u + V
        if gB is not None:
            gB = gB.clone()
            gB[:, self.lj] = ZB[:, self.lj] + lam * (gB[:, self.lj] - lam * ZB[:, self.lj])
            if gV is not None:
                gB = gB + gV
        return u, gW, gB

    # fast path
    def U_exact(self, ZW, ZB):
        u, fac = super().U_exact(ZW, self._scaled(ZB))
        return u + self._prior_fix(ZB, self.lam_) + self.wall(ZW, ZB, grad=False)[0], fac

    def grad32(self, ZW, ZB):
        lam = self.lam_
        gW, gB, ok = super().grad32(ZW, self._scaled(ZB))
        gB = gB.clone()
        gB[:, self.lj] = ZB[:, self.lj] + lam * (gB[:, self.lj] - lam * ZB[:, self.lj])
        _, gV = self.wall(ZW, ZB)
        if gV is not None:
            gB = gB + gV.to(gB.dtype)
        return gW, gB, ok

    def draw_a(self, W, B):
        Bl = B.clone()
        Bl[:, self.lj] *= self.lam_
        return FCN3.draw_a(self, W, Bl)

    def u_at(self, ZW, ZB, lam):
        """Exact energy of the lam-network (for MBAR)."""
        if hasattr(self, "fsweep"):
            from fcn3_fast import FastFCN3
            u, _ = FastFCN3.U_exact(self, ZW, self._scaled(ZB, lam))
        else:
            u, _, _ = FCN3.U(self, ZW, self._scaled(ZB, lam), grad=False)
        return u + self._prior_fix(ZB, lam)


def make_model(cfg, dev, fast=True, seed=1000):
    a = cfg["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
    X, Y = X.float(), Y.float()
    base = FCN3
    if fast:
        from fcn3_fast import FastFCN3 as base
    cls = type("LamFCN3", (Lam, base), {})
    ks = tuple(range(1, a["nmax"] + 1, 2))
    return cls(X, Y, a["N1"], a["N2"], a["chi"], a["kappa"], a["sigA"], a["sigB"], a["sigw"], ks=ks,
               n_cand=a["n_cand"], cg_iters=a["cg_iters"], redraw_frac=a["redraw_frac"], seed=seed)


def identity(m, ZW, ZB, j):
    """r^2 of j, zeta_j = cos(B[:, j], team pattern of the OTHER outer neurons), number of other outer neurons."""
    W = m.sw * ZW.double()
    w0 = W[:, 0]
    r2 = w0 ** 2 / (1 + 2 * (W * W).sum(1))
    outer = r2 > R2_THR
    s = torch.sign(w0)
    oth = outer.clone()
    oth[j] = False
    Bs = ZB.double() * s[None, :]
    u = Bs[:, oth].sum(1)
    zeta = float((Bs[:, j] @ u) / (Bs[:, j].norm() * u.norm()))
    return float(r2[j]), zeta, int(oth.sum())


def run_window(m, ZW, ZB, step, iters, burn, leap=24, sweep_every=2, rec_every=5, rec=None, grid=GRID,
               target_acc=0.8, log=None):
    fast = any(c.__name__ == "FastFCN3" for c in type(m).__mro__)
    u, gW, gB = None, None, None
    accs = []
    for it in range(iters):
        ZW, ZB = m.flip_signs(ZW, ZB)
        if not fast:
            u, gW, gB = m.U(ZW, ZB)
        ZW, ZB, u, gW, gB, acc, L = m.hmc_step(ZW, ZB, u, gW, gB, step, leap)
        if it < burn:
            step = math.exp(math.log(step) + (acc - target_acc) * 2.0 / math.sqrt(it + 10))
        accs.append(acc)
        if sweep_every and fast and it % sweep_every == 0:
            W, B = m.sw * ZW, m.sb * ZB
            a = m.draw_a(W, B)
            W, B, a, _ = m.sweep(W, B, a, ("middle", "readin"))
            ZW, ZB = W / m.sw, B / m.sb
        if rec is not None and it >= burn and (it - burn) % rec_every == 0:
            r2, zeta, mo = identity(m, ZW, ZB, m.lj)
            if hasattr(m, "others"):
                zeta = m.zeta(ZW, ZB)
            us = [m.u_at(ZW, ZB, l) for l in grid]
            rec.append([r2, zeta, mo] + us)
        if log and it % 50 == 0:
            r2, zeta, mo = identity(m, ZW, ZB, m.lj)
            log(f"    it {it:4d} acc {acc:.2f} step {step:.4f}  r2_j {r2:.3f} zeta_j {zeta:+.3f} m {mo}")
    return ZW, ZB, step, float(np.mean(accs[burn:] if iters > burn else accs))


# ------------------------------------------------------------------------------------------------ subcommands
def load_state(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["ZW"].to(dev, torch.float32), ck["ZB"].to(dev, torch.float32), ck.get("step", 0.04)


def cmd_init(a):
    """Make a run folder from a chain's final state: tag an ordinary (inner) neuron."""
    dev = torch.device(a.device)
    src = Path(a.src)
    cfg = json.loads((src / "config.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ZW, ZB, step = load_state(src / "checkpoint.pt", dev)
    m = make_model(cfg, dev)
    W = m.sw * ZW.double()
    r2 = (W[:, 0] ** 2 / (1 + 2 * (W * W).sum(1))).cpu().numpy()
    pool = np.where(r2 < 0.03)[0] if a.tag == "inner" else np.where(r2 > R2_THR)[0]
    j = int(pool[np.argmin(abs(r2[pool] - np.median(r2[pool])))])
    r2j, zeta, mo = identity(m, ZW, ZB, j)
    Wd = m.sw * ZW.double()
    r2a = Wd[:, 0] ** 2 / (1 + 2 * (Wd * Wd).sum(1))
    oth = r2a > R2_THR
    oth[j] = False
    u0 = (ZB.double() * torch.sign(Wd[:, 0])[None, :])[:, oth].sum(1)
    torch.save(u0.cpu(), out / "u0.pt")
    zs = ((ZB.double() * torch.sign(Wd[:, 0])[None, :]).T @ u0) / (ZB.double().norm(dim=0) * u0.norm())
    print(f"zeta wrt u0: outer neurons {float(zs[oth].mean()):.3f} +- {float(zs[oth].std()):.3f}, "
          f"others {float(zs[~oth].mean()):.3f} +- {float(zs[~oth].std()):.3f}")
    meta = dict(src=str(src), j=j, tag=a.tag, m_others=mo, others=torch.where(oth)[0].tolist(), N1=cfg["args"]["N1"], N2=cfg["args"]["N2"], grid=GRID,
                cfg=cfg, zeta_thr=ZETA_THR)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step), out / "start_base.pt")
    print(f"tagged neuron {j}: r2 {r2j:.4f}, zeta {zeta:+.3f}; other outer neurons m = {mo}")


def cmd_window(a):
    dev = torch.device(a.device)
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    m = make_model(meta["cfg"], dev, seed=a.seed)
    for lam in a.lams:
        tag = f"{a.label}_l{lam:.3f}"
        f = out / f"win_{tag}.npz"
        start = a.start.replace("LAM", f"{lam:.3f}")
        if f.exists():
            continue
        ZW, ZB, step = load_state(out / start, dev)
        m.set_lam(meta["j"], lam)
        if a.side != "none":
            if a.side == "out":
                m.set_conf(2, meta["zeta_thr"], meta["others"], K_WALL)
            else:
                m.set_conf(-1, meta["zeta_thr"], meta["others"])
            assert a.side == "out" or m.in_region(ZW, ZB), f"start state not in region {a.side}"
        rec, t0 = [], time.time()
        print(f"window {tag}: lam {lam}, start {a.start}", flush=True)
        grid = a.grid if a.grid else GRID
        ZW, ZB, step, acc = run_window(m, ZW, ZB, step, a.iters, a.burn, rec=rec, grid=grid,
                                       log=lambda s: print(s, flush=True))
        x = np.array(rec)
        np.savez(f, lam=lam, side=a.side, r2=x[:, 0], zeta=x[:, 1], m=x[:, 2].astype(int), u=x[:, 3:], grid=np.array(grid),
                 acc=acc, step=step)
        torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step), out / f"end_{tag}.pt")
        print(f"window {tag} done: zeta_j mean {x[:, 1].mean():+.3f} (frac team {np.mean(x[:, 1] > ZETA_THR):.2f}), "
              f"r2_j mean {x[:, 0].mean():.3f}, m {sorted(set(x[:, 2].astype(int)))}, acc {acc:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)


def cmd_seed_in(a):
    """Ordinary-neuron start states for a tagged MEMBER: at lam = 0 its weights are redrawn from the prior (outgoing
    column with zeta <= thr), then lam is ramped up through a.lams (ascending), confined to zeta <= thr."""
    dev = torch.device(a.device)
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    m = make_model(meta["cfg"], dev, seed=a.seed)
    j = meta["j"]
    ZW, ZB, step = load_state(out / "start_base.pt", dev)
    u0 = torch.load(out / "u0.pt")
    g = torch.Generator(device=dev).manual_seed(a.seed)
    ZW[j] = torch.randn(ZW.shape[1], device=dev, generator=g)
    ZB[:, j] = torch.randn(ZB.shape[0], device=dev, generator=g)
    m.set_lam(j, 0.0)
    m.set_conf(-1, meta["zeta_thr"], meta["others"])
    while not m.in_region(ZW, ZB):
        ZB[:, j] = torch.randn(ZB.shape[0], device=dev, generator=g)
    for lam in a.lams:
        m.set_lam(j, lam)
        m.set_conf(-1, meta["zeta_thr"], meta["others"])
        ZW, ZB, step, acc = run_window(m, ZW, ZB, step, a.iters, a.iters)
        r2j, _, mo = identity(m, ZW, ZB, j)
        torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step), out / f"start_{a.label}_l{lam:.3f}.pt")
        print(f"  seed {a.label} lam {lam:.3f}: r2_j {r2j:.3f} zeta_j(u0) {m.zeta(ZW, ZB):+.3f} m {mo} acc {acc:.2f}",
              flush=True)


def cmd_seed(a):
    """Team-member start states: neuron j gets the incoming weights of an existing outer neuron (own perpendicular
    direction kept, rescaled) and an outgoing column with zeta = 0.35 along u0; relaxed at lam = 1 and then ramped
    down through a.lams (descending), confined to the team region; a start state is saved at each lam."""
    dev = torch.device(a.device)
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    m = make_model(meta["cfg"], dev, seed=a.seed)
    j = meta["j"]
    ZW, ZB, step = load_state(out / "start_base.pt", dev)
    sg = torch.sign(ZW[:, 0].double())
    u0 = ZB.double()[:, meta["others"]] @ sg[meta["others"]]                     # current team pattern
    W = m.sw * ZW.double()
    r2 = W[:, 0] ** 2 / (1 + 2 * (W * W).sum(1))
    k = int(torch.argmin(torch.where(r2 > R2_THR, (r2 - r2[r2 > R2_THR].median()).abs(), torch.inf)))
    zw = ZW[j].clone().double()
    zw[0] = ZW[k, 0].double()
    zw[1:] *= ZW[k, 1:].double().norm() / zw[1:].norm()
    ZW[j] = zw.to(ZW.dtype)
    # outgoing weights: share the team load equally among the n others and j. Team coefficient of neuron i:
    # c_i = s_i B_i . u_hat. Others keep n/(n+1) of theirs, j gets n/(n+1) of their mean (total load unchanged).
    sgn = torch.sign(ZW[:, 0].double())
    uh = u0 / u0.norm()
    oth = r2 > R2_THR
    oth[j] = False
    n = int(oth.sum())
    Zd = ZB.double()
    c = sgn * (Zd.T @ uh)                                               # (N1,)
    for i in torch.where(oth)[0].tolist():
        Zd[:, i] -= sgn[i] * (c[i] / (n + 1)) * uh
    bj = Zd[:, j] - (Zd[:, j] @ uh) * uh                                 # j's own random part, perpendicular to u
    Zd[:, j] = bj + sgn[j] * (n / (n + 1)) * c[oth].mean() * uh
    ZB = Zd.to(ZB.dtype)
    zj = float(sgn[j] * (Zd[:, j] @ uh) / Zd[:, j].norm())
    print(f"  team load shared: n = {n} others, mean team coefficient {float(c[oth].mean()):.2f}; zeta_j = {zj:+.3f}",
          flush=True)
    m.set_lam(j, 1.0)
    m.set_conf(2, meta["zeta_thr"], meta["others"], K_WALL)
    for lam in a.lams:
        m.set_lam(j, lam)
        m.set_conf(2, meta["zeta_thr"], meta["others"], K_WALL)
        ZW, ZB, step, acc = run_window(m, ZW, ZB, step, a.iters, a.iters)
        r2j, _, mo = identity(m, ZW, ZB, j)
        torch.save(dict(ZW=ZW.cpu(), ZB=ZB.cpu(), step=step), out / f"start_{a.label}_l{lam:.3f}.pt")
        print(f"  seed {a.label} lam {lam:.3f}: r2_j {r2j:.3f} zeta_j(u0) {m.zeta(ZW, ZB):+.3f} m {mo} acc {acc:.2f}",
              flush=True)


def mbar(u, N, iters=5000, tol=1e-9):
    """u: (K states, n samples) reduced energies; N: samples per state (sum = n). Returns f (K,) with f[0] = 0."""
    K = u.shape[0]
    f = np.zeros(K)
    lN = np.log(np.maximum(N, 1e-300))
    for _ in range(iters):
        lden = np.logaddexp.reduce(lN[:, None] + f[:, None] - u, axis=0)
        fn = -np.logaddexp.reduce(-u - lden[None, :], axis=1)
        fn -= fn[0]
        if np.max(abs(fn - f)) < tol:
            return fn
        f = fn
    return f


def identity_free_energies(wins, grid, masks, Vs=None):
    """MBAR on the samples of one identity (masks select them in each window). States are the windows' lam values;
    returns F(lam) = -log Z_X(lam) on every grid lam, relative to lam = 1, and the sample counts per state."""
    """MBAR over ALL samples of these windows (each window sampled exp(-U_lam - V), V the soft wall, the same in
    every window); the free energy of the hard region (masks) at every grid lam is
        F(lam) = -log sum_{n in region} exp(-U_lam(x_n)) / sum_k N_k exp(f_k - U_lam_k(x_n) - V(x_n))."""
    lams = sorted(set(float(w["lam"]) for w in wins))
    if not lams:
        return None
    gi = [int(np.argmin(abs(np.asarray(grid) - l))) for l in lams]
    U = np.concatenate([w["u"] for w in wins], 0)                             # (n, G) energies at every grid lam
    V = np.concatenate(Vs if Vs is not None else [np.zeros(len(w["u"])) for w in wins])
    M = np.concatenate(masks)
    N = np.array([sum(len(w["u"]) for w in wins if float(w["lam"]) == l) for l in lams], float)
    U = U - U.min()
    us = U[:, gi].T + V[None, :]                                              # (K, n) energies actually sampled
    f = mbar(us, N)
    lden = np.logaddexp.reduce(np.log(N)[:, None] + f[:, None] - us, axis=0)  # (n,)
    Fg = -np.logaddexp.reduce(np.where(M[None, :], -U.T - lden[None, :], -np.inf), axis=1)   # (G,)
    return Fg - Fg[-1], dict(zip(lams, N.tolist()))


def cmd_analyze(a):
    from scipy.special import betainc
    out = Path(a.out)
    meta = json.loads((out / "meta.json").read_text())
    grid = np.array(meta["grid"])
    wins = [dict(np.load(f)) for f in sorted(out.glob("win_*.npz"))]
    print(f"{len(wins)} windows")
    for w in sorted(wins, key=lambda w: (str(w["side"]), float(w["lam"]))):
        print(f"  {str(w['side']):4s} lam {float(w['lam']):.3f}: {len(w['zeta'])} samples, zeta {w['zeta'].mean():+.3f}, "
              f"r2_j {w['r2'].mean():.3f}, m {sorted(set(w['m'].tolist()))}")
    F = {}
    for X in ("out", "in"):
        ws = [w for w in wins if str(w["side"]) == X]
        if not ws:
            continue
        thr, k = meta["zeta_thr"], meta.get("k_wall", K_WALL)
        if X == "out":
            masks = [w["zeta"] > thr for w in ws]
            Vs = [0.5 * k * np.clip(thr - w["zeta"], 0, None) ** 2 for w in ws]
            print(f"  team side: {np.mean(np.concatenate(masks)):.2f} of the samples inside the region")
        else:
            masks = [w["zeta"] <= thr for w in ws]
            Vs = None
        r = identity_free_energies(ws, grid, masks, Vs)
        F[X] = r[0]
        print(f"  F_{X}(lam) - F_{X}(1):", " ".join(f"{l:.2f}:{v:+.1f}" for l, v in zip(grid, r[0])))
    # lam = 0: neuron j decoupled, B[:, j] isotropic Gaussian -> P(zeta > t) = 1/2 I_{1-t^2}((N2-1)/2, 1/2)
    t, N2 = meta["zeta_thr"], meta["N2"]
    p_out0 = 0.5 * betainc((N2 - 1) / 2, 0.5, 1 - t * t)
    lo0 = math.log(p_out0) - math.log1p(-p_out0)
    print(f"  lam = 0 (prior): P(team region) = {p_out0:.3e}, log odds {lo0:+.2f}")
    if "out" in F and "in" in F:
        k0 = int(np.argmin(abs(grid - 0.0)))
        logR = lo0 + F["out"][k0] - F["in"][k0]                 # -[F_out(1)-F_out(0)] + [F_in(1)-F_in(0)]
        mo = meta["m_others"]
        print(f"  log R(m={mo}) = {logR:+.2f}  ->  log P(n={mo + 1})/P(n={mo}) = "
              f"{logR + math.log((meta['N1'] - mo) / (mo + 1)):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--from", dest="src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--tag", choices=["inner", "outer"], default="inner")
    q = sub.add_parser("window"); q.add_argument("--out", required=True); q.add_argument("--lams", type=float, nargs="+")
    q.add_argument("--start", default="start_base.pt"); q.add_argument("--label", default="base")
    q.add_argument("--side", choices=["in", "out", "none"], default="none")
    q.add_argument("--grid", type=float, nargs="+", default=None, help="lam values at which energies are recorded")
    q.add_argument("--iters", type=int, default=400); q.add_argument("--burn", type=int, default=100)
    si = sub.add_parser("seed_in"); si.add_argument("--out", required=True)
    si.add_argument("--lams", type=float, nargs="+"); si.add_argument("--label", default="in")
    si.add_argument("--iters", type=int, default=30)
    s = sub.add_parser("seed"); s.add_argument("--out", required=True); s.add_argument("--lams", type=float, nargs="+")
    s.add_argument("--label", default="out")
    s.add_argument("--iters", type=int, default=30)
    z = sub.add_parser("analyze"); z.add_argument("--out", required=True)
    for x in (p, q, s, si):
        x.add_argument("--device", default="cuda:0")
    for x in (q, s, si):
        x.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dict(init=cmd_init, window=cmd_window, seed=cmd_seed, seed_in=cmd_seed_in, analyze=cmd_analyze)[a.cmd](a)


if __name__ == "__main__":
    main()
