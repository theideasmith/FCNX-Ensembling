#!/usr/bin/env python3
"""The theory with the perpendicular directions locked to their prior, against the full theory and the sampler, on every
theory-bearing panel of plots/ray/readin_ray_fcn2_vs_fcn3.png.

Locked theory:
  FCN2: the q-theory of readin_action_theory_q.py with every read-in neuron's |w_perp|^2 fixed at sigma_w^2 (d-1)/d
        (one q column; same x0 tower He1..He11).
  FCN3: fcn3_matrix_theory/fcn3_locked_perp.py: read-in q fixed at sigma_w^2 (d-1)/d, no tilt on the perpendicular part
        of the read-in activation, and every hidden neuron's perpendicular variance fixed at its prior value
        sigma_B^2 E_prior[P_perp] (no spread, no response to the read-in law).
Full theory and sampler data: the caches of plot_ray_readin_fcn2_vs_fcn3.py and plot_onset.py (not recomputed here).

Locked solves: every ray point (FCN2 d = 75..1200, FCN3 d = 75..600), the fixed-theory points (d = 150, P x1, x2, x4
with kappa ∝ P) and the onset points (d = 150, kappa = 0.015, P = 500 .. 48000). Every point is solved from the default
start and from the full theory's read-in law projected onto the locked family; ray points also by continuation in d and
in P (x 1/8 ... 4 at the ray's kappa), onset points by continuation in P, both ways; fixed-theory multiples also from their x1 solution; the largest W is kept
(Newton only finds local maxima: at ray d = 75 the default start alone missed the best state by 448 in W). Cached in plots/cache/locked_perp_theory.npz (--recompute to redo).

Figures:
  plots/locked_perp/locked_perp_ray.png    read-in actions along the ray (FCN2, FCN3), L1 and L3 against d, the outer
                                           read-in well against d, the fixed-theory read-in actions and the L3 gap
  plots/locked_perp/locked_perp_onset.png  onset (d = 150, kappa = 0.015): L3, L1, outer count against P; per P the
                                           read-in action and the hidden-layer marginals h|He1, h|He3

    python plot_locked_perp.py [--recompute] [--device cuda:0]
"""
import argparse
import gc
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
import plot_ray_readin_fcn2_vs_fcn3 as rr

HERE = Path(__file__).resolve().parent
CACHE = HERE / "plots" / "cache" / "locked_perp_theory.npz"
OUT = HERE / "plots" / "locked_perp"
C_LOCK = "#8a3ffc"          # locked theory
RAY3 = (75, 150, 300, 600)


# ------------------------------------------------------------------------------------------------ FCN2, locked q
def fcn2_locked(ks=(1, 3, 5, 7, 9, 11)):
    """{d: dict(w, nlp, L1, L3)} of the FCN2 q-theory with q fixed at q0 (same parameters and tower as rr.fcn2_data)."""
    sys.path.insert(0, str(rr.FCN2))
    import readin_action_theory_q as RQ
    from chains import chain_runs

    class KernelLocked(RQ.KernelQ):
        def __init__(self, d, s0, wmax=6.0, nw=2401, nq=None, a=0.5, ks=(1, 3)):
            self.d, self.s0, self.ks = d, s0, ks
            self.beta = d / (2 * s0)
            self.w = a * np.sinh(np.linspace(0, np.arcsinh(wmax / a), nw))
            dw = np.diff(self.w)
            tw = np.zeros(nw)
            tw[:-1] += dw / 2
            tw[1:] += dw / 2
            self.q = np.array([s0 * (d - 1) / d])
            self.tw, self.tq = tw, np.ones(1)
            W, Q = np.meshgrid(self.w, self.q, indexing="ij")
            self.C = np.stack([RQ.he_coeff(W, Q, k) for k in ks])
            self.c1, self.c3 = self.C[0], self.C[1]
            self.base = self.beta * W * W                 # the q part of the prior is a constant
            self.logm = np.log(tw)[:, None]
            self.W, self.Q = W, Q

    out = {}
    keep = RQ.KernelQ
    RQ.KernelQ = KernelLocked
    try:
        for d, g in rr.GROUPS2.items():
            a = chain_runs(g)[0][2]
            q = RQ.analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], a["chi"],
                             ks=ks)
            out[d] = dict(w=q["w"], nlp=q["nlp"], L1=float(q["L1"]), L3=float(q["L3"]), P=float(a["P"]), d=d)
            print(f"FCN2 locked d={d}: L1 {q['L1']:.4f} L3 {q['L3']:.4f}", flush=True)
    finally:
        RQ.KernelQ = keep
    return out


# ------------------------------------------------------------------------------------------------ FCN3, locked
def fcn3_params(d, P, kappa):
    from fcn3_matrix_solver import Params
    return Params(d=d, P=float(P), N1=po.N, N2=po.N, chi=1.0, kappa=float(kappa), eps=po.EPS, sigA=math.sqrt(po.SIGA2),
                  sigB=po.SIGB, sigw=math.sqrt(po.SIGW2), nmax=11)


def solve3(p, device, start=None, particles=False):
    """One locked solve; start: read-in exponents theta of another locked solution. Returns what the plots need."""
    import fcn3_fixed as F
    import fcn3_locked_perp as L
    try:
        o = L.solve_locked(p, device=device, theta0=start)
    except Exception as e:
        print(f"  locked P={p.P:g} kappa={p.kappa:g} d={p.d} ({'default' if start is None else 'warm'}) failed: "
              f"{e!r}"[:300])
        return None
    Fn = o.pop("_F")
    w, nlp = L.readin_nlp(Fn)
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy().sum(1)
    r = dict(W=float(o["W"]), L1=float(o["L1"]), L3=float(o["L3"]), theta=np.array(o["theta"]), w=w, nlp=nlp,
             dens=pw / np.gradient(w), conv=bool(o["converged"]))
    if particles:                                   # for po.hidden_marginals (as po._solve)
        pr = Fn.p
        G = pr.sigB ** 2 * Fn.last["J"]
        qbar = pr.sigB ** 2 * Fn.last["T"]
        r.update(G=G.detach().cpu().numpy(), t=Fn.t.detach().cpu().numpy(), gt=float(pr.gt), ks=np.array(pr.ks),
                 H=np.array(o["H"]))
        with torch.no_grad():
            e, h, q = Fn.mc.features(F.scaled_cholesky(G), qbar)
            wt = torch.softmax(Fn.mc.logw(o["M"]) + 0.5 * pr.gt * (e @ Fn.t) ** 2, 0)
            n = 2 ** 14
            u = (torch.rand(1, dtype=wt.dtype, device=wt.device, generator=torch.Generator(wt.device).manual_seed(3))
                 + torch.arange(n, dtype=wt.dtype, device=wt.device)) / n
            idx = torch.searchsorted(torch.cumsum(wt, 0), u).clamp(max=len(wt) - 1)
            r["hs"], r["qs"] = h[idx].cpu().numpy(), q[idx].cpu().numpy()
    del Fn
    gc.collect()
    torch.cuda.empty_cache()
    return r


def best_of(p, device, starts, particles=False, label=""):
    """Largest-W locked solution over the default start and the given warm starts."""
    best = None
    for name, th in [("default", None)] + list(starts):
        r = solve3(p, device, th, particles)
        if r is None:
            continue
        if best is None or r["W"] > best["W"] + 1e-6:
            if best is not None:
                print(f"  {label}: start {name} beats the previous by {r['W'] - best['W']:+.3f} in W")
            best = r
    return best


def full_laws():
    """{key: (w, -log density on |w0|)} of the full theory's read-in law at every locked point (from the caches of
    plot_ray_readin_fcn2_vs_fcn3.py and plot_onset.py; onset densities are pruned 60 nats below the peak)."""
    laws = {f"ray_{d}": (v[0], v[1]) for d, v in rr.fcn3_theory().items()}
    th = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    for k in th:
        if k.startswith("dens_"):
            P, dn = int(k[5:]), th[k]
            laws[f"onset_{P}"] = (th[f"w0_{P}"], np.where(dn > 0, -np.log(np.maximum(dn, 1e-300)), np.inf))
    ekr = dict(np.load(rr.CACHE_EKR))
    for Pt in rr.EK:
        for mult in (2, 4):
            if f"w_{Pt}_{mult}" in ekr:
                laws[f"ek_{Pt * mult}_{po.KAPPA * mult:g}"] = (ekr[f"w_{Pt}_{mult}"], ekr[f"nlp_{Pt}_{mult}"])
    return laws


def projected_starts(key, p, device, laws):
    """Starts from the full theory's read-in law at this point, projected onto the locked family (two conditioning
    levels: the nearly collinear r^2k statistics make one or the other fail numerically at some points)."""
    import fcn3_locked_perp as L
    if key not in laws:
        return []
    F = L.LockedPerpFreeEnergy(p, device=device, verbose=False)
    out = []
    for tol in (1e-9, 1e-6):
        try:
            th = L.project_readin(F, *laws[key], rank_tol=tol)
            if np.isfinite(th).all():
                out.append((f"full-theory law ({tol:g})", th))
        except Exception as e:
            print(f"  {key}: projection at {tol:g} failed: {e!r}"[:200])
    del F
    gc.collect()
    torch.cuda.empty_cache()
    return out


def continuation(keys, params, best, device, label, particles=()):
    """Warm start each point from its neighbour's solution, up the list and then down; keep the larger W.
    particles: keys whose solves also record hidden-field particles."""
    for seq in (keys, keys[::-1]):
        for a_, b_ in zip(seq[:-1], seq[1:]):
            if best.get(a_) is None:
                continue
            r = solve3(params[b_], device, best[a_]["theta"], particles=b_ in particles)
            if r is not None and (best.get(b_) is None or r["W"] > best[b_]["W"] + 1e-6):
                print(f"  {label} {b_}: start from {a_} beats the previous by "
                      f"{r['W'] - (best[b_]['W'] if best.get(b_) else float('nan')):+.3f} in W", flush=True)
                best[b_] = r


def fcn3_locked(device):
    """{key: record}; keys: ray_{d}, ek_{P}_{kappa}, onset_{P}. Every point: the default start and the full theory's
    read-in law projected onto the locked family; ray and onset points also by continuation from their neighbours
    (in d, in P) both ways; fixed-theory multiples also from their x1 solution. The largest W is kept."""
    import glob
    laws = full_laws()
    out = {}
    params = {}
    for d in RAY3:
        a = [json.loads(Path(f).read_text())["args"]
             for f in glob.glob(str(HERE / "runs" / f"*_ray_d{d}_P*_N1500x1500_prior_r1" / "config.json"))][0]
        params[d] = replace(fcn3_params(d, a["P"], a["kappa"]), eps=a["eps"], sigA=a["sigA"], sigB=a["sigB"],
                            sigw=a["sigw"])
    best = {d: best_of(params[d], device, projected_starts(f"ray_{d}", params[d], device, laws), label=f"ray d={d}")
            for d in RAY3}
    continuation(list(RAY3), params, best, device, "ray d =")
    # and in P at each d (kappa fixed at the ray's value): P x 1/8 ... 4, up and down, keeping the best at x1
    for d in RAY3:
        mults = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
        pp = {m: replace(params[d], P=params[d].P * m) for m in mults}
        bp = {m: best_of(pp[m], device, [], label=f"ray d={d} P x{m:g}") for m in mults if m != 1.0}
        bp[1.0] = best[d]
        continuation(list(mults), pp, bp, device, f"ray d={d}, P x")
        best[d] = bp[1.0]
    for d in RAY3:
        out[f"ray_{d}"] = dict(best[d], P=params[d].P, d=d)
        print(f"FCN3 locked ray d={d}: L1 {best[d]['L1']:.4f} L3 {best[d]['L3']:.4f}", flush=True)
    # onset
    groups = po.by_P(po.sampler_runs())
    Ps = sorted(set(groups) | set(po.P_EXTRA))
    params = {P: fcn3_params(po.D, P, po.KAPPA) for P in Ps}
    best = {P: best_of(params[P], device, projected_starts(f"onset_{P}", params[P], device, laws),
                       particles=P in groups, label=f"onset P={P}") for P in Ps}
    continuation(Ps, params, best, device, "onset P =", particles=set(groups))
    for P in Ps:
        r = best[P]
        if P in groups:                                  # hidden-layer marginals at the sampler's P values
            r["h1g"], r["lp1"], r["h3g"], r["lp3"] = po.hidden_marginals(r, device)
        out[f"onset_{P}"] = dict(r, P=P, d=po.D)
        print(f"FCN3 locked onset P={P}: L1 {r['L1']:.4f} L3 {r['L3']:.4f}", flush=True)
    # fixed theory, P doubled twice with kappa ∝ P
    ek = rr.ek_data()
    for Pt, e in ek.items():
        base = out.get(f"onset_{Pt}")
        for mult, run in sorted(e["runs"].items()):
            key = f"ek_{int(run['P'])}_{run['kappa']:g}"
            if key in out:
                continue
            if mult == 1 and base is not None:
                out[key] = base
                continue
            p = fcn3_params(po.D, run["P"], run["kappa"])
            starts = ([("x1", base["theta"])] if base is not None else []) + projected_starts(key, p, device, laws)
            out[key] = dict(best_of(p, device, starts, label=key), P=run["P"], d=po.D)
            print(f"FCN3 locked {key}: L1 {out[key]['L1']:.4f} L3 {out[key]['L3']:.4f}", flush=True)
    return out


def load_or_solve(device, recompute):
    if CACHE.exists() and not recompute:
        z = np.load(CACHE, allow_pickle=True)
        return z["fcn2"].item(), z["fcn3"].item()
    f2, f3 = fcn2_locked(), fcn3_locked(device)
    np.savez(CACHE, fcn2=np.array(f2, dtype=object), fcn3=np.array(f3, dtype=object))
    return f2, f3


# ------------------------------------------------------------------------------------------------ helpers
def sym(w, nlp, lim):
    k = w <= lim
    return np.concatenate([-w[k][::-1], w[k][1:]]), np.concatenate([nlp[k][::-1], nlp[k][1:]])


def hist_nlp(x, edges):
    c, _ = np.histogram(x, edges)
    k = c > 0
    mid = 0.5 * (edges[1:] + edges[:-1])
    return mid[k], -np.log(c[k] / (c.sum() * np.diff(edges)[k]))


def readin_panel(ax, full, locked, w0, title, lim=1.6, nb=81, top_min=12.0):
    """-log p(w0): full theory (solid), locked (dashed), sampler (points). full, locked: (symmetric w, nlp)."""
    xs, ys = hist_nlp(w0, np.linspace(-lim, lim, nb))
    ax.plot(*full, "-", color=po.C_THEORY, lw=2, zorder=2)
    ax.plot(*locked, "--", color=C_LOCK, lw=2, zorder=2)
    ax.plot(xs, ys, "o", color=po.C_HMC, ms=3.4, mec="white", mew=0.3, zorder=3)
    lo = min(ys.min(), np.nanmin(full[1]), np.nanmin(locked[1]))
    inner = [y[np.abs(x) < 1.0] for x, y in (full, locked)]
    top = max(top_min, 1.1 * (max(np.nanmax(v) for v in inner) - lo), ys.max() - lo + 1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lo - 1, lo + top)
    ax.set_xlabel("w$_0$")
    ax.set_title(title, color=po.INK, loc="left", fontsize=10)


def log_d_axis(ax, ds):
    ax.set_xscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.minorticks_off()
    ax.set_xlabel("d")


def alpha(P, d):
    """Training-set exponent: P = d^alpha."""
    return math.log(P) / math.log(d)


def alpha_axis(ax, pts, name):
    """x axis in alpha = ln P / ln d, ticked at the given (alpha, label value) points: 'alpha (name = value)'."""
    seen = {}
    for a, v in sorted(pts):
        if all(abs(a - b) > 0.025 for b in seen):
            seen[a] = v
    ax.set_xticks(list(seen))
    ax.set_xticklabels([f"{a:.2f}\n{name}={v}" for a, v in seen.items()], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("α = ln P / ln d")


# ------------------------------------------------------------------------------------------------ figure 1: ray
def fig_ray(plt, f2l, f3l):
    d2, d3 = rr.fcn2_data(), rr.fcn3_data()
    l2 = rr.fcn2_learn()
    ek = rr.ek_data()
    ek[3000]["theory"] = rr.fcn3_theory()[150]
    fig = plt.figure(figsize=(21, 16.5))
    gs = fig.add_gridspec(3, 6, hspace=0.42, wspace=0.32)
    # row 1: FCN2 read-in action per d
    for i, (d, r) in enumerate(sorted(d2.items())):
        ax = fig.add_subplot(gs[0, i])
        lk = f2l[d]
        readin_panel(ax, (r["wq"], r["th"]), sym(lk["w"], lk["nlp"], 1.6), r["w0"],
                     f"FCN2, d = {d}\nL$_3$ {l2[d][1]:.3f} / {lk['L3']:.3f} / "
                     f"{np.mean([x[1] for x in l2[d][2]]):.3f}")
        if i == 0:
            ax.set_ylabel("−log p(w$_0$)")
    # row 2: FCN3 read-in action per d, and learnabilities against d
    for i, d in enumerate(RAY3):
        ax = fig.add_subplot(gs[1, i])
        r, lk = d3[d], f3l[f"ray_{d}"]
        readin_panel(ax, (r["wq"], r["th"]), sym(lk["w"], lk["nlp"], 1.6), r["w0"],
                     f"FCN3, d = {d}\nL$_3$ {r['L3th']:.3f} / {lk['L3']:.3f} / "
                     f"{np.mean([x[1] for x in r['L']]):.3f}")
        if i == 0:
            ax.set_ylabel("−log p(w$_0$)")
    for j, (key, lab) in enumerate(((0, "L$_1$"), (1, "L$_3$"))):
        ax = fig.add_subplot(gs[1, 4 + j])
        ds2 = sorted(l2)
        a2 = [alpha(f2l[d]["P"], d) for d in ds2]
        a3 = [alpha(f3l[f"ray_{d}"]["P"], d) for d in RAY3]
        ax.plot(a2, [l2[d][key] for d in ds2], "--", color=po.C_THEORY, lw=1.6, label="FCN2, full theory")
        ax.plot(a2, [f2l[d]["L1" if key == 0 else "L3"] for d in ds2], "--", color=C_LOCK, lw=1.6,
                label="FCN2, locked")
        ax.plot(a3, [d3[d]["L1th" if key == 0 else "L3th"] for d in RAY3], "-", color=po.C_THEORY, lw=2,
                label="FCN3, full theory")
        ax.plot(a3, [f3l[f"ray_{d}"]["L1" if key == 0 else "L3"] for d in RAY3], "-", color=C_LOCK, lw=2,
                label="FCN3, locked")
        for a, d in zip(a2, ds2):
            v = [x[key] for x in l2[d][2]]
            ax.plot([a] * len(v), v, "s", color=po.C_HMC, ms=6, mfc="white", mew=1.2, zorder=3)
        for a, d in zip(a3, RAY3):
            v = [x[key] for x in d3[d]["L"]]
            ax.plot([a] * len(v), v, "o", color=po.C_HMC, ms=6, mec="white", mew=0.5, zorder=3)
        ax.plot([], [], "s", color=po.C_HMC, mfc="white", mew=1.2, label="FCN2, sampler")
        ax.plot([], [], "o", color=po.C_HMC, label="FCN3, sampler")
        alpha_axis(ax, list(zip(a2, ds2)) + list(zip(a3, RAY3)), "d")
        ax.set_ylim(0, 1.15 if key == 0 else 1)
        ax.set_title(f"{lab} along the ray", color=po.INK, loc="left", fontsize=11)
        if key == 1:
            ax.legend(fontsize=7.5, loc="lower right")
    # row 3: fixed theory at d = 150 with P doubled (kappa ∝ P), the L3 gap there, the outer well against d
    edges = np.linspace(-1.3, 1.3, 61)
    for i, (Pt, e) in enumerate(ek.items()):
        ax = fig.add_subplot(gs[2, i])
        lo = np.inf
        for mult, run in sorted(e["runs"].items()):
            col = rr.EK_COLS[mult]
            w, nlp = (e["theory"] if mult == 1 else run["theory"])[:2]
            lk = f3l[f"ek_{int(run['P'])}_{run['kappa']:g}"]
            ax.plot(*sym(w, nlp, 1.3), "-", color=col, lw=1.8, zorder=2)
            ax.plot(*sym(lk["w"], lk["nlp"], 1.3), "--", color=col, lw=1.8, zorder=2)
            xs, ys = hist_nlp(run["w0"], edges)
            full_L3 = (e["theory"] if mult == 1 else run["theory"])[3]
            ax.plot(xs, ys, "o", color=col, ms=3.4, zorder=3,
                    label=f"P = {int(run['P'])}: L$_3$ {full_L3:.3f} / {lk['L3']:.3f} / "
                          f"{np.mean(run['L3']):.3f}")
            lo = min(lo, ys.min(), nlp.min())
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(lo - 1, lo + 56)
        ax.set_xlabel("w$_0$")
        if i == 0:
            ax.set_ylabel("−log p(w$_0$)   (d = 150)")
        ax.set_title(f"theory point P = {Pt} (dataset {e['seed']})", color=po.INK, loc="left", fontsize=10.5)
        ax.legend(fontsize=7, loc="upper center", title="L$_3$: full / locked / sampler", title_fontsize=7)
    ax = fig.add_subplot(gs[2, 3])
    cols = {3000: "#9cc3ee", 12000: "#2a78d6", 24000: "#0b3d91"}
    ticks = []
    for Pt, e in ek.items():
        m = sorted(e["runs"])
        s = [np.mean(e["runs"][k]["L3"]) for k in m]
        full = [(e["theory"] if k == 1 else e["runs"][k]["theory"])[3] for k in m]
        lk = [f3l[f"ek_{int(e['runs'][k]['P'])}_{e['runs'][k]['kappa']:g}"]["L3"] for k in m]
        am = [alpha(e["runs"][k]["P"], po.D) for k in m]
        ax.plot(am, np.array(full) - s, "o-", color=cols[Pt], lw=2, ms=5, label=f"theory point P = {Pt}: full")
        ax.plot(am, np.array(lk) - s, "s--", color=cols[Pt], lw=2, ms=5, mfc="white", label=f"P = {Pt}: locked")
        ticks += [(a, int(e["runs"][k]["P"])) for a, k in zip(am, m)]
    ax.axhline(0, color=po.MUTED, lw=0.8, ls=":")
    alpha_axis(ax, ticks, "P")
    ax.text(0.02, 0.02, "d = 150; each line: P × 1, 2, 4 with κ ∝ P", transform=ax.transAxes, fontsize=7.5,
            color=po.MUTED)
    ax.set_ylabel("L$_3$ theory − sampler")
    ax.set_title("L$_3$ gap at fixed theory, κ ∝ P", color=po.INK, loc="left", fontsize=11)
    ax.legend(fontsize=7.5, ncol=1, loc="upper right")
    axm, axs = fig.add_subplot(gs[2, 4]), fig.add_subplot(gs[2, 5])
    for name, dd, lk, ls, mk in (("FCN2", d2, {d: sym(f2l[d]["w"], f2l[d]["nlp"], 1.6) for d in d2}, "--", "s"),
                                 ("FCN3", d3, {d: sym(f3l[f"ray_{d}"]["w"], f3l[f"ray_{d}"]["nlp"], 1.6) for d in d3},
                                  "-", "o")):
        ds = sorted(dd)
        full = [rr.well_stats_theory(dd[d]["wq"], dd[d]["th"]) for d in ds]
        lock = [rr.well_stats_theory(*lk[d]) for d in ds]
        samp = []
        for d, (wb, _, _) in zip(ds, full):
            x = np.abs(dd[d]["w0"])
            o = x[(x > wb) & (x < 1.5)]
            samp.append((o.mean(), o.std()))
        for ax, j in ((axm, 1), (axs, 2)):
            ax.plot(ds, [f[j] for f in full], ls, color=po.C_THEORY, lw=1.8, label=f"{name}, full theory")
            ax.plot(ds, [f[j] for f in lock], ls, color=C_LOCK, lw=1.8, label=f"{name}, locked")
            ax.plot(ds, [s[j - 1] for s in samp], mk, color=po.C_HMC, ms=6, mfc="white" if name == "FCN2" else None,
                    mew=1.2, ls="none", label=f"{name}, sampler")
    for ax, t in ((axm, "outer read-in well: mean |w$_0$|"), (axs, "outer read-in well: width (sd of |w$_0$|)")):
        log_d_axis(ax, [75, 150, 300, 600, 1200])
        ax.set_title(t, color=po.INK, loc="left", fontsize=11)
    axs.set_ylim(0, None)
    axm.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Theory with the perpendicular directions locked to their prior (purple, dashed) against the full theory "
                 "(blue) and the sampler (orange).\nRows 1–2: along the κ$_0$ = 0.015 ray (P = 3000 (d/150)$^2$, "
                 "κ = 0.015 d/150, N = 1500). Row 3: theory held fixed at d = 150 while P doubles with κ ∝ P, "
                 "and the outer read-in well along the ray\n(sampler: |w$_0$| beyond the full theory's barrier). "
                 "Panel titles: L$_3$ full theory / locked / sampler.",
                 color=po.INK, x=0.01, ha="left", fontsize=12.5, y=0.965)
    out = OUT / "locked_perp_ray.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------------------------------------------------------ figure 2: onset
def fig_onset(plt, f3l):
    Ps, zs, th = rr.onset_data()
    runs = po.sampler_runs()
    po.set_outer(runs, th)
    groups = po.by_P(runs)
    Pth = sorted(int(k[3:]) for k in th if k.startswith("sc_") and np.isfinite(th[k]).all())
    zl = {}                                                # locked, in plot_onset's key layout (for barrier_cut)
    for P in Pth:
        r = f3l[f"onset_{P}"]
        zl[f"w0_{P}"], zl[f"dens_{P}"] = r["w"], r["dens"]
    fig = plt.figure(figsize=(3.3 * len(Ps), 17.5))
    gs = fig.add_gridspec(4, len(Ps), height_ratios=[1.15, 1, 1, 1], hspace=0.45, wspace=0.3)
    span = max(1, len(Ps) // 3)
    tops = [(fig.add_subplot(gs[0, k * span:(k + 1) * span] if k < 2 else gs[0, 2 * span:])) for k in range(3)]
    for ax, (key, it, lab, ylim) in zip(tops, (("L3", 1, "L$_3$", (0, 1.05)), ("L1", 0, "L$_1$", (0, 1.5)),
                                               ("outer", 2, "read-in neurons in the outer well (of 1500)", None))):
        sc = po.N if key == "outer" else 1
        full = [po.outer_theory(th, P, po.barrier_cut(th, P)) if key == "outer" else th[f"sc_{P}"][it] for P in Pth]
        lock = [po.outer_theory(zl, P, po.barrier_cut(zl, P)) if key == "outer" else f3l[f"onset_{P}"][key]
                for P in Pth]
        ath = [alpha(P, po.D) for P in Pth]
        ax.plot(ath, sc * np.array(full), "-", color=po.C_THEORY, lw=2, label="full theory")
        ax.plot(ath, sc * np.array(lock), "--", color=C_LOCK, lw=2, label="locked")
        for k, (P, rs) in enumerate(groups.items()):
            vals = [(r[key], 0.0) if key == "outer" else r[key] for r in po.per_dataset(rs)]
            vals = [v for v in vals if np.isfinite(v[0])]
            if not vals:
                continue
            m, e = po.combine(vals)
            ax.errorbar([alpha(P, po.D)], [sc * m], yerr=[sc * e] if np.isfinite(e) else None, fmt="o", ms=7, color=po.C_HMC,
                        mec="white", capsize=0, zorder=4, label="sampler, mean over datasets" if k == 0 else None)
        alpha_axis(ax, [(alpha(P, po.D), P) for P in Ps], "P")
        ax.set_ylim(*(ylim or (0, None)))
        ax.set_title(lab, color=po.INK, loc="left", fontsize=11)
    tops[0].legend(fontsize=8.5, loc="upper left")
    tops[2].text(0.02, 0.97, "outer well: |w$_0$| beyond each theory's barrier (sampler: the full theory's)",
                 transform=tops[2].transAxes, fontsize=8, color=po.MUTED, va="top")
    for c, P in enumerate(Ps):
        r = f3l[f"onset_{P}"]
        # read-in
        ax = fig.add_subplot(gs[1, c])
        w, dn = th[f"w0_{P}"], th[f"dens_{P}"]
        x = zs[f"w0_{P}"]
        lim = max(1.5, 1.1 * np.abs(x).max())
        # the solver keeps only points within 60 nats of the peak: show the pruned part as a gap, not a line
        full = sym(w, np.where(dn > 1e-300, -np.log(0.5 * np.maximum(dn, 1e-300)), np.nan), lim)
        readin_panel(ax, full, sym(r["w"], r["nlp"], lim), x, f"P = {P}  (α = {alpha(P, po.D):.2f})\nL$_3$ full {th[f'sc_{P}'][1]:.3f}, "
                     f"locked {r['L3']:.3f}", lim=lim, nb=61, top_min=14)
        if c == 0:
            ax.set_ylabel("−log p   (read-in w$_0$)")
        # hidden layer
        for row, j in ((2, 1), (3, 3)):
            ax = fig.add_subplot(gs[row, c])
            x = zs[f"h_{P}"][:, j // 2]
            gf, yf = th[f"h{j}g_{P}"], -th[f"lp{j}_{P}"]
            gl, yl = r[f"h{j}g"], -r[f"lp{j}"]
            lim = 1.08 * max(np.abs(x).max(), np.abs(gf[yf <= yf.min() + 9]).max(), np.abs(gl[yl <= yl.min() + 9]).max())
            kf, kl = np.abs(gf) <= lim, np.abs(gl) <= lim
            ax.plot(gf[kf], yf[kf], "-", color=po.C_THEORY, lw=2, zorder=2)
            ax.plot(gl[kl], yl[kl], "--", color=C_LOCK, lw=2, zorder=2)
            xs, ys = hist_nlp(x, np.linspace(-lim, lim, 81))
            ax.plot(xs, ys, "o", color=po.C_HMC, ms=3, mec="white", mew=0.3, zorder=3)
            lo = min(ys.min(), yf.min(), yl.min())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(lo - 0.5, lo + 11)
            ax.set_xlabel(f"h | He$_{j}$")
            if c == 0:
                ax.set_ylabel(f"−log p   (hidden layer, He$_{j}$)")
    fig.suptitle(f"Onset at d = {po.D}, κ = {po.KAPPA}, N = {po.N}: theory with the perpendicular directions locked to "
                 "their prior (purple, dashed), full theory (blue), sampler (orange; all datasets pooled)",
                 color=po.INK, x=0.01, ha="left", fontsize=12.5, y=0.925)
    out = OUT / "locked_perp_onset.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------------------------------------------------------ figure 3: summary
def outer_well(wq, th, wmax=1.5):
    """(barrier |w0|, mean |w0| of the outer well) of a theory read-in law, or (nan, nan) when the law has no outer well
    (no local minimum of -log p in 0.3 < |w0| < wmax behind a barrier). Mean over the density beyond the barrier, below
    wmax (as rr.well_stats_theory)."""
    w, y = np.abs(np.asarray(wq)), np.asarray(th, float)
    o = np.argsort(w)
    w, y = w[o], y[o]
    ok = np.isfinite(y)
    w, y = w[ok], y[ok]
    mins = [j for j in range(1, len(w) - 1) if 0.3 < w[j] < wmax and y[j] < y[j - 1] and y[j] <= y[j + 1]]
    if not mins:
        return np.nan, np.nan
    jo = min(mins, key=lambda j: y[j])
    kb = (w > 0.1) & (w < w[jo])
    if not kb.any() or y[kb].max() < y[jo] + 0.5:
        return np.nan, np.nan
    wb = w[kb][np.argmax(y[kb])]
    p = np.exp(-(y - y.min())) * np.gradient(w)
    oo = (w > wb) & (w < wmax)
    return wb, float((p[oo] * w[oo]).sum() / p[oo].sum())


def sampler_outer(w0, wb, wmax=1.5):
    x = np.abs(np.asarray(w0))
    o = x[(x > wb) & (x < wmax)]
    return float(o.mean()) if len(o) else np.nan


def fig_summary(plt, f2l, f3l):
    """L1, L3 and the outer read-in well's position (mean |w0| of its neurons) against alpha: full theory, locked
    theory, sampler. Top: the ray (FCN2 and FCN3). Bottom: the FCN3 P-sweep at d = 150, kappa = 0.015."""
    d2, d3, l2 = rr.fcn2_data(), rr.fcn3_data(), rr.fcn2_learn()
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    # ---- ray
    ds2 = sorted(l2)
    a2 = [alpha(f2l[d]["P"], d) for d in ds2]
    a3 = [alpha(f3l[f"ray_{d}"]["P"], d) for d in RAY3]
    ticks = list(zip(a2, ds2)) + list(zip(a3, RAY3))
    for j, key in enumerate(("L1", "L3")):
        ax = axs[0, j]
        k = 0 if key == "L1" else 1
        ax.plot(a2, [l2[d][k] for d in ds2], "--", color=po.C_THEORY, lw=1.6, label="FCN2, full theory")
        ax.plot(a2, [f2l[d][key] for d in ds2], "--", color=C_LOCK, lw=1.6, label="FCN2, locked")
        ax.plot(a3, [d3[d][key + "th"] for d in RAY3], "-", color=po.C_THEORY, lw=2, label="FCN3, full theory")
        ax.plot(a3, [f3l[f"ray_{d}"][key] for d in RAY3], "-", color=C_LOCK, lw=2, label="FCN3, locked")
        for a, d in zip(a2, ds2):
            v = [x[k] for x in l2[d][2]]
            ax.plot([a] * len(v), v, "s", color=po.C_HMC, ms=6, mfc="white", mew=1.2, zorder=3)
        for a, d in zip(a3, RAY3):
            v = [x[k] for x in d3[d]["L"]]
            ax.plot([a] * len(v), v, "o", color=po.C_HMC, ms=6, mec="white", mew=0.5, zorder=3)
    ax = axs[0, 2]
    for name, dd, lk, ls, mk, aa in (("FCN2", d2, {d: sym(f2l[d]["w"], f2l[d]["nlp"], 1.6) for d in d2}, "--", "s",
                                      {d: alpha(f2l[d]["P"], d) for d in d2}),
                                     ("FCN3", d3, {d: sym(f3l[f"ray_{d}"]["w"], f3l[f"ray_{d}"]["nlp"], 1.6)
                                                   for d in d3}, "-", "o",
                                      {d: alpha(f3l[f"ray_{d}"]["P"], d) for d in d3})):
        ds = sorted(dd)
        full = [outer_well(dd[d]["wq"], dd[d]["th"]) for d in ds]
        lock = [outer_well(*lk[d]) for d in ds]
        samp = [sampler_outer(dd[d]["w0"], f[0]) for d, f in zip(ds, full)]
        x = [aa[d] for d in ds]
        ax.plot(x, [f[1] for f in full], ls, color=po.C_THEORY, lw=1.8)
        ax.plot(x, [f[1] for f in lock], ls, color=C_LOCK, lw=1.8)
        ax.plot(x, samp, mk, color=po.C_HMC, ms=7.5, mfc="white" if name == "FCN2" else po.C_HMC, mew=1.2,
                mec=po.C_HMC if name == "FCN2" else "white", ls="none")
        print(f"ray {name}: outer well mean |w0| full / locked / sampler: " +
              ", ".join(f"d={d} {f[1]:.3f}/{l[1]:.3f}/{s:.3f}" for d, f, l, s in zip(ds, full, lock, samp)))
    for j, (t, yl) in enumerate((("L$_1$ along the ray", (0, 1.15)), ("L$_3$ along the ray", (0, 1)),
                                 ("outer read-in well along the ray: mean |w$_0$|", None))):
        alpha_axis(axs[0, j], ticks, "d")
        axs[0, j].set_title(t, color=po.INK, loc="left", fontsize=11.5)
        if yl:
            axs[0, j].set_ylim(*yl)
    axs[0, 2].set_ylabel("mean |w$_0$| in the outer well")
    h = [plt.Line2D([], [], color=po.C_THEORY, lw=2, ls="-"), plt.Line2D([], [], color=C_LOCK, lw=2, ls="-"),
         plt.Line2D([], [], color=po.C_HMC, marker="o", ls="none", mec="white", ms=7),
         plt.Line2D([], [], color=po.INK, lw=1.6, ls="--"), plt.Line2D([], [], color=po.C_HMC, marker="s", ls="none",
                                                                       mfc="white", mew=1.2, ms=7)]
    axs[0, 1].legend(h, ["full theory", "locked theory", "FCN3 sampler (one point per run)", "FCN2 (dashed lines)",
                         "FCN2 sampler"], fontsize=8.5, loc="lower right")
    # ---- onset (FCN3, d = 150, kappa = 0.015)
    th = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    runs = po.sampler_runs()
    po.set_outer(runs, th)
    groups = po.by_P(runs)
    Pth = sorted(int(k[3:]) for k in th if k.startswith("sc_") and np.isfinite(th[k]).all())
    ath = [alpha(P, po.D) for P in Pth]
    for j, (key, it) in enumerate((("L1", 0), ("L3", 1))):
        ax = axs[1, j]
        ax.plot(ath, [th[f"sc_{P}"][it] for P in Pth], "-", color=po.C_THEORY, lw=2)
        ax.plot(ath, [f3l[f"onset_{P}"][key] for P in Pth], "-", color=C_LOCK, lw=2)
        for P, rs in groups.items():
            m, e = po.combine([r[key] for r in po.per_dataset(rs)])
            ax.errorbar([alpha(P, po.D)], [m], yerr=[e] if np.isfinite(e) else None, fmt="o", ms=7,
                        color=po.C_HMC, mec="white", capsize=0, zorder=4)
    ax = axs[1, 2]
    full, lock = {}, {}
    for P in Pth:
        w, dn = th[f"w0_{P}"], th[f"dens_{P}"]
        full[P] = outer_well(*sym(w, np.where(dn > 1e-300, -np.log(0.5 * np.maximum(dn, 1e-300)), np.nan), 1.6))
        lock[P] = outer_well(*sym(f3l[f"onset_{P}"]["w"], f3l[f"onset_{P}"]["nlp"], 1.6))
    ax.plot(ath, [full[P][1] for P in Pth], "-", color=po.C_THEORY, lw=2)
    ax.plot(ath, [lock[P][1] for P in Pth], "-", color=C_LOCK, lw=2)
    for P, rs in groups.items():
        wb = full[P][0] if P in full else np.nan
        per = {}
        for r in rs:
            if r["w0"] is not None:
                per.setdefault(r["seed"], []).append(sampler_outer(r["w0"], wb))
        v = [np.nanmean(x) for x in per.values()]
        v = [x for x in v if np.isfinite(x)]
        if v:
            e = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None
            ax.errorbar([alpha(P, po.D)], [np.mean(v)], yerr=[e] if e is not None else None, fmt="o", ms=7,
                        color=po.C_HMC, mec="white", capsize=0, zorder=4)
        print(f"onset P={P}: outer well mean |w0| full {full.get(P, (0, np.nan))[1]:.3f} locked "
              f"{lock.get(P, (0, np.nan))[1]:.3f} sampler {np.mean(v) if v else np.nan:.3f}")
    for j, (t, yl) in enumerate((("L$_1$, P-sweep", (0, 1.15)), ("L$_3$, P-sweep", (0, 1)),
                                 ("outer read-in well, P-sweep: mean |w$_0$|", None))):
        alpha_axis(axs[1, j], [(alpha(P, po.D), P) for P in groups], "P")
        axs[1, j].set_title(t, color=po.INK, loc="left", fontsize=11.5)
        if yl:
            axs[1, j].set_ylim(*yl)
    axs[1, 2].set_ylabel("mean |w$_0$| in the outer well")
    box = dict(boxstyle="round,pad=0.5", fc="white", ec=po.GRID)
    axs[0, 0].text(0.03, 0.05,
                   "Ray parameters (scaled with d):\n"
                   "  P = 3000 (d/150)$^2$,  κ = 0.015 d/150,  σ$_A^2$ = 0.8/d\n"
                   "  d = 75, 150, 300, 600 (FCN3);  also 800, 1200 (FCN2)\n"
                   "Fixed:  N = N$_1$ = N$_2$ = 1500,  χ = 1,  ε = −0.3√6,\n"
                   "  σ$_w^2$ = 0.5,  σ$_B$ ≈ 1.255 (FCN3)",
                   transform=axs[0, 0].transAxes, fontsize=9, color=po.INK, va="bottom", bbox=box)
    seeds = sorted({int(r["seed"]) for r in runs})
    axs[1, 0].text(0.03, 0.05,
                   f"P-sweep parameters (FCN3):\n"
                   f"  P = {', '.join(str(P) for P in groups)}\n"
                   f"  d = {po.D},  κ = {po.KAPPA},  N$_1$ = N$_2$ = {po.N},  χ = 1\n"
                   f"  ε = −0.3√6,  σ$_A^2$ = 8/1500,  σ$_B$ = {po.SIGB:.4f},  σ$_w^2$ = {po.SIGW2}\n"
                   f"  datasets (seeds) {', '.join(map(str, seeds))}",
                   transform=axs[1, 0].transAxes, fontsize=9, color=po.INK, va="bottom", bbox=box)
    axs[1, 1].legend(h[:3], ["full theory", "locked theory", "sampler (mean over datasets)"], fontsize=8.5,
                     loc="lower right")
    for ax in axs[:, 2]:
        ax.text(0.02, 0.02, "neurons beyond the barrier, |w$_0$| < 1.5 (sampler: the full theory's barrier);\n"
                "locked line missing where its law has no outer well",
                transform=ax.transAxes, fontsize=7.5, color=po.MUTED)
    fig.suptitle("Perpendicular directions fluctuating (full theory) vs locked to their prior, against the sampler.\n"
                 "Top: the κ$_0$ = 0.015 ray (P = 3000 (d/150)$^2$, κ = 0.015 d/150). "
                 "Bottom: FCN3 P-sweep at d = 150, κ = 0.015. N = 1500.",
                 color=po.INK, x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    out = OUT / "locked_perp_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f2l, f3l = load_or_solve(a.device, a.recompute)
    bad = [k for k, v in f3l.items() if not v.get("conv", True)]
    if bad:
        print("locked solves not converged:", bad)
    plt = po.style()
    fig_ray(plt, f2l, f3l)
    fig_onset(plt, f3l)
    fig_summary(plt, f2l, f3l)
    # table
    d2, d3, l2 = rr.fcn2_data(), rr.fcn3_data(), rr.fcn2_learn()
    print("\nL3 along the ray: full / locked / sampler")
    for d in sorted(set(l2) | set(RAY3)):
        s = f"d={d:5d}"
        if d in l2:
            s += f"  FCN2 {l2[d][1]:.3f} / {f2l[d]['L3']:.3f} / {np.mean([x[1] for x in l2[d][2]]):.3f}"
        if d in d3:
            s += f"  FCN3 {d3[d]['L3th']:.3f} / {f3l[f'ray_{d}']['L3']:.3f} / {np.mean([x[1] for x in d3[d]['L']]):.3f}"
        print(s)
    th = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    runs = po.sampler_runs()
    po.set_outer(runs, th)
    groups = po.by_P(runs)
    print("\nonset L3: full / locked / sampler (mean over datasets)")
    for P, rs in groups.items():
        m, _ = po.combine([r["L3"] for r in po.per_dataset(rs)])
        print(f"P={P:6d}  {th[f'sc_{P}'][1]:.3f} / {f3l[f'onset_{P}']['L3']:.3f} / {m:.3f}")


if __name__ == "__main__":
    main()
