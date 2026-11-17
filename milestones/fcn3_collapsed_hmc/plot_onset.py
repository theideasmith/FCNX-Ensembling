#!/usr/bin/env python3
"""Onset of learnability in the FCN3 as P grows at fixed kappa = 0.015, d = 150 (queue_onset.sh): theory against the
exact posterior sampler.

Runs: group onset_d150 (P = 750 ... 48000, datasets 0-2, prior start, full networks in nets/) and, for P = 3000, also
the prior-start ray runs at d = 150 (groups ray_k0p015_d150 and ray_hist_d150, datasets 0-3), which have the same
parameters; runs on the same dataset are averaged first, so each dataset counts once. Finished runs use
summary.json; runs still going use the post-burn rows of obs.csv (open markers).
Theory: (E1)-(E8) at n_max = 11 at each P, the largest-W solution over the default start and continuation from the
neighbouring P in both directions (see theory_all), cached in plots/cache/theory_onset_d150_v6.npz.

Figure 1  plots/onset/onset_d150_learnability.png: L3, L1 and the number of read-in neurons in the outer well against P; the
          outer well is |w0| beyond the top of the theory's barrier at that P (the well moves in as P grows).
          Small points: one per dataset. Large points: mean over datasets with its standard error.
Figure 2  plots/onset/onset_d150_actions.png: one column per P; rows: -log p of the read-in w0 and of the He1 and He3
          components of the second hidden layer's preactivation. Theory: lines (read-in grid; hidden-layer marginals of
          (E2), hidden_marginals). Sampler: points (read-in: every post-burn snapshot of every dataset; hidden layer:
          every hidden neuron of every saved network, nets/ every 50 iterations after burn-in). Poisson error bars.

    python plot_onset.py [--recompute] [--no-theory] [--device cuda:0]
"""
import argparse
import csv
import gc
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
THEORY = HERE.parent / "fcn3_matrix_theory"
sys.path.insert(0, str(THEORY))
D, N, KAPPA = 150, 1500, 0.015
SIGB, SIGW2, SIGA2 = 1.25540121081682, 0.5, 8 / 1500
EPS = -0.3 * math.sqrt(6)
W_OUT = 0.6
P_EXTRA = (500, 1000, 2000, 4000, 8000, 16000, 32000)      # extra theory points for a smooth curve
C_THEORY, C_HMC, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b66", "#e4e3df"


# ---------------------------------------------------------------------------------------------------- theory
def _params(P, nmax=11):
    from fcn3_matrix_solver import Params
    return Params(d=D, P=float(P), N1=N, N2=N, chi=1.0, kappa=KAPPA, eps=EPS, sigA=math.sqrt(SIGA2), sigB=SIGB,
                  sigw=math.sqrt(SIGW2), nmax=nmax)


def _solve(P, device, start=None):
    """One n_max = 11 solve. start=None: the default path of solve_fcn3 (n_max = 3 from the prior, then 11); otherwise
    warm-started from another solution's read-in exponents theta and M. Returns what the plots need."""
    import fcn3_fixed as F
    from fcn3_matrix_solver import DT
    if start is None:
        F3 = F.BlazingFreeEnergy(_params(P, 3), device=device)
        o3 = F3.solve()
        Fn = F.BlazingFreeEnergy(_params(P), device=device)
        th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
        th[:3] = F3.theta_ref[:3].to(Fn.dev)
        th[-1] = F3.theta_ref[-1].to(Fn.dev)
        o = Fn.solve(theta0=th, M0=o3["M"])
        del F3
    else:
        Fn = F.BlazingFreeEnergy(_params(P), device=device)
        o = Fn.solve(theta0=torch.tensor(start["theta"], dtype=DT, device=Fn.dev), M0=float(start["M"]))
    p = Fn.p
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]                    # grid: |w0| along axis 0, q along axis 1
    mass = pw.sum(1)
    r = dict(W=float(o["W"]), L1=float(o["L1"]), L3=float(o["L3"]), theta=np.array(o["theta"]), M=float(o["M"]),
             w0=w0, dens=mass / np.gradient(w0), outer=float(mass[w0 >= W_OUT].sum() / mass.sum()),
             G=(p.sigB ** 2 * Fn.last["J"]).detach().cpu().numpy(), qbar=float(p.sigB ** 2 * Fn.last["T"]),
             t=Fn.t.detach().cpu().numpy(), gt=float(p.gt), ks=np.array(p.ks), H=np.array(o["H"]))
    with torch.no_grad():                                # hidden-field particles: systematic resampling, 2^14
        import fcn3_fixed as F
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(o["M"]) + 0.5 * p.gt * (e @ Fn.t) ** 2, 0)
        n = 2 ** 14
        u = (torch.rand(1, dtype=w.dtype, device=w.device, generator=torch.Generator(w.device).manual_seed(3))
             + torch.arange(n, dtype=w.dtype, device=w.device)) / n
        idx = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)
        r["hs"], r["qs"] = h[idx].cpu().numpy(), q[idx].cpu().numpy()
        r["ess"] = float(1 / (w * w).sum())
    del Fn
    return r


def _try_solve(P, device, start=None):
    try:
        return _solve(P, device, start)
    except Exception as e:
        print(f"  theory P={P} ({'default' if start is None else 'warm'}) failed: {e!r}"[:300])
        return None
    finally:
        gc.collect()                                     # the solver's autograd graphs hold GPU memory until collected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def hidden_marginals(r, device, h1max=3.2, n1=1281, n3=1001, n_part=4096, chunk=64, h3max=None):
    """Marginal densities of h1 = h|He1 and h3 = h|He3 in (E2) at a solution, smooth and exact on average.
    Write the prior h ~ N(0, G) as h = L z with z standard normal. For component a, moving z along the direction that
    changes only h_a, with the orthogonal part z_perp and q_h fixed, moves the whole vector along the prior regression
    line h(x) = h_i + G[:, a] / G_aa (x - h_i,a), and the conditional density of h_a = x given (z_perp, q_h) is
    N(x; 0, G_aa) exp(g~/2 (t.e(h(x), q_h))^2), normalised on a fine grid. The marginal is its average over particles
    (z_perp, q_h) resampled from the solver's importance-weighted samples (r["hs"], r["qs"]; 4096 of 2^14).
    (G is nearly singular, cond ~ 1e11: conditioning on the other components instead pins h3 to within ~3e-4, far
    below any grid; averaging over the prior of the other components fails at large P.) h3 grid: h3max, by default 12
    posterior standard deviations, between 0.2 and 0.7."""
    from numpy.polynomial.hermite_e import hermegauss
    from fcn3_matrix_solver import hermite_norm
    dev, dt = torch.device(device), torch.float64
    G = torch.tensor(r["G"], dtype=dt, device=dev)
    t = torch.tensor(r["t"], dtype=dt, device=dev)
    stride = max(1, len(r["hs"]) // n_part)
    hs = torch.tensor(np.asarray(r["hs"])[::stride], dtype=dt, device=dev)
    sq = 1 / torch.sqrt(1 + 2 * torch.tensor(np.asarray(r["qs"])[::stride], dtype=dt, device=dev))
    x, wx = hermegauss(64)
    Phi = hermite_norm(tuple(int(k) for k in r["ks"]), torch.tensor(x, dtype=dt, device=dev))       # (S, nx)
    WPhi = (Phi * torch.tensor(wx / wx.sum(), dtype=dt, device=dev)).T                             # (nx, S)
    if h3max is None:
        h3max = float(np.clip(12 * math.sqrt(r["H"][1][1]), 0.2, 0.7))
    out, worst = [], []
    for a, g in ((0, np.linspace(-h1max, h1max, n1)), (1, np.linspace(-h3max, h3max, n3))):
        gx = torch.tensor(g, dtype=dt, device=dev)
        dx = float(g[1] - g[0])
        slope = G[:, a] / G[a, a]                                              # prior regression of h on h_a
        acc = torch.zeros(len(g), dtype=dt, device=dev)
        mx = []
        with torch.no_grad():
            for c0 in range(0, len(hs), chunk):
                hi, si = hs[c0:c0 + chunk], sq[c0:c0 + chunk]
                h = hi[:, None, :] + (gx[None, :, None] - hi[:, None, a:a + 1]) * slope[None, None, :]  # (c, ng, S)
                tb = (torch.erf((h @ Phi) * si[:, None, None]) @ WPhi) @ t
                lc = -0.5 * gx[None, :] ** 2 / G[a, a] + 0.5 * r["gt"] * tb * tb
                pc = torch.softmax(lc, 1)                                      # per particle, sums to 1 over the grid
                mx.append(pc.max(1).values)
                acc += pc.sum(0) / dx
        dens = (acc / len(hs)).cpu().numpy()
        worst.append(float(torch.quantile(torch.cat(mx), 0.99)))
        out += [g, np.log(np.maximum(dens, 1e-300))]
    if max(worst) > 0.2:
        print(f"  warning: some particle's conditional sits in one grid cell (99th pct of max cell mass {worst})")
    return out


def barrier_cut(z, P):
    """|w0| at the top of the theory's barrier between the inner and outer read-in wells at this P (the outer well
    moves in from |w0| ~ 1.2 to ~ 0.6 as P grows, so no fixed cut separates them); W_OUT if there is no outer well."""
    if f"dens_{P}" not in z or not len(z[f"dens_{P}"]):
        return W_OUT
    w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
    out = (w > 0.4) & (w < 2.0)
    if not out.any():
        return W_OUT
    i_out = np.flatnonzero(out)[np.argmax(dn[out])]                 # bottom of the outer well
    between = np.flatnonzero((w > 0.2) & (w < w[i_out]))
    if len(between) == 0:
        return W_OUT
    i_top = between[np.argmin(dn[between])]
    if dn[i_top] > 0.5 * dn[i_out]:                                 # no barrier: a shoulder, not a second well
        return W_OUT
    return float(w[i_top])


def outer_theory(z, P, cut):
    w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
    mass = dn * np.gradient(w)
    return float(mass[w >= cut].sum() / mass.sum())


def theory_all(Ps, device, recompute):
    """Theory at every P. Near P ~ 500-750 two solutions of (E1)-(E8) coexist (theory_phase_check.py: a far group of
    hidden neurons, or an outer read-in well), and the default start can land on the wrong one. So every P is solved
    from the default start and by continuation from its neighbours in both directions (warm start from the best
    solution at the previous P going up, and at the next P going down), and the largest W is kept: the equilibrium is
    the global maximum of W. Then the hidden-layer marginals are computed on grids. Cached in
    plots/cache/theory_onset_d150_v6.npz (recomputed whenever a P is missing), with the solution itself (G, qbar, M, t, ...)
    and 2^14 hidden-field particles."""
    cache = HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"
    z = {} if recompute or not cache.exists() else dict(np.load(cache))
    if all(f"W_{P}" in z for P in Ps):
        return z
    Ps = sorted(Ps)
    best, how = {}, {}

    def keep(P, r, label):
        if r is None:
            return
        if P not in best or r["W"] > best[P]["W"] + 1e-6:
            if P in best:
                print(f"  P={P}: {label} beats {how[P]} by {r['W'] - best[P]['W']:+.3f} in W "
                      f"(outer read-in {N * best[P]['outer']:.1f} -> {N * r['outer']:.1f})")
            best[P], how[P] = r, label

    for P in Ps:
        keep(P, _try_solve(P, device), "default start")
    for seq in (Ps, Ps[::-1]):                           # continuation up in P, then down
        for a, b in zip(seq[:-1], seq[1:]):
            if a in best:
                keep(b, _try_solve(b, device, best[a]), f"start from P={a}")
    z = {}
    for P, r in best.items():
        h1g, lp1, h3g, lp3 = hidden_marginals(r, device)
        z.update({f"W_{P}": np.array(r["W"]), f"sc_{P}": np.array([r["L1"], r["L3"], r["outer"]]),
                  f"w0_{P}": r["w0"], f"dens_{P}": r["dens"], f"h1g_{P}": h1g, f"lp1_{P}": lp1, f"h3g_{P}": h3g,
                  f"lp3_{P}": lp3})
        for k in ("G", "qbar", "M", "t", "gt", "ks", "H", "theta", "hs", "qs", "ess"):   # for later diagnostics
            z[f"{k}_{P}"] = np.asarray(r[k])
        print(f"theory P={P}: L1={r['L1']:.4f} L3={r['L3']:.4f} outer read-in={N * r['outer']:.1f} W={r['W']:.3f} "
              f"({how[P]})")
    np.savez(cache, **z)
    return z


# ---------------------------------------------------------------------------------------------------- sampler
def mean_err(x, nb=10):
    """Mean and batch-means standard error."""
    x = np.asarray(x, float)
    if len(x) < 2 * nb:
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    bm = np.array([b.mean() for b in np.array_split(x, nb)])
    return float(x.mean()), float(bm.std(ddof=1) / math.sqrt(nb))


def wanted(cfg):
    if cfg.get("d") != D or cfg.get("N1") != N or cfg.get("N2") != N or abs(cfg.get("kappa", 0) - KAPPA) > 1e-12:
        return False
    if cfg.get("group") == "onset_d150":
        return True
    return (cfg.get("P") == 3000 and cfg.get("tag") == "ray" and cfg.get("init") == "prior"
            and cfg.get("group") in ("ray_k0p015_d150", "ray_hist_d150"))


def sampler_runs(min_rows=8):
    out = []
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_d{D}_P*_N{N}x{N}_*" / "config.json"))):
        run = Path(f).parent
        if "aborted" in str(run):
            continue
        cfg = json.loads(Path(f).read_text())["args"]
        if not wanted(cfg):
            continue
        done = (run / "summary.json").exists()
        if done:
            s = json.loads((run / "summary.json").read_text())["posterior"]
            L1, L3 = tuple(s["L1"]), tuple(s["L3"])
            last = cfg["iters"]
        else:
            if not (run / "obs.csv").exists():
                continue
            rows = [r for r in csv.DictReader(open(run / "obs.csv")) if r["burn"] == "0"]
            if len(rows) < min_rows:
                continue
            L1 = mean_err([float(r["fbar1"]) for r in rows])
            L3 = mean_err([float(r["fbar3"]) / cfg["eps"] for r in rows])
            last = int(rows[-1]["iter"])
        w0 = None
        if (run / "snaps.npz").exists():
            sn = np.load(run / "snaps.npz")
            post = sn["iter"] >= cfg["burn"]
            if post.sum() > 0:
                w0 = sn["w0"][post].ravel().astype(float)
        if w0 is None and (run / "checkpoint.pt").exists():
            ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
            w0 = (ck["ZW"][:, 0].double() * math.sqrt(SIGW2 / D)).numpy()
        out.append(dict(P=int(cfg["P"]), seed=int(cfg["seed"]), done=done, last=last, L1=L1, L3=L3, w0=w0,
                        run=run.name, path=run, cfg=cfg))
    return out


def sampler_hidden(rs):
    """He1 and He3 components h_n,k = sum_j B_kj c_n(w0_j, q_j) of every hidden neuron in every saved network
    (nets/net_*.pt, every 50 iterations after burn-in) of these runs."""
    from fcn3_matrix_solver import readin_features
    H, n = [], 0
    for r in rs:
        for f in sorted((r["path"] / "nets").glob("net_*.pt")):
            nt = torch.load(f, map_location="cpu", weights_only=False)
            ZW, ZB = nt["ZW"].double(), nt["ZB"].double()
            w0 = nt["sw"] * ZW[:, 0]
            q = nt["sw"] ** 2 * (ZW[:, 1:] ** 2).sum(1)
            H.append(((nt["sb"] * ZB) @ readin_features(w0, q, (1, 3)).T).numpy())
            n += 1
    return (np.concatenate(H), n) if H else (np.zeros((0, 2)), 0)


def set_outer(runs, z):
    """Fraction of read-in neurons beyond the theory's barrier at the run's P (the same cut for theory and sampler)."""
    for r in runs:
        cut = barrier_cut(z, r["P"])
        r["cut"] = cut
        r["outer"] = float(np.mean(np.abs(r["w0"]) >= cut)) if r["w0"] is not None else float("nan")


def by_P(runs):
    g = {}
    for r in runs:
        g.setdefault(r["P"], []).append(r)
    return dict(sorted(g.items()))


def per_dataset(rs):
    """One entry per dataset: runs on the same dataset (the onset run and the earlier ray run at P = 3000) are
    averaged, so that they count once."""
    out = []
    for seed in sorted({r["seed"] for r in rs}):
        g = [r for r in rs if r["seed"] == seed]
        ent = dict(done=all(r["done"] for r in g))
        for k in ("L1", "L3"):
            ent[k] = (float(np.mean([r[k][0] for r in g])), float(np.sqrt(sum(r[k][1] ** 2 for r in g)) / len(g)))
        o = [r["outer"] for r in g if np.isfinite(r["outer"])]
        ent["outer"] = float(np.mean(o)) if o else float("nan")
        out.append(ent)
    return out


def combine(vals):
    """Mean over datasets and its standard error (a single dataset: its own error)."""
    m = np.array([v[0] for v in vals])
    if len(m) == 1:
        return float(m[0]), float(vals[0][1])
    return float(m.mean()), float(m.std(ddof=1) / math.sqrt(len(m)))


# ---------------------------------------------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    return plt


def fig_learnability(plt, groups, z, out):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9))
    panels = [("L3", 1, "L$_3$", (0, 1.05)), ("L1", 0, "L$_1$", (0, 1.5)),
              ("outer", 2, "read-in neurons in the outer well (of 1500)", None)]
    Pth = sorted(int(k[3:]) for k in z if k.startswith("sc_") and np.isfinite(z[k]).all())
    running = not all(r["done"] for rs in groups.values() for r in rs)
    for ax, (key, it, lab, ylim) in zip(axs, panels):
        scale = N if key == "outer" else 1
        if Pth:
            th = [outer_theory(z, P, barrier_cut(z, P)) if key == "outer" else z[f"sc_{P}"][it] for P in Pth]
            ax.plot(Pth, [scale * v for v in th], "-", color=C_THEORY, lw=2.2, label="theory", zorder=2)
        first = True
        for P, rs in groups.items():
            rs = per_dataset(rs)
            vals = [(r[key], 0.0) if key == "outer" else r[key] for r in rs]
            if key == "outer":
                vals = [v for v in vals if np.isfinite(v[0])]
                if not vals:
                    continue
            xs = P * np.exp(np.linspace(-0.05, 0.05, len(vals)))
            ax.plot(xs, [scale * v[0] for v in vals], "o", ms=4, color=C_HMC, alpha=0.45, mec="none", zorder=3,
                    label="sampler, one dataset" if first else None)
            m, e = combine(vals)
            alldone = all(r["done"] for r in rs)
            ax.errorbar([P], [scale * m], yerr=[scale * e] if np.isfinite(e) else None, fmt="o", ms=8,
                        color=C_HMC, mfc=C_HMC if alldone else "white", mec=C_HMC, mew=1.4, elinewidth=1.2,
                        capsize=0, zorder=4,
                        label=("sampler, mean over datasets" + (" (open: still running)" if running else ""))
                        if first else None)
            first = False
        ax.set_xscale("log")
        ticks = sorted(set(groups) | ({3000} if not groups else set()))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(P) for P in ticks], fontsize=8.5)
        ax.minorticks_off()
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("training-set size P")
        ax.set_title(lab, color=INK, loc="left", fontsize=11)
    axs[0].legend(loc="upper left", fontsize=8.5)
    axs[2].text(0.02, 0.97, "outer well: |w$_0$| beyond the top of the theory's barrier at that P",
                transform=axs[2].transAxes, fontsize=8, color=MUTED, va="top")
    fig.suptitle(f"FCN3 onset of learnability: d = {D}, N$_1$ = N$_2$ = {N}, κ = {KAPPA} fixed, prior start",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def fig_per_P(plt, groups, z, out):
    """One column per P; rows: -log p of the read-in w0, of h|He1 and of h|He3 (second hidden layer)."""
    Ps = list(groups)
    hid = {P: sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()]) for P in Ps}
    fig, axs = plt.subplots(4, len(Ps), figsize=(3.6 * len(Ps), 13.6), squeeze=False)
    rows = [("w$_0$", "read-in layer, full height"), ("w$_0$", "read-in layer, near the wells"),
            ("h | He$_1$", "hidden layer, He$_1$"), ("h | He$_3$", "hidden layer, He$_3$")]
    for c, P in enumerate(Ps):
        for i, (xl, name) in enumerate(rows):
            ax = axs[i, c]
            lo, tops, reach = np.inf, [], None
            if i <= 1:
                x = np.concatenate([r["w0"] for r in groups[P] if r["w0"] is not None])
                lim = max(1.5, 1.1 * np.abs(x).max())
                if f"dens_{P}" in z and len(z[f"dens_{P}"]):
                    w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
                    k = dn > 1e-300
                    nl = -np.log(0.5 * dn[k])
                    mins = [w[k][j] for j in range(1, k.sum() - 1) if nl[j] < nl[j - 1] and nl[j] < nl[j + 1]
                            and nl[j] < nl.min() + 12]               # every theory well within 12 of the deepest
                    lim = max([lim] + [1.15 * m for m in mins])
                    k = (dn > 1e-300) & (w > 0) & (w <= lim)
                    xs = np.concatenate([-w[k][::-1], w[k]])
                    ys = -np.log(0.5 * np.concatenate([dn[k][::-1], dn[k]]))
                    ax.plot(xs, ys, color=C_THEORY, lw=2, zorder=2)
                    lo = min(lo, ys.min())
                    tops = [ys[m] for m in range(1, len(ys) - 1) if ys[m] > ys[m - 1] and ys[m] > ys[m + 1]]
                nb, top = 61, 13
            else:
                j = 2 * (i - 1) - 1
                x = hid[P][0][:, i - 2]
                m = [np.abs(x).max()] if len(x) else []
                if f"lp{j}_{P}" in z:
                    g, yt = z[f"h{j}g_{P}"], -z[f"lp{j}_{P}"]
                    m.append(np.abs(g[yt <= yt.min() + 9]).max())
                lim = 1.08 * max(m)
                if f"lp{j}_{P}" in z:
                    k = np.abs(g) <= lim
                    ax.plot(g[k], yt[k], color=C_THEORY, lw=2, zorder=2)
                    lo = min(lo, yt.min())
                    if i == 2 and len(x):                # He1: show the theory out to the sampler's widest neuron
                        xm = np.abs(x).max()
                        reach = max(np.interp(xm, g, yt), np.interp(-xm, g, yt))
                nb, top = 81, 10
            if len(x):
                edges = np.linspace(-lim, lim, nb)
                mid = 0.5 * (edges[1:] + edges[:-1])
                cnt, _ = np.histogram(x, edges)
                k = cnt > 0
                y = -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k]))
                ax.errorbar(mid[k], y, yerr=1 / np.sqrt(cnt[k]), xerr=0.5 * np.diff(edges)[k], fmt="o", ms=2.8,
                            color=C_HMC, mec="white", mew=0.3, elinewidth=0.7, capsize=0, zorder=3)
                lo = min(lo, y.min())
            ax.set_xlim(-lim, lim)
            if i == 0 and tops:                          # full height: up to the tallest theory barrier
                top = max(top, max(tops) - lo + 3)
            if reach is not None:
                top = max(top, reach - lo + 2)
            ax.set_ylim(lo - 0.5, lo + top)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"−log p   ({name})")
            if i == 0:
                ax.set_title(f"P = {P}", color=INK, loc="left", fontsize=12)
    fig.legend([plt.Line2D([], [], color=C_THEORY, lw=2),
                plt.Line2D([], [], color=C_HMC, marker="o", ls="none", ms=4)],
               ["theory", "sampler (read-in: every post-burn snapshot; hidden layer: every hidden neuron of the "
                "45 networks saved per P; all datasets pooled)"], loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.025))
    fig.suptitle(f"FCN3 actions as P grows: d = {D}, N$_1$ = N$_2$ = {N}, κ = {KAPPA} fixed", color=INK, x=0.01,
                 ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="re-solve the theory at every P")
    ap.add_argument("--no-theory", action="store_true", help="sampler only (no theory solves)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    (HERE / "plots").mkdir(exist_ok=True)
    groups = by_P(sampler_runs())
    for P, rs in groups.items():
        for r in sorted(rs, key=lambda r: r["seed"]):
            st = "done" if r["done"] else f"running, to it {r['last']}"
            print(f"P={P:6d} dataset {r['seed']}: L1={r['L1'][0]:.4f}±{r['L1'][1]:.4f} "
                  f"L3={r['L3'][0]:.4f}±{r['L3'][1]:.4f}  ({st})")
    z = {} if a.no_theory else theory_all(sorted(set(groups) | set(P_EXTRA)), a.device, a.recompute)
    set_outer([r for rs in groups.values() for r in rs], z)
    for P, rs in groups.items():
        o = [N * x["outer"] for x in per_dataset(rs)]
        th = N * outer_theory(z, P, barrier_cut(z, P)) if f"dens_{P}" in z else float("nan")
        print(f"P={P:6d}: cut |w0| = {barrier_cut(z, P):.3f}, outer neurons sampler {np.round(o, 1)} theory {th:.1f}")
    plt = style()
    fig_learnability(plt, groups, z, HERE / "plots" / "onset" / "onset_d150_learnability.png")
    fig_per_P(plt, groups, z, HERE / "plots" / "onset" / "onset_d150_actions.png")


if __name__ == "__main__":
    main()
