#!/usr/bin/env python3
"""Is the second hidden layer's preactivation Gaussian?  Its He1 and He3 components, theory against the sampler.

For hidden neuron k, h_k(x) = sum_j B_kj erf(w_j . x) has components h_n = <phi_n | h_k>_x, phi_n = He_n(x0)/sqrt(n!),
    h_n,k = sum_j B_kj c_n(w0_j, q_j)        (c_n in closed form, fcn3_matrix_solver.readin_features).
Sampler: every hidden neuron of the final state of every free run on the kappa0 = 0.015 ray (runs whose number of
outer read-in neurons was set by hand are left out).  Theory: the marginals of p(h, q_h) in (E2) at the solution of
(E1)-(E8), n_max = 11, from the solver's importance-weighted hidden-field samples.  Dashed: a Gaussian with the
sampler's variance.

    python plot_hidden_fields.py [--replot]     ->  plots/ray/hidden_fields_ray.png   (--replot: reuse the cache)
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
from fcn3_fixed import BlazingFreeEnergy, scaled_cholesky  # noqa: E402
from fcn3_matrix_solver import DT, Params, readin_features  # noqa: E402

DS = (150, 300, 600)
SIGB = {150: 1.25540121081682, 300: 1.254358125197598, 600: 1.2538362420680749}
SIGW2, EPS = 0.5, -0.3 * math.sqrt(6)
C_THEORY, C_HMC, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b66", "#e4e3df"


def params(d, nmax):
    r = d / 150
    return Params(d=d, P=3000.0 * r * r, N1=1500, N2=1500, chi=1.0, kappa=0.015 * r, eps=EPS,
                  sigA=math.sqrt((8 / 1500) / r), sigB=SIGB[d], sigw=math.sqrt(SIGW2), nmax=nmax)


def theory_samples(d, dev):
    """(h1, h3, weight) from the solver's fixed hidden-field samples at the n_max = 11 equilibrium."""
    F3 = BlazingFreeEnergy(params(d, 3), device=dev)
    o3 = F3.solve()
    p = params(d, 11)
    F = BlazingFreeEnergy(p, device=dev)
    th = torch.zeros(F.nth, dtype=DT, device=F.dev)
    th[:3] = F3.theta_ref[:3].to(F.dev)
    th[-1] = F3.theta_ref[-1].to(F.dev)
    o = F.solve(theta0=th, M0=o3["M"])
    J, T, M = F.last["J"], F.last["T"], o["M"]
    with torch.no_grad():
        e, h, q = F.mc.features(scaled_cholesky(p.sigB ** 2 * J), p.sigB ** 2 * T)
        lt = F.mc.logw(M) + 0.5 * p.gt * (e @ F.t) ** 2
        w = torch.exp(lt - torch.logsumexp(lt, 0))
    print(f"d={d}: theory L3={o['L3']:.4f}, ESS {float(1 / (w * w).sum()):.0f}/{len(w)}")
    return h[:, 0].cpu().numpy(), h[:, 1].cpu().numpy(), w.cpu().numpy()


def sampler_fields(d):
    """(h1, h3) of every hidden neuron in the final state of every free run at this d."""
    P = int(3000 * (d / 150) ** 2)
    H1, H3, n = [], [], 0
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_d{d}_P{P}_*" / "config.json"))):
        r = Path(f).parent
        c = json.loads(Path(f).read_text())["args"]
        if ("aborted" in str(r) or c.get("tag") != "ray" or str(c.get("group", "")).startswith("fixedn")
                or not (r / "checkpoint.pt").exists()):
            continue
        ck = torch.load(r / "checkpoint.pt", map_location="cpu", weights_only=False)
        ZW, ZB = ck["ZW"].double(), ck["ZB"].double()
        sw, sb = math.sqrt(SIGW2 / d), c["sigB"] / math.sqrt(c["N1"])
        w0 = sw * ZW[:, 0]
        q = sw * sw * (ZW[:, 1:] ** 2).sum(1)
        cf = readin_features(w0, q, (1, 3))                    # (2, N1)
        h = (sb * ZB) @ cf.T                                   # (N2, 2)
        H1.append(h[:, 0].numpy())
        H3.append(h[:, 1].numpy())
        n += 1
    return np.concatenate(H1), np.concatenate(H3), n


def nlogp_points(x, edges):
    cnt, _ = np.histogram(x, edges)
    k = cnt > 0
    mid = 0.5 * (edges[1:] + edges[:-1])
    dens = cnt / (cnt.sum() * np.diff(edges))
    return mid[k], -np.log(dens[k]), 1 / np.sqrt(cnt[k])


def nlogp_weighted(x, w, edges, min_ess=30):
    s, _ = np.histogram(x, edges, weights=w)
    s2, _ = np.histogram(x, edges, weights=w * w)
    k = (s > 1e-14) & (s * s > min_ess * s2)               # drop bins resting on a handful of importance samples
    mid = 0.5 * (edges[1:] + edges[:-1])
    return mid[k], -np.log(s[k] / (s.sum() * np.diff(edges)[k]))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    cpath = HERE / "plots" / "cache" / "cache_hidden_fields.npz"
    old = np.load(cpath) if ("--replot" in sys.argv and cpath.exists()) else None
    cache = {}
    for i, d in enumerate(DS):
        if old is not None:
            t1, t3, tw, s1, s3 = (old[f"{k}_{d}"] for k in ("t1", "t3", "tw", "s1", "s3"))
            nrun = int(old[f"nrun_{d}"])
        else:
            t1, t3, tw = theory_samples(d, dev)
            s1, s3, nrun = sampler_fields(d)
        cache[d] = dict(t1=t1, t3=t3, tw=tw, s1=s1, s3=s3, nrun=np.array(nrun))
        for row, (s, t, name) in enumerate([(s1, t1, "He$_1$"), (s3, t3, "He$_3$")]):
            ax = axs[row, i]
            sd = s.std()
            lim = 4.5 * max(sd, math.sqrt(np.sum(tw * t * t)))
            edges = np.linspace(-lim, lim, 61)
            kurt = np.mean((s / sd) ** 4) - 3
            tm2 = np.sum(tw * t * t)
            tkurt = np.sum(tw * t ** 4) / tm2 ** 2 - 3
            xm, ym = nlogp_weighted(t, tw, edges)
            ax.plot(xm, ym, color=C_THEORY, lw=2.2, label=f"theory: excess kurtosis {tkurt:+.2f}", zorder=2)
            xs, ys, es = nlogp_points(s, edges)
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, color=C_HMC, mec="white", mew=0.5, elinewidth=0.9,
                        capsize=0, label=f"sampler, {len(s)} hidden neurons in {nrun} networks: {kurt:+.2f}",
                        zorder=3)
            g = np.linspace(-lim, lim, 400)
            ax.plot(g, 0.5 * (g / sd) ** 2 + math.log(sd * math.sqrt(2 * math.pi)), "--", color=MUTED, lw=1.2,
                    label="Gaussian with the sampler's variance: 0", zorder=1)
            ax.set_xlim(-lim, lim)
            lo = min(ym.min(), ys.min())
            ax.set_ylim(lo - 0.5, lo + 9)
            ax.set_xlabel(f"h | {name}")
            if i == 0:
                ax.set_ylabel(f"−log p(h | {name})")
            ax.set_title(f"d = {d}, {name} component", color=INK, loc="left", fontsize=11)
            ax.legend(fontsize=7.5, loc="upper center")
            print(f"  {name}: sampler std {sd:.4f} kurt {kurt:+.3f} | theory std {math.sqrt(tm2):.4f} kurt {tkurt:+.3f}")
    fig.suptitle("Second hidden layer: He$_1$ and He$_3$ components of each neuron's preactivation, "
                 "κ$_0$ = 0.015 ray", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / "hidden_fields_ray.png"
    fig.savefig(out, dpi=150)
    np.savez(cpath,
             **{f"{k}_{d}": v for d, c in cache.items() for k, v in c.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
