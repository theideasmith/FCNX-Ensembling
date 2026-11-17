#!/usr/bin/env python3
"""The theory's read-in distribution held at the sampler's outer count, against the sampler's.

In the sampler the number of outer read-in neurons is fixed from burn-in on, and at P >= 24000 it is about half the
theory's. If the theory held at the sampler's count (read-in prior tilted by exp(mu 1[|w0| > cut]), as in
theory_fixed_count.py) gives the sampler's w0 distribution, the count is the only read-in difference; if the wells
still differ (position, width, far well), the read-in equation is off beyond the count.

For each P: mu is taken from the ladder in plots/cache/theory_fixed_count_d150.npz (interpolated to the sampler's mean
count), reached in steps from mu = 0 with warm starts, and corrected by secant steps until the held count matches.
Sampler: every post-burn snapshot of every onset run at that P.

    PYTHONPATH=.:../fcn3_matrix_theory python readin_at_count.py   ->  plots/onset_checks/readin_at_count_d150.png
"""
import gc
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_matrix_solver import DT

HERE = po.HERE
PS = (12000, 24000, 48000)
CACHE = HERE / "plots" / "cache" / "readin_at_count_d150.npz"


def solve_tilted(P, mu, cut, start, device="cuda:0"):
    import fcn3_fixed as F
    Fn = F.BlazingFreeEnergy(po._params(P), device=device, verbose=False)
    Fn.lp0 = Fn.lp0 + mu * (Fn.ri.W >= cut).to(DT)
    o = Fn.solve(theta0=torch.tensor(start["theta"], dtype=DT, device=Fn.dev), M0=float(start["M"]))
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape).detach().cpu().numpy()
    w0 = Fn.ri.W.cpu().numpy()[:, 0]
    mass = pw.sum(1)
    out = dict(mu=mu, w0=w0, dens=mass / np.gradient(w0), n=po.N * mass[w0 >= cut].sum() / mass.sum(),
               L3=float(o["L3"]), theta=np.array(o["theta"]), M=float(o["M"]))
    del Fn
    gc.collect()
    torch.cuda.empty_cache()
    return out


def stats(w, p, cut):
    """outer well (cut <= |w0| <= 1.2): mean |w0| and sd; far well (|w0| > 1.2): neurons per network; inner well sd.
    w, p: grid points and masses, or samples with p = 1."""
    m = p / p.sum()
    o, f, i = (w >= cut) & (w <= 1.2), w > 1.2, w < cut
    mo = (m[o] * w[o]).sum() / m[o].sum()
    return dict(out_mean=mo, out_sd=np.sqrt((m[o] * (w[o] - mo) ** 2).sum() / m[o].sum()), far=po.N * m[f].sum(),
                in_sd=np.sqrt((m[i] * w[i] ** 2).sum() / m[i].sum()))


def main():
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    lad = dict(np.load(HERE / "plots" / "cache" / "theory_fixed_count_d150.npz"))
    runs = po.sampler_runs()
    po.set_outer(runs, z)
    groups = po.by_P(runs)
    res = {}
    for P in PS:
        cut = po.barrier_cut(z, P)
        rs = [r for r in groups[P] if r["w0"] is not None]
        n_s = np.mean([po.N * r["outer"] for r in rs])
        mus, ns = lad[f"mu_{P}"], lad[f"n_{P}"]
        k = np.argsort(ns)
        mu_t = float(np.interp(n_s, ns[k], mus[k]))
        start = dict(theta=z[f"theta_{P}"], M=float(z[f"M_{P}"]))
        r = None
        def safe(mu, st):
            try:
                return solve_tilted(P, float(mu), cut, st)
            except Exception as ex:
                print(f"   mu {mu:+.1f}: {type(ex).__name__}", flush=True)
                return None
        for mu in np.linspace(0, mu_t, 8)[1:]:                       # walk to mu_t with warm starts
            r1 = safe(mu, start)
            if r1 is None:
                break
            r = start = r1
        for _ in range(8):                                           # bisection-safe secant on the count
            if abs(r["n"] - n_s) < 1.0:
                break
            r2 = safe(r["mu"] + (2.0 if r["n"] < n_s else -2.0), r)
            if r2 is None or r2["n"] == r["n"]:
                break
            slope = (r2["n"] - r["n"]) / (r2["mu"] - r["mu"])
            step = (n_s - r2["n"]) / slope
            for _h in range(6):
                r3 = safe(r2["mu"] + step, r2)
                if r3 is not None:
                    break
                step /= 2
            r = r3 if r3 is not None else r2
        x = np.concatenate([r_["w0"] for r_ in rs])
        lim = max(1.5, 1.1 * np.abs(x).max())
        edges = np.linspace(0, lim, 41)
        c, _ = np.histogram(np.abs(x), edges)
        res[P] = dict(cut=cut, n_s=n_s, held=r, free=dict(w0=z[f"w0_{P}"], dens=z[f"dens_{P}"]), edges=edges, counts=c,
                      L3_s=np.mean([r_["L3"][0] for r_ in rs]))
        ws = 0.5 * (edges[1:] + edges[:-1])
        s_free = stats(z[f"w0_{P}"], z[f"dens_{P}"] * np.gradient(z[f"w0_{P}"]), cut)
        s_held = stats(r["w0"], r["dens"] * np.gradient(r["w0"]), cut)
        s_samp = stats(np.abs(x), np.ones_like(x), cut)
        print(f"\nP = {P}: outer cut |w0| > {cut:.3f}")
        print(f"   {'':22s} {'outer n':>8s} {'outer mean |w0|':>16s} {'outer sd':>9s} {'far (|w0|>1.2)':>15s} {'inner sd':>9s} {'L3':>6s}")
        for lab, s, n, L3 in (("theory, count free", s_free, po.outer_theory(z, P, cut) * po.N, float(z[f"sc_{P}"][1])),
                              (f"theory, held (mu {r['mu']:+.1f})", s_held, r["n"], r["L3"]),
                              ("sampler", s_samp, n_s, res[P]["L3_s"])):
            print(f"   {lab:22s} {n:8.1f} {s['out_mean']:16.3f} {s['out_sd']:9.3f} {s['far']:15.2f} {s['in_sd']:9.3f} {L3:6.3f}",
                  flush=True)
    np.savez(CACHE, **{f"{k}_{P}": v for P, d in res.items() for k, v in
                       dict(w0h=d["held"]["w0"], dh=d["held"]["dens"], nh=d["held"]["n"], muh=d["held"]["mu"],
                            edges=d["edges"], counts=d["counts"], n_s=d["n_s"]).items()})

    plt = po.style()
    fig, axs = plt.subplots(1, len(PS), figsize=(5.2 * len(PS), 4.6), squeeze=False)
    for ax, P in zip(axs[0], PS):
        d = res[P]
        lo = np.inf
        for w, dn, col, ls, lab in ((d["free"]["w0"], d["free"]["dens"], po.MUTED, "--",
                                     f"theory, count free ({po.outer_theory(z, P, d['cut']) * po.N:.0f} outer)"),
                                    (d["held"]["w0"], d["held"]["dens"], po.C_THEORY, "-",
                                     f"theory held at the sampler's count ({d['held']['n']:.0f})")):
            k = (dn > 1e-300) & (w > 0)
            y = -np.log(0.5 * dn[k])
            ax.plot(w[k], y, ls, color=col, lw=2, label=lab, zorder=2)
            lo = min(lo, y.min())
        e, c = d["edges"], d["counts"]
        mid = 0.5 * (e[1:] + e[:-1])
        k = c > 0
        y = -np.log(c[k] / (c.sum() * np.diff(e)[k]) / 2)
        ax.errorbar(mid[k], y, yerr=1 / np.sqrt(c[k]), fmt="o", ms=3.5, color=po.C_HMC, mec="white", mew=0.3,
                    elinewidth=0.7, label=f"sampler ({d['n_s']:.0f} outer)", zorder=3)
        lo = min(lo, y.min())
        ax.axvline(d["cut"], color=po.MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, e[-1])
        ax.set_ylim(lo - 0.5, lo + 14)
        ax.set_xlabel("|w$_0$|")
        ax.set_title(f"P = {P}", color=po.INK, loc="left", fontsize=12)
        ax.legend(fontsize=8.5, loc="upper left")
    axs[0, 0].set_ylabel("−log p(w$_0$)")
    fig.suptitle("Read-in layer: the theory held at the sampler's number of outer neurons, against the sampler "
                 "(d = 150, N = 1500, κ = 0.015; dotted: barrier used for the count)", color=po.INK, x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "readin_at_count_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
