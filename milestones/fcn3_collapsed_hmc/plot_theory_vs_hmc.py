#!/usr/bin/env python3
"""
FCN3 along the kappa0 = 0.015 ray (N1 = N2 = 1500): equilibrium theory (fcn3_fixed.solve_fcn3, n_max = 11 and 3)
against the exact posterior sampler (runs/ ray chains, prior and aligned starts).

Top row: read-in action -log p(w0) at d = 75, 150, 300, 600. Theory: the read-in distribution marginalised over q.
Sampler: histogram of |w0| over the N1 read-in neurons of each chain's final state (checkpoint.pt), both starts
pooled, signed (the posterior is symmetric under w -> -w). Bottom row: <w0^2> = E[w0^2], L1, L3 against d.
(summary.json of run_fcn3.py stores d E[w0^2] / sigma_w^2; the theory's d_w0sq is d E[w0^2].)

    python plot_theory_vs_hmc.py            # theory read-in distributions cached in plots/cache/theory_readin_ray.npz
    python plot_theory_vs_hmc.py --recompute
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
THEORY = HERE.parent / "fcn3_matrix_theory"
SIGW2 = 0.5
SIGB = {75: 1.2574845533917596, 150: 1.25540121081682, 300: 1.254358125197598, 600: 1.2538362420680749}
DS = (75, 150, 300, 600)

# categorical slots (dataviz reference palette, light)
C_THEORY, C_HMC, C_N3, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8a85", "#1f1f1e", "#6b6b66", "#e4e3df"
C_N3K = "#1a9e77"          # sampler at N1 = N2 = 3000 (chi = 2: same theory point)


def theory_readin(d, device):
    """n_max = 11 equilibrium at ray point d: (w0 nodes, p(|w0|) density, L1, L3, d<w0^2>)."""
    sys.path.insert(0, str(THEORY))
    from fcn3_matrix_solver import Params
    import fcn3_fixed as F
    orig = F.BlazingFreeEnergy.solve

    def solve(self, *a, **k):
        o = orig(self, *a, **k)
        o["_pw"] = self.last["pw"].reshape(self.ri.W.shape).detach().cpu().numpy()
        o["_W"] = self.ri.W.cpu().numpy()
        return o

    F.BlazingFreeEnergy.solve = solve
    r = d / 150
    p = Params(d=d, P=round(3000 * r * r), N1=1500, N2=1500, chi=1.0, kappa=0.015 * r, eps=-0.3 * math.sqrt(6),
               sigA=math.sqrt((8 / 1500) / r), sigB=SIGB[d], sigw=math.sqrt(SIGW2), nmax=11)
    o = F.solve_fcn3(p, nmax=11, device=device)
    F.BlazingFreeEnergy.solve = orig
    S = len(o["H"])
    print(f"d={d}: solve_fcn3 nmax={o.get('nmax')} (H is {S}x{S}, S = {{1,3,...,{2 * S - 1}}}), "
          f"L1={o['L1']:.4f} L3={o['L3']:.4f} d<w0^2>={o['d_w0sq']:.4f}; its n_max=3 stage: L3={o['nmax3']['L3']:.4f}")
    assert o.get("nmax") == 11 and S == 6, "not the n_max = 11 solution"
    w0 = o["_W"][:, 0]                                 # grid: w0 along axis 0 (|w0| >= 0), q along axis 1
    mass = o["_pw"].sum(1)
    dens = mass / np.gradient(w0)
    return w0, dens, o["L1"], o["L3"], o["d_w0sq"]


def sampler_runs():
    out = {}
    for f in sorted(glob.glob(str(HERE / "runs" / "*_ray_d*_N1500x1500_*" / "summary.json"))):
        run = Path(f).parent
        cfg = json.loads((run / "config.json").read_text())["args"]
        if cfg.get("tag") != "ray" or str(cfg.get("group", "")).startswith(("ray_hist", "fixedn")):
            continue
        s = json.loads(Path(f).read_text())["posterior"]
        if (run / "snaps.npz").exists():          # every post-burn snapshot (run_fcn3.py --snap-every)
            sn = np.load(run / "snaps.npz")
            w0 = sn["w0"][sn["iter"] >= cfg["burn"]].ravel().astype(float)
        else:                                     # final state only
            ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
            w0 = (ck["ZW"][:, 0].double() * math.sqrt(SIGW2 / cfg["d"])).numpy()
        # summary.json stores d E[w0^2] / sigma_w^2
        out[(cfg["d"], cfg["init"])] = dict(L1=s["L1"], L3=s["L3"], w0sq=[v * SIGW2 / cfg["d"] for v in s["d_w0sq"]],
                                            w0=w0)
    return out


def hist_runs(min_rows=8):
    """Snapshot chains on further datasets (queue_hist.sh: group ray_hist_d*, data seeds 1-3, prior start), finished
    or still running: post-burn means of obs.csv (batch-means errors) and every post-burn read-in snapshot."""
    import csv
    out = {}
    for f in sorted(glob.glob(str(HERE / "runs" / "*_ray_d*_N1500x1500_*" / "snaps.npz"))):
        run = Path(f).parent
        cfg = json.loads((run / "config.json").read_text())["args"]
        if not str(cfg.get("group", "")).startswith("ray_hist"):
            continue
        rows = [r for r in csv.DictReader(open(run / "obs.csv")) if r["burn"] == "0"]
        sn = np.load(f)
        post = sn["iter"] >= cfg["burn"]
        if len(rows) < min_rows or not post.any():
            continue

        def mb(x):
            x = np.asarray(x, float)
            nb = min(10, len(x) // 2)
            b = np.array([c.mean() for c in np.array_split(x, nb)])
            return [float(x.mean()), float(b.std(ddof=1) / math.sqrt(nb))]
        out[(cfg["d"], cfg["seed"])] = dict(
            L1=mb([r["fbar1"] for r in rows]), L3=mb([float(r["fbar3"]) / cfg["eps"] for r in rows]),
            w0sq=mb([float(r["d_w0sq"]) * SIGW2 / cfg["d"] for r in rows]),
            w0=sn["w0"][post].astype(float), last=int(rows[-1]["iter"]), done=(run / "summary.json").exists())
    return out


def provisional_runs(ds=(600,), first_iter=100, N=1500, running_only=True):
    """Ray chains still running (no summary.json): means of obs.csv rows with iter >= first_iter, errors from batch
    means over 10 batches (L1, L3 decorrelate in ~5-10 iterations). running_only=False also takes stopped chains
    without summary.json (the N = 3000 chains)."""
    out = {}
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_ray_d*_N{N}x{N}_*" / "obs.csv"))):
        run = Path(f).parent
        cfg = json.loads((run / "config.json").read_text())["args"]
        if cfg.get("tag") != "ray" or cfg["d"] not in ds or (running_only and (run / "summary.json").exists()):
            continue
        import csv
        rows = [r for r in csv.DictReader(open(f)) if int(r["iter"]) >= first_iter]
        if len(rows) < 8:
            continue

        def mb(x):
            x = np.asarray(x)
            nb = min(10, len(x) // 2)
            b = np.array([c.mean() for c in np.array_split(x, nb)])
            return [float(x.mean()), float(b.std(ddof=1) / math.sqrt(nb))]
        out[(cfg["d"], cfg["init"])] = dict(
            L1=mb([float(r["fbar1"]) for r in rows]), L3=mb([float(r["fbar3"]) / cfg["eps"] for r in rows]),
            w0sq=mb([float(r["d_w0sq"]) * SIGW2 / cfg["d"] for r in rows]), last=int(rows[-1]["iter"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "plots" / "ray" / "theory_vs_hmc_ray.png"))
    a = ap.parse_args()
    (HERE / "plots").mkdir(exist_ok=True)
    cache = HERE / "plots" / "cache" / "theory_readin_ray.npz"
    z = {} if a.recompute or not cache.exists() else dict(np.load(cache))
    missing = [d for d in DS if f"w0_{d}" not in z]
    for d in missing:
        z[f"w0_{d}"], z[f"dens_{d}"] = theory_readin(d, a.device)[:2]
    if missing:
        np.savez(cache, **z)
    th = json.loads((THEORY / "ray_theory_fcn3_N1500.json").read_text())
    th600 = json.loads((THEORY / "ray_theory_fcn3_d600_N1500.json").read_text())
    tds = [75, 150, 300, 600]
    T11 = {k: [th[str(d)]["11"][k] for d in DS[:3]] + [th600[k]] for k in ("L1", "L3", "d_w0sq")}
    T3 = {k: [th[str(d)]["3"][k] for d in DS[:3]] + [th600["nmax3"][k]] for k in ("L1", "L3", "d_w0sq")}
    for T in (T11, T3):
        T["w0sq"] = [v / d for v, d in zip(T["d_w0sq"], tds)]
    S = sampler_runs()
    Sp = provisional_runs()
    S3k = provisional_runs(ds=(150, 600), first_iter=300, N=3000, running_only=False)
    Hs = hist_runs()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    fig = plt.figure(figsize=(15, 7.8))
    top, bot = fig.subfigures(2, 1)
    axs = [top.subplots(1, 4), bot.subplots(1, 3)]

    # ---- top: read-in action
    for ax, d in zip(axs[0], DS):
        w0t, dt_ = z[f"w0_{d}"], z[f"dens_{d}"]
        keep = (dt_ > 1e-30) & (w0t > 0)                  # the w0 = 0 node is a half cell
        xs = np.concatenate([-w0t[keep][::-1], w0t[keep]])
        ys = -np.log(0.5 * np.concatenate([dt_[keep][::-1], dt_[keep]]))   # p(w0) = p(|w0|) / 2
        ax.plot(xs, ys, color=C_THEORY, lw=2, label="theory, n$_{max}$ = 11")
        hs = [Hs[k]["w0"] for k in sorted(Hs) if k[0] == d]
        if hs:
            # every post-burn snapshot of the new datasets; uniform bins. Error per bin: Poisson on the counts
            # deflated by 2 for the snapshot correlation, combined with the spread between datasets.
            w = np.concatenate([h.ravel() for h in hs])
            edges = np.arange(-1.6, 1.6001, 0.04)
            cnt, _ = np.histogram(w, edges)
            width, mid = np.diff(edges), 0.5 * (edges[1:] + edges[:-1])
            ok = cnt > 0
            dens = cnt[ok] / (len(w) * width[ok])
            err = np.sqrt(2.0 / cnt[ok])
            if len(hs) > 1:
                per = []
                for h in hs:
                    c_, _ = np.histogram(h.ravel(), edges)
                    with np.errstate(divide="ignore"):
                        per.append(-np.log(c_[ok] / (h.size * width[ok])))
                per = np.array(per)
                fin = np.isfinite(per).all(0)
                sp = np.where(fin, np.nanstd(np.where(np.isfinite(per), per, np.nan), 0, ddof=1) / math.sqrt(len(hs)), 0)
                err = np.hypot(err, sp)
            nsnap = sum(h.shape[0] for h in hs)
            lab = f"sampler, {len(hs)} dataset{'s' if len(hs) > 1 else ''} ({nsnap} snapshots × 1500)"
        else:
            w = np.concatenate([S[(d, i)]["w0"] for i in ("prior", "aligned") if (d, i) in S])
            # equal-count bins over the bulk (|w0| < 0.3), 0.05-wide bins beyond it, mirrored about 0
            a_ = np.sort(np.abs(w[np.abs(w) < 0.3]))
            pos = np.unique(np.concatenate([a_[::max(1, len(a_) // 15)][1:], [0.3],
                                            np.arange(0.35, max(1.6, np.abs(w).max() + 0.05), 0.05)]))
            edges = np.concatenate([-pos[::-1], [0.0], pos])
            cnt, _ = np.histogram(w, edges)
            width, mid = np.diff(edges), 0.5 * (edges[1:] + edges[:-1])
            ok = cnt > 0
            dens = cnt[ok] / (len(w) * width[ok])
            err = 1 / np.sqrt(cnt[ok])                             # Poisson error on -log density
            lab = f"sampler ({len(w)} neurons, final states only)"
        ax.errorbar(mid[ok], -np.log(dens), xerr=0.5 * width[ok], yerr=err, fmt="o", ms=3.5, color=C_HMC, mec="white", mew=0.6,
                    elinewidth=1, capsize=0, label=lab)
        ax.legend(loc="upper center", fontsize=8)
        lo = float(ys.min())
        ax.set_ylim(lo - 1, lo + 14)
        ax.set_xlim(-1.6, 1.6)
        ax.set_title(f"read-in action, d = {d}", color=INK, loc="left")
        ax.set_xlabel("w$_0$")
        if d == DS[0]:
            ax.set_ylabel("−log p(w$_0$)")

    # ---- bottom: d<w0^2>, L1, L3 vs d
    for ax, key, skey, lab in ((axs[1][0], "w0sq", "w0sq", "⟨w$_0^2$⟩"), (axs[1][1], "L1", "L1", "L$_1$"),
                               (axs[1][2], "L3", "L3", "L$_3$")):
        ax.plot(tds, T11[key], "-", color=C_THEORY, lw=2, marker="o", ms=5, mec="white", label="theory, n$_{max}$ = 11")
        ax.plot(tds, T3[key], "--", color=C_N3, lw=1.5, marker="o", ms=4, mec="white", label="theory, n$_{max}$ = 3")
        for init, mk in (("prior", "s"), ("aligned", "^")):
            pts = [(d, S[(d, init)][skey]) for d in DS if (d, init) in S]
            x = [p[0] * (0.97 if init == "prior" else 1.03) for p in pts]
            y = [p[1][0] for p in pts]
            e = [p[1][1] for p in pts]
            ax.errorbar(x, y, yerr=e, fmt=mk, ms=7, color=C_HMC, mec="white", mew=0.8, elinewidth=1, capsize=0,
                        label=f"sampler, {init} start")
            if (600, init) in Sp:
                q = Sp[(600, init)]
                ax.errorbar([600 * (0.97 if init == "prior" else 1.03)], [q[skey][0]], yerr=[q[skey][1]], fmt=mk,
                            ms=7, mfc="white", mec=C_HMC, mew=1.5, color=C_HMC, elinewidth=1, capsize=0,
                            label=(f"sampler d = 600, in progress (it 100–{q['last']})" if init == "prior" else None))
        first = True
        for (d, seed), q in sorted(Hs.items()):
            ax.errorbar([d * (0.985 + 0.03 * (seed - 1))], [q[skey][0]], yerr=[q[skey][1]], fmt="D", ms=5.5,
                        color=C_HMC, mfc="white" if not q["done"] else C_HMC, mec=C_HMC, mew=1.2, elinewidth=1,
                        capsize=0, label="sampler, further datasets 1–3" if first else None)
            first = False
        for init, mk in (("prior", "s"), ("aligned", "^")):
            pts = [(d, S3k[(d, init)]) for d in DS if (d, init) in S3k]
            if pts:
                ax.errorbar([p[0] * (0.94 if init == "prior" else 1.06) for p in pts], [p[1][skey][0] for p in pts],
                            yerr=[p[1][skey][1] for p in pts], fmt=mk, ms=7, color=C_N3K, mec="white", mew=0.8,
                            elinewidth=1, capsize=0, label=f"sampler N = 3000, {init} start (it ≥ 300)")
        ax.set_xscale("log")
        ax.set_xticks(tds)
        ax.set_xticklabels([str(t) for t in tds])
        ax.minorticks_off()
        ax.set_xlabel("d  (ray: P ∝ d², κ ∝ d)")
        ax.set_title(lab, color=INK, loc="left")
        if key == "L3":
            ax.set_ylim(bottom=0)
        if key == "L1":
            ax.set_ylim(0, 1.5)
        if key == "w0sq":
            ax.set_yscale("log")
            ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("FCN3, κ$_0$ = 0.015 ray, N$_1$ = N$_2$ = 1500 (green: N = 3000, χ = 2, same theory point): equilibrium theory vs exact posterior sampler",
                 color=INK, x=0.01, y=0.995, ha="left", fontsize=12)
    top.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.14, wspace=0.25)
    bot.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.14, wspace=0.25)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    for d in DS:
        for init in ("prior", "aligned"):
            if (d, init) in S:
                s = S[(d, init)]
                print(f"d={d} {init:7s} L1={s['L1'][0]:.4f} L3={s['L3'][0]:.4f} <w0^2>={s['w0sq'][0]:.5f}")
    for (d, init), q in Sp.items():
        print(f"d={d} {init:7s} (provisional, it 100-{q['last']}) L1={q['L1'][0]:.4f}±{q['L1'][1]:.4f} "
              f"L3={q['L3'][0]:.4f}±{q['L3'][1]:.4f} <w0^2>={q['w0sq'][0]:.5f}")
    for (d, seed), q in sorted(Hs.items()):
        print(f"dataset {seed} d={d} ({'done' if q['done'] else 'running'}, to it {q['last']}, {q['w0'].shape[0]} snapshots) "
              f"L1={q['L1'][0]:.4f}±{q['L1'][1]:.4f} L3={q['L3'][0]:.4f}±{q['L3'][1]:.4f} <w0^2>={q['w0sq'][0]:.5f}")
    for (d, init), q in S3k.items():
        print(f"N=3000 d={d} {init:7s} (it 300-{q['last']}) L1={q['L1'][0]:.4f}±{q['L1'][1]:.4f} "
              f"L3={q['L3'][0]:.4f}±{q['L3'][1]:.4f} <w0^2>={q['w0sq'][0]:.5f}")


if __name__ == "__main__":
    main()
