#!/usr/bin/env python3
"""L3 and L1 along the FCN3 ray (d = 75, 150, 300, 600; P = 3000 (d/150)^2, kappa = 0.015 d/150, sigma_A^2 = 0.8/d)
against alpha = ln P / ln d: theory, sampler and the infinite-width (NNGP) network.

Theory: the cached ray solutions (plot_ray_readin_fcn2_vs_fcn3.fcn3_theory, plots/cache/theory_readin_ray_full.npz;
not re-solved). Sampler: every finished N1 = N2 = 1500 ray run (and the onset runs at d = 150, P = 3000), one point per
run, filled = started from the prior, open = started aligned. NNGP: exact GP regression with the twice-composed arcsin
kernel on dataset 0's training set at each d (nngp_fcn3.gpr, cached in plots/cache/nngp_ray_fcn3.json), and the
eigenlearning curve along the ray (nngp_fcn3.eigen, inputs taken on the sphere |x|^2 = d).

    python plot_ray_learnability.py      -> plots/ray/ray_learnability_fcn3.png
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

import plot_onset as po
import plot_ray_readin_fcn2_vs_fcn3 as rr
from plot_perp_compare import C_NNGP, GRID, HERE, INK

RAY = (75, 150, 300, 600)
OUT = HERE / "plots" / "ray" / "ray_learnability_fcn3.png"
CACHE = HERE / "plots" / "cache" / "nngp_ray_fcn3.json"


def ray_runs():
    """{d: [(run dir, args, L1, L3)]} of finished runs at the ray parameters (same selection as rr.fcn3_data)."""
    out = {}
    for d in RAY:
        pats = [f"*_ray_d{d}_P*_N1500x1500_*"] + (["*_onset_d150_P3000_N1500x1500_*"] if d == 150 else [])
        for pat in pats:
            for f in sorted(glob.glob(str(HERE / "runs" / pat / "config.json"))):
                run = Path(f).parent
                if "aborted" in run.name or run.name.endswith(("r71", "r72")) or not (run / "summary.json").exists():
                    continue
                a = json.loads(Path(f).read_text())["args"]
                ps = json.loads((run / "summary.json").read_text())["posterior"]
                out.setdefault(d, []).append((run, a, ps["L1"][0], ps["L3"][0]))
    return out


def nngp(runs):
    import nngp_fcn3
    c = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out = {}
    for d, rs in runs.items():
        run = next(r for r, a, *_ in rs if a["seed"] == 0 and a["init"] == "prior" and "_ray_" in r.name)
        if run.name not in c:
            g = nngp_fcn3.gpr(run)
            c[run.name] = [g["L1"], g["L3"]]
            print(f"NNGP d={d}: L1 {g['L1']:.4f} L3 {g['L3']:.4f}", flush=True)
            CACHE.write_text(json.dumps(c, indent=1))
        out[d] = c[run.name]
    return out


def main():
    import nngp_fcn3
    plt = po.style()
    th = rr.fcn3_theory()
    runs = ray_runs()
    G = nngp(runs)
    al = {d: math.log(3000 * (d / 150) ** 2) / math.log(d) for d in RAY}
    for d in RAY:
        L = np.array([(x[2], x[3]) for x in runs[d]])
        print(f"d={d}: theory L1 {th[d][2]:.4f} L3 {th[d][3]:.4f}; sampler L1 {L[:, 0].mean():.4f} L3 "
              f"{L[:, 1].mean():.4f} ({len(L)} runs); NNGP L1 {G[d][0]:.4f} L3 {G[d][1]:.4f}")
    dE = np.geomspace(75, 600, 60)
    sB = {d: runs[d][0][1]["sigB"] for d in RAY}
    E = [nngp_fcn3.eigen(int(round(d)), 3000 * (d / 150) ** 2, 0.015 * d / 150, sA2=0.8 / d,
                         sB=float(np.interp(d, RAY, [sB[x] for x in RAY]))) for d in dE]
    aE = [math.log(3000 * (d / 150) ** 2) / math.log(d) for d in dE]
    lD = np.log(RAY)
    col = {d: plt.cm.viridis(0.9 * (math.log(d) - lD.min()) / (lD.max() - lD.min())) for d in RAY}
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
    for ax, (name, key, i) in zip(axs, (("L$_3$", "L3", 1), ("L$_1$", "L1", 0))):
        x = [al[d] for d in RAY]
        ax.plot(x, [th[d][2 + i] for d in RAY], "-", color=INK, lw=2, zorder=2)
        ax.plot(aE, [e[key] for e in E], "-", color=C_NNGP, lw=1.4, zorder=2)
        ax.plot(x, [G[d][i] for d in RAY], "s", color=C_NNGP, ms=7, mfc="white", mew=1.4, zorder=3)
        for d in RAY:
            for run, a, L1, L3 in runs[d]:
                pr = a["init"] == "prior"
                ax.plot([al[d]], [(L1, L3)[i]], "o", ms=7.5, color=col[d], mfc=col[d] if pr else "white",
                        mec="white" if pr else col[d], mew=0.6 if pr else 1.4, alpha=0.9, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{al[d]:.3f}\nd={d}, P={3000 * d * d // 22500}" for d in RAY], fontsize=8)
        ax.set_xlabel("α = ln P / ln d")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name} along the ray", color=INK, loc="left", fontsize=11.5)
    ax = axs[2]
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Parameters (FCN3 ray)\n"
            "  d = 75, 150, 300, 600\n"
            "  P = 3000 (d/150)$^2$,  κ = 0.015 d/150\n"
            "  σ$_A^2$ = 0.8/d,  σ$_w^2$ = 0.5,  σ$_B$ ≈ 1.255\n"
            "  N$_1$ = N$_2$ = 1500,  χ = 1,  ε = −0.3√6\n"
            "Sampler: one point per run, all datasets\n"
            "NNGP: infinite width (N$_1$, N$_2$ → ∞)",
            transform=ax.transAxes, fontsize=10, color=INK, va="top", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRID))
    h = [plt.Line2D([], [], color=INK, lw=2),
         plt.Line2D([], [], color=C_NNGP, ls="", marker="s", mfc="white", mew=1.4, ms=7),
         plt.Line2D([], [], color=C_NNGP, lw=1.4),
         plt.Line2D([], [], color=col[150], marker="o", ls="", mec="white", ms=7.5),
         plt.Line2D([], [], color=col[150], marker="o", ls="", mfc="white", mew=1.4, ms=7.5)]
    ax.legend(h, ["theory", "NNGP, exact regression (dataset 0)", "NNGP, eigenlearning (sphere)",
                  "sampler, started from the prior", "sampler, started aligned"] +
              [], fontsize=9, loc="lower left")
    fig.suptitle("FCN3 along the ray: theory, sampler and the infinite-width network; colour = d",
                 color=INK, x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
