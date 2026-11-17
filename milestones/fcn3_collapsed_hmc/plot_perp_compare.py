#!/usr/bin/env python3
"""Theory against the sampler at one theory point, with the infinite-width (NNGP) learnabilities.

Theory point: d = 150, kappa/P = 0.015/24000 (the theory depends on P and kappa only through kappa/P).
Sampler: dataset 0, started from the prior, every finished run on that line (kappa = P x 0.015/24000).

Top: -log p of the read-in w0, of h|He1 and of h|He3 (every neuron of the networks saved after burn-in), all P
overlaid, colour = P. Bottom: L3 and L1 against alpha = ln P / ln d (y axis from 0).
Theory: plots/cache/theory_onset_d150_v6.npz (P = 24000, kappa = 0.015, from plot_onset.py; not recomputed).
NNGP: exact GP regression with the twice-composed arcsin kernel on each run's own training set (nngp_fcn3.gpr),
cached in plots/cache/nngp_fixed_theory_d150.json.

    python plot_perp_compare.py          -> plots/onset_checks/perp_compare_fixed_theory_d150.png
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
from fcn3_matrix_solver import readin_features  # noqa: E402

RUNS = HERE / "runs"
D, RATIO = 150, 0.015 / 24000
OUT = HERE / "plots" / "onset_checks" / "perp_compare_fixed_theory_d150.png"
NNGP_CACHE = HERE / "plots" / "cache" / "nngp_fixed_theory_d150.json"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
C_NNGP = "#e34948"


def runs_on_point():
    """{P: run dir} of the finished prior-start dataset-0 runs at kappa/P = RATIO (the latest one per P)."""
    out = {}
    for f in sorted(RUNS.glob("*/config.json")):
        a = json.loads(f.read_text())["args"]
        if (a["d"] == D and a["seed"] == 0 and a["init"] == "prior" and not a.get("init_from")
                and a["N1"] == 1500 and abs(a["kappa"] / a["P"] / RATIO - 1) < 1e-9
                and (f.parent / "summary.json").exists()):
            out[int(a["P"])] = f.parent
    return dict(sorted(out.items()))


def sampler(run):
    """read-in w0 and (h1, h3) of every neuron of every network saved after burn-in; posterior L1, L3."""
    w0, H = [], []
    for f in sorted((run / "nets").glob("net_*.pt")):
        nt = torch.load(f, map_location="cpu", weights_only=False)
        ZW, ZB = nt["ZW"].double(), nt["ZB"].double()
        w, q = nt["sw"] * ZW[:, 0], nt["sw"] ** 2 * (ZW[:, 1:] ** 2).sum(1)
        w0.append(w.numpy())
        H.append(((nt["sb"] * ZB) @ readin_features(w, q, (1, 3)).T).numpy())
    ps = json.loads((run / "summary.json").read_text())["posterior"]
    return dict(w0=np.concatenate(w0), h=np.concatenate(H), L1=ps["L1"], L3=ps["L3"])


def theory():
    z = np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz")
    w, dn = z["w0_24000"], z["dens_24000"]
    k = dn > 1e-300
    th = dict(w=w[k], nlp=-np.log(0.5 * dn[k]), h1=(z["h1g_24000"], -z["lp1_24000"]),
              h3=(z["h3g_24000"], -z["lp3_24000"]))
    s = json.loads((next(iter(runs_on_point().values())) / "summary.json").read_text())["theory"]
    th["L1"], th["L3"] = float(s["L1"]), float(s["L3"])
    return th


def nngp(runs):
    """{P: (L1, L3)} of exact NNGP regression on each run's training set (cached by run name)."""
    import nngp_fcn3
    c = json.loads(NNGP_CACHE.read_text()) if NNGP_CACHE.exists() else {}
    for P, r in runs.items():
        if r.name not in c:
            g = nngp_fcn3.gpr(r)
            c[r.name] = [g["L1"], g["L3"]]
            print(f"NNGP P={P}: L1 {g['L1']:.4f} L3 {g['L3']:.4f}", flush=True)
            NNGP_CACHE.write_text(json.dumps(c, indent=1))
    return {P: c[r.name] for P, r in runs.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    th = theory()
    runs = runs_on_point()
    S = {P: sampler(r) for P, r in runs.items()}
    G = nngp(runs)
    for P, r in runs.items():
        print(f"P={P:6d} kappa={P * RATIO:.6g}: L1 {S[P]['L1'][0]:.4f} L3 {S[P]['L3'][0]:.4f};  "
              f"NNGP L1 {G[P][0]:.4f} L3 {G[P][1]:.4f}  ({r.name})")
    print(f"theory L1 {th['L1']:.4f} L3 {th['L3']:.4f}")
    Ps = list(S)
    lP = np.log(Ps)
    col = {P: plt.cm.viridis(0.9 * (math.log(P) - lP.min()) / (lP.max() - lP.min())) for P in Ps}
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    lim_w = max(1.6, 1.08 * max(np.abs(s["w0"]).max() for s in S.values()))
    lim_h = [1.08 * max(max(np.abs(s["h"][:, j]).max() for s in S.values()),
                        np.abs(th[k][0][th[k][1] <= th[k][1].min() + 9]).max())
             for j, k in ((0, "h1"), (1, "h3"))]
    rows = [("w$_0$  (read-in neuron)", "read-in layer", lim_w, 81, 14),
            ("h | He$_1$  (hidden neuron)", "hidden layer, He$_1$", lim_h[0], 81, 10),
            ("h | He$_3$  (hidden neuron)", "hidden layer, He$_3$", lim_h[1], 81, 10)]
    for i, (xl, name, lim, nb, top) in enumerate(rows):
        ax = axs[0, i]
        if i == 0:
            g, y = np.concatenate([-th["w"][::-1], th["w"]]), np.concatenate([th["nlp"][::-1], th["nlp"]])
        else:
            g, y = th["h1" if i == 1 else "h3"]
        k = np.abs(g) <= lim
        ax.plot(g[k], y[k], "-", color=INK, lw=2, zorder=4)
        lo = y[k].min()
        for P in Ps:
            x = S[P]["w0"] if i == 0 else S[P]["h"][:, i - 1]
            edges = np.linspace(-lim, lim, nb)
            c, _ = np.histogram(x, edges)
            ym = np.where(c > 0, -np.log(np.maximum(c, 1) / (c.sum() * np.diff(edges))), np.nan)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), ym, "o-", ms=3.2, lw=0.7, color=col[P], mec="white", mew=0.3,
                    zorder=3)                            # lines join adjacent occupied bins only
            lo = min(lo, np.nanmin(ym))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lo - 0.5, lo + top)
        ax.set_xlabel(xl)
        ax.set_ylabel("−log p")
        ax.set_title(name, color=INK, loc="left", fontsize=11.5)
    # learnabilities against alpha
    al = np.array([math.log(P) / math.log(D) for P in Ps])
    for j, (name, key, idx) in enumerate((("L$_3$", "L3", 1), ("L$_1$", "L1", 0))):
        ax = axs[1, j]
        ax.axhline(th[key], color=INK, lw=2)
        ax.plot(al, [G[P][idx] for P in Ps], "s--", color=C_NNGP, lw=1.6, ms=6, mfc="white", mew=1.4, zorder=2)
        ax.plot(al, [S[P][key][0] for P in Ps], "-", color=MUTED, lw=0.8, zorder=2)
        for a, P in zip(al, Ps):
            ax.errorbar([a], [S[P][key][0]], yerr=[S[P][key][1]], fmt="o", ms=8, color=col[P], mec="white",
                        mew=0.5, capsize=0, zorder=3)
        ax.set_xticks(al)
        ax.set_xticklabels([f"{a:.2f}\nP={P}" for a, P in zip(al, Ps)], fontsize=8)
        ax.set_xlabel("α = ln P / ln d")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name} against α (theory is the same at every P)", color=INK, loc="left", fontsize=11.5)
    hT = [plt.Line2D([], [], color=INK, lw=2),
          plt.Line2D([], [], color=C_NNGP, lw=1.6, ls="--", marker="s", mfc="white", mew=1.4, ms=6)]
    lT = ["theory", "NNGP (infinite width, same training set)"]
    axs[1, 0].legend(hT, lT, fontsize=9, loc="center right")
    fig.suptitle("Theory against the sampler at one theory point: P and κ scaled together (κ/P fixed), so the theory "
                 "is the same for every P; colour = P", color=INK, x=0.01, ha="left", fontsize=12.5)
    ax = axs[1, 2]
    ax.axis("off")
    ax.text(0.02, 1.0,
            "Parameters (FCN3)\n"
            f"  P = {', '.join(str(P) for P in Ps)}\n"
            f"  κ = P × {RATIO:.4g}  (κ = 0.015 at P = 24000)\n"
            f"  d = {D},  N$_1$ = N$_2$ = 1500,  χ = 1\n"
            "  ε = −0.3√6,  σ$_A^2$ = 8/1500,  σ$_B$ = 1.2554,  σ$_w^2$ = 0.5\n"
            "Sampler: dataset 0, started from the prior,\n"
            "  1000 iterations, first 300 discarded;\n"
            "  points: every neuron of every network saved\n"
            "  after burn-in (one every 50 iterations)\n"
            "NNGP: GP regression with the arcsin kernel composed\n"
            "  twice (N$_1$, N$_2$ → ∞), on the same training set",
            transform=ax.transAxes, fontsize=10, color=INK, va="top", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRID))
    hP = [plt.Line2D([], [], color=col[P], marker="o", ms=5, lw=0.7) for P in Ps]
    ax.legend([hT[0]] + hP, ["theory"] + [f"sampler, P = {P}, κ = {P * RATIO:g}  (α = {math.log(P) / math.log(D):.2f})"
                                          for P in Ps], fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
