#!/usr/bin/env python3
"""Actions at fixed theory as P grows with kappa ∝ P (queue_fixedtheory.sh): theory points P = 12000 and 24000 on the
onset line (d = 150, N = 1500, kappa = 0.015), reached at P x 1 (the onset run, dataset 0), x 2 (kappa 0.03) and x 4
(kappa 0.06). Every parameter of the theory is the same within a column, so the theory curve is one curve; if the
sampler's distributions move towards it as P grows, the difference is the finite training set.

Rows: -log p of the read-in w0 (every snapshot after burn-in), of h|He1 and of h|He3 (every hidden neuron of the saved
networks).

    python plot_ek_actions.py   ->  plots/onset/ek_actions_d150.png
"""
import glob
import json
from pathlib import Path

import numpy as np

import plot_onset as po

HERE = Path(__file__).resolve().parent
POINTS = [12000, 24000]
COLS = ["#f4a582", "#d6604d", "#8a1c1c"]          # x1, x2, x4: light to dark


def runs_for(Pt):
    out = []
    onset = [r for r in po.by_P(po.sampler_runs())[Pt] if "onset" in r["path"].name and r["seed"] == 0][0]
    out.append((1, onset["path"]))
    for f in sorted(glob.glob(str(HERE / "runs" / "*ptheory*" / "config.json"))):
        a = json.loads(Path(f).read_text())["args"]
        if int(round(a["P"] * po.KAPPA / a["kappa"])) == Pt:
            out.append((a["P"] // Pt, Path(f).parent))
    return sorted(out)


def nlp(x, edges):
    c, _ = np.histogram(x, edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    k = c > 0
    return mid[k], -np.log(c[k] / (c.sum() * np.diff(edges)[k]))


def main():
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    wide = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_hidden_wide.npz"))
    plt = po.style()
    fig, axs = plt.subplots(3, len(POINTS), figsize=(7.2 * len(POINTS), 12), squeeze=False)
    for c, Pt in enumerate(POINTS):
        rs = runs_for(Pt)
        # theory
        w, dn = z[f"w0_{Pt}"], z[f"dens_{Pt}"]
        k = (dn > 1e-300) & (w <= 1.5)
        xs = np.concatenate([-w[k][::-1], w[k]])
        ys = -np.log(0.5 * np.concatenate([dn[k][::-1], dn[k]]))
        th = [(xs, ys), (wide[f"h1g_{Pt}"], -wide[f"lp1_{Pt}"]), (wide[f"h3g_{Pt}"], -wide[f"lp3_{Pt}"])]
        lims = [1.5, 3.0, 0.4]
        tops = [32, 12, 12]
        for i in range(3):
            ax = axs[i, c]
            g, y = th[i]
            kk = np.abs(g) <= lims[i]
            ax.plot(g[kk], y[kk], color=po.C_THEORY, lw=2.2, label=f"theory (point P = {Pt})", zorder=2)
            lo = y[kk].min()
            for (mult, path), col in zip(rs, COLS):
                cfg = json.loads((path / "config.json").read_text())["args"]
                if i == 0:
                    sn = np.load(path / "snaps.npz")
                    x = sn["w0"][sn["iter"] >= cfg["burn"]].ravel()
                    edges = np.linspace(-lims[0], lims[0], 61)
                else:
                    H, _ = po.sampler_hidden([dict(path=path)])
                    x = H[:, i - 1]
                    edges = np.linspace(-lims[i], lims[i], 81)
                mid, yy = nlp(x, edges)
                ax.plot(mid, yy, "o-", color=col, ms=3.2, lw=1.0, zorder=3,
                        label=f"sampler, P = {cfg['P']}, κ = {cfg['kappa']:g} (P × {mult})")
                lo = min(lo, yy.min())
            ax.set_xlim(-lims[i], lims[i])
            ax.set_ylim(lo - 0.5, lo + tops[i])
            ax.set_xlabel(["w$_0$", "h | He$_1$", "h | He$_3$"][i])
            ax.set_ylabel(f"−log p({['w$_0$', 'h | He$_1$', 'h | He$_3$'][i]})")
            if i == 0:
                ax.set_title(f"theory point P = {Pt}", color=po.INK, loc="left", fontsize=12)
                ax.legend(fontsize=9, loc="upper center")
    fig.suptitle(f"Actions at fixed theory as P grows with κ ∝ P, d = {po.D}, N = {po.N}, dataset 0", color=po.INK,
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    out = HERE / "plots" / "onset" / "ek_actions_d150.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
