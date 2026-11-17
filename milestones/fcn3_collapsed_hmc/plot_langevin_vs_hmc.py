#!/usr/bin/env python3
"""Does Langevin dynamics, started from HMC end states, move the read-in distribution? (langevin_fcn3.py --init-from)

Top row, against Langevin step, one line per network (solid: full step, dashed: half step; colour: which HMC state it
started from): number of read-in neurons beyond the theory's barrier, d <w0^2>, <q> = <|w_perp|^2>, and L3. The grey
band is the HMC posterior on the same data (onset run, dataset 0, every snapshot after burn-in: mean +- 1 sd; for L3
the band is the four HMC end states at step 0 with the same estimator, one readout draw on 20000 fixed fresh inputs,
and the line HMC's own estimate). Bottom: -log p(|w0|) of the read-in neurons, HMC (every post-burn snapshot) against
the Langevin networks at the first save and from the second half of the run.

    python plot_langevin_vs_hmc.py [run dir]   ->  plots/langevin/langevin_vs_hmc_P3000.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import plot_onset as po

HERE = Path(__file__).resolve().parent
HMC_RUN = HERE / "runs" / "20261109-173149_onset_d150_P3000_N1500x1500_prior_r10"
START = {0: "HMC end state 1 (prior start)", 1: "HMC end state 2 (aligned start)"}


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in (HERE / "runs" / "langevin").glob("*from_hmc*") if p.is_dir())[-1]
    cfg = json.loads((run / "config.json").read_text())
    cut = cfg["args"]["cut"]
    init = cfg["args"]["init_from"]
    uniq = list(dict.fromkeys(init))
    start_of = lambda e: uniq.index(init[e % len(init)])
    tr = pd.read_csv(run / "trace.csv")
    hc = json.loads((HMC_RUN / "config.json").read_text())["args"]
    sn = np.load(HMC_RUN / "snaps.npz")
    post = sn["iter"] >= hc["burn"]
    w0h, qh = sn["w0"][post], sn["q"][post]
    ref = dict(outer=(np.abs(w0h) > cut).sum(1), w0sq_d=(w0h ** 2).mean(1) * hc["d"], q=qh.mean(1))
    L3h = json.loads((HMC_RUN / "summary.json").read_text())["posterior"]["L3"]

    plt = po.style()
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
    cols = ["#2a78d6", "#1baf7a"]
    panels = [("outer", f"read-in neurons with |w$_0$| > {cut:.2f}"), ("w0sq_d", "d ⟨w$_0^2$⟩"),
              ("q", "⟨q⟩ = ⟨|w$_\\perp|^2$⟩"), ("L3", "L$_3$")]
    for i, (k, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if k == "L3":                      # different estimator from HMC's; step 0 is the HMC states themselves
            m0 = tr[tr["step"] == 0]["L3"]
            ax.axhspan(m0.min(), m0.max(), color="#d7d6d0", zorder=0)
            ax.axhline(L3h[0], color=po.C_HMC, lw=1.5)
        else:
            m, s = ref[k].mean(), ref[k].std()
            ax.axhspan(m - s, m + s, color="#d7d6d0", zorder=0)
            ax.axhline(m, color=po.C_HMC, lw=1.5, label="HMC posterior")
        for e, g in tr.groupby("net"):
            st = start_of(e)
            ax.plot(g["step"], g[k], color=cols[st % 2], lw=1.2, ls="--" if g["half_step"].iloc[0] else "-",
                    label=f"Langevin from {START.get(st, st)}" + (", half step" if g["half_step"].iloc[0] else ""))
        ax.set_xlabel("Langevin step")
        ax.set_title(lab, color=po.INK, loc="left", fontsize=11)
        if k in ("outer", "w0sq_d", "q"):
            ax.set_ylim(bottom=0)
    h_, l_ = fig.axes[0].get_legend_handles_labels()

    # read-in |w0| distribution
    nets = sorted((run / "nets").glob("net_*.pt"))
    steps = sorted({int(f.name.split("_")[1]) for f in nets})
    if steps:
        last = steps[-1]
        early = [f for f in nets if int(f.name.split("_")[1]) == steps[0]]
        late = [f for f in nets if int(f.name.split("_")[1]) >= last // 2 and int(f.name.split("_")[1]) > steps[0]]
        load = lambda fs: np.concatenate([(torch.load(f, weights_only=False)["ZW"][:, 0] *
                                           torch.load(f, weights_only=False)["sw"]).numpy() for f in fs]) if fs else None
        ax = fig.add_subplot(gs[1, :2])
        edges = np.linspace(0, 1.6, 65)
        mid = 0.5 * (edges[1:] + edges[:-1])
        for x, lab, col, mk in ((np.abs(w0h).ravel(), "HMC, every post-burn snapshot", po.C_HMC, "o"),
                                (np.abs(load(early)), f"Langevin, step {steps[0]}", "#9cc3ee", "s"),
                                (np.abs(load(late)) if late else None, f"Langevin, steps {last // 2}–{last}",
                                 "#0b3d91", "D")):
            if x is None:
                continue
            c, _ = np.histogram(x, edges)
            k = c > 0
            ax.plot(mid[k], -np.log(c[k] / (c.sum() * np.diff(edges)[k])), mk + "-", color=col, ms=4, lw=1.2,
                    label=lab)
        ax.axvline(cut, color=po.MUTED, lw=0.8, ls=":")
        ax.set_xlabel("|w$_0$|")
        ax.set_ylabel("−log p(|w$_0$|)")
        ax.set_title("read-in distribution", color=po.INK, loc="left", fontsize=11)
        ax.legend(fontsize=9)
    fig.legend(h_, l_, loc="lower right", bbox_to_anchor=(0.98, 0.08), fontsize=9.5, ncol=1)
    a = cfg["args"]
    fig.suptitle(f"Langevin dynamics started from HMC end states, d = {a['d']}, P = {a['P']}, N = {a['N']}, "
                 f"κ = {a['kappa']}, dataset {a['seed']} (steps: W {a['lr_w']:.0e}, A {a['lr_a']:.0e})",
                 color=po.INK, x=0.01, ha="left", fontsize=12)
    out = HERE / "plots" / "langevin" / "langevin_vs_hmc_P3000.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
