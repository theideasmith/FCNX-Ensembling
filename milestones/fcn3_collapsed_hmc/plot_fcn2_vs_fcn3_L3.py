#!/usr/bin/env python3
"""L3 against P at fixed kappa = 0.015, d = 150: the two-layer network against the three-layer one, same read-in width
N = 1500, sigma_w^2 = 0.5, sigma_A^2 = 8/1500, target He1 + eps He3 with eps = -0.3 sqrt(6) (the FCN3 adds the hidden
layer, N2 = 1500, sigma_B = 1.2554).

FCN2 theory: per-neuron perpendicular norm, x0 tower to He11 (fcn2_collapsed_hmc/readin_action_theory_q.analyse_q),
cached by the run that made plots/cache/fcn2_theory_onset_d150.json. FCN2 sampler: the d = 150 runs at P = 3000
(the only kappa = 0.015 FCN2 runs at d = 150), exact population coefficients. FCN3: theory and sampler of the onset scan (plot_onset.py), one point
per dataset.

    python plot_fcn2_vs_fcn3_L3.py   ->  plots/onset/fcn2_vs_fcn3_L3_d150.png
"""
import glob
import json
from pathlib import Path

import numpy as np

import plot_onset as po

HERE = Path(__file__).resolve().parent
FCN2 = HERE.parent / "fcn2_collapsed_hmc"
C2, C3 = "#1baf7a", "#2a78d6"


def fcn2_hmc():
    """Exact population He3 coefficient of the FCN2 posterior-mean predictor (fcn2_collapsed_hmc/chains.chain_fbar
    cache; the sampler's own summary uses a fixed set of 3000 test inputs and reads 0.349 instead of 0.370 here)."""
    eps = -0.3 * 6 ** 0.5
    out = []
    for f in glob.glob(str(FCN2 / "plots" / "cache" / "chain_fbar_*d150_P*_1-3-5-7-9-11.json")):
        P = int(Path(f).name.split("_P")[1].split("_")[0])
        out += [(P, v[1] / eps) for v in json.loads(Path(f).read_text()).values()]
    return out


def main():
    th2 = {int(k): v for k, v in json.loads((HERE / "plots" / "cache" / "fcn2_theory_onset_d150.json").read_text()).items()}
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    runs = po.sampler_runs()
    po.set_outer(runs, z)
    groups = po.by_P(runs)
    Ps = sorted(th2)
    plt = po.style()
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(Ps, [z[f"sc_{P}"][1] for P in Ps], "-", color=C3, lw=2.2, label="FCN3, theory")
    first = True
    for P in Ps:
        if P not in groups:
            continue
        L = [e["L3"][0] for e in po.per_dataset(groups[P])]
        ax.plot([P] * len(L), L, "o", color=C3, ms=6, mec="white", zorder=3,
                label="FCN3, sampler (one point per dataset)" if first else None)
        first = False
    ax.plot(Ps, [th2[P]["L3"] for P in Ps], "--", color=C2, lw=2.2, label="FCN2, theory")
    h2 = fcn2_hmc()
    if h2:
        ax.plot([p for p, _ in h2], [l for _, l in h2], "s", color=C2, ms=6, mec="white", zorder=3,
                label="FCN2, sampler")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("P")
    ax.set_ylabel("L$_3$")
    ax.legend(loc="upper left")
    ax.set_title(f"He$_3$ learnability at fixed κ = 0.015, d = 150, N = 1500", color=po.INK, loc="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset" / "fcn2_vs_fcn3_L3_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    print("FCN2 sampler points:", sorted(h2))


if __name__ == "__main__":
    main()
