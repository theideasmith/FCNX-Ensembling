#!/usr/bin/env python3
"""L3 along the kappa0 = 0.015 invariant ray (P = 3000 (d/150)^2, kappa = 0.015 d/150, N = 1500,
sigma_A^2 = (8/1500)(150/d), sigma_w^2 = 0.5, eps = -0.3 sqrt(6)): two-layer against three-layer network (the FCN3 adds
the hidden layer, N2 = 1500, sigma_B = 1.254-1.258).

FCN2: exact population He3 coefficient of the posterior-mean predictor from every saved sample of each chain
(fcn2_ray_L3_exact.py), one point per chain; theory at He11 from the headline table of fcn2_collapsed_hmc/FINDINGS.md. FCN3: each run's summary.json (exact x0 quadrature); theory re-solved with
the current solver (plots/cache/fcn3_theory_ray.json); one point per run, N = 3000 (chi = 2, theory unchanged) marked separately.

    python plot_ray_fcn2_vs_fcn3.py   ->  plots/ray/fcn2_vs_fcn3_L3_ray.png
"""
import glob
import json
from pathlib import Path

import numpy as np

import plot_onset as po

HERE = Path(__file__).resolve().parent
C2, C3 = "#1baf7a", "#2a78d6"
# fcn2_collapsed_hmc/FINDINGS.md, headline table: d -> (L3 HMC, L3 theory)
FCN2 = {75: (0.318, 0.391), 150: (0.370, 0.387), 300: (0.376, 0.385), 600: (0.381, 0.383), 800: (0.382, 0.382),
        1200: (0.382, 0.382)}


def fcn3_runs():
    out = []
    for f in sorted(glob.glob(str(HERE / "runs" / "*_ray_d*" / "config.json"))):
        if "aborted" in f:
            continue
        a = json.loads(Path(f).read_text())["args"]
        s = Path(f).with_name("summary.json")
        if not s.exists():
            continue
        s = json.loads(s.read_text())
        if a["P"] != round(3000 * (a["d"] / 150) ** 2) or abs(a["kappa"] - 0.015 * a["d"] / 150) > 1e-9:
            continue
        out.append(dict(d=a["d"], N=a["N1"], L3=s["posterior"]["L3"][0], theory=s["theory"]["L3"], seed=a["seed"],
                        fixed="r7" in Path(f).parent.name[-3:]))
    return out


def main():
    rs = fcn3_runs()
    plt = po.style()
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ds2 = sorted(FCN2)
    ax.plot(ds2, [FCN2[d][1] for d in ds2], "--", color=C2, lw=2.2, label="FCN2, theory")
    ex = json.loads((HERE / "plots" / "cache" / "fcn2_ray_L3_exact.json").read_text())   # fcn2_ray_L3_exact.py
    pts = [(int(d), c["L3"]) for d, ch in ex.items() for c in ch.values()]
    ax.plot([p for p, _ in pts], [l for _, l in pts], "s", color=C2, ms=6, mec="white", zorder=3,
            label="FCN2, sampler (one point per chain)")
    # current solver (fcn3_fixed.solve_fcn3, n_max = 11); the theory stored in older run summaries is out of date
    th3 = {int(k): v for k, v in json.loads((HERE / "plots" / "cache" / "fcn3_theory_ray.json").read_text()).items()}
    ds3 = sorted(th3)
    ax.plot(ds3, [th3[d] for d in ds3], "-", color=C3, lw=2.2, label="FCN3, theory")
    a = [r for r in rs if r["N"] == 1500 and not r["fixed"]]
    b = [r for r in rs if r["N"] == 3000]
    ax.plot([r["d"] for r in a], [r["L3"] for r in a], "o", color=C3, ms=6, mec="white", zorder=3,
            label="FCN3, sampler (one point per run)")
    if b:
        ax.plot([r["d"] for r in b], [r["L3"] for r in b], "D", color="#0b3d91", ms=6, mec="white", zorder=3,
                label="FCN3, sampler at N = 3000 (χ = 2)")
    ax.set_xscale("log")
    ax.set_xticks([75, 150, 300, 600, 1200])
    ax.set_xticklabels(["75", "150", "300", "600", "1200"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("d   (P = 3000 (d/150)$^2$, κ = 0.015 d/150)")
    ax.set_ylabel("L$_3$")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("He$_3$ learnability along the κ$_0$ = 0.015 ray, N = 1500", color=po.INK, loc="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / "fcn2_vs_fcn3_L3_ray.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    for d in ds3:
        x = [r["L3"] for r in a if r["d"] == d]
        y = [c["L3"] for c in ex[str(d)].values()]
        print(f"d={d}: FCN3 sampler {np.round(x, 3).tolist()} theory {th3[d]:.3f};  FCN2 sampler {np.round(y, 3).tolist()} "
              f"theory {FCN2[d][1]:.3f}")


if __name__ == "__main__":
    main()
