"""Plot He1/He3 learnabilities vs alpha (P = d^alpha) and vs chi from learnability_sweeps_rubin.jl output."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent / "action_h0_activation_plots" / "RubinRegime"
FIGS = [
    ("learnability_vs_alpha_kappa0p07_chi1", r"$\alpha \;\; (P=d^{\alpha})$", r"$\chi=1,\ \kappa=0.07,\ d=150$", False),
    ("learnability_vs_chi_kappa0p07_P3000", r"$\chi$", r"$\kappa=0.07,\ P=3000,\ d=150$", True),
]
COLORS = {"L1": "#1f4e79", "L3": "#c0504d"}

for stem, xlabel, title, logx in FIGS:
    x, eps, L1, L3 = np.genfromtxt(HERE / f"{stem}.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2, 3), unpack=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for e, ls in ((-0.3, "-"), (0.3, "--")):
        m = np.isclose(eps, e)
        for key, L in (("L1", L1), ("L3", L3)):
            name = r"\mathrm{He}_1" if key == "L1" else r"\mathrm{He}_3"
            ax.plot(x[m], L[m], ls, marker="o", ms=6, lw=3, color=COLORS[key],
                    label=rf"${name},\ \varepsilon={e:+.1f}$")
    ax.axhline(0.0, color="0.75", lw=0.8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel("learnability", fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=17, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(HERE / f"{stem}.png", dpi=200)
