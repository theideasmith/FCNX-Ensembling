"""Replot V(w) at kappa=0.07 across chi for y=(1,-0.3) and y=(1,+0.3) from the saved RubinRegime action CSVs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent / "action_h0_activation_plots" / "RubinRegime"
CHIS = ["1p0", "2p0", "5p0", "10p0", "20p0", "50p0"]
PANELS = [
    ("k07center", r"$\kappa=0.07,\ y=(1,-0.3)$", (-6.2, 6.2), (-3000, 120)),
    ("k07plus", r"$\kappa=0.07,\ y=(1,+0.3)$", (-7.2, 7.2), (-1400, 220)),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(CHIS)))
for ax, (prefix, title, xlim, ylim) in zip(axes, PANELS):
    for tag, color in zip(CHIS, colors):
        w, V = np.loadtxt(HERE / "actions" / f"{prefix}_chi_{tag}.csv", delimiter=",", skiprows=1, unpack=True)
        ax.plot(w, V, color=color, lw=3, label=rf"$\chi={float(tag.replace('p', '.')):g}$")
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$w$", fontsize=24)
    ax.set_ylabel(r"$V(w)$", fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.tick_params(labelsize=18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(CHIS), fontsize=20, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(HERE / "V_action_kappa0p07_chi.png", dpi=200)
