"""Replot V(w) at chi=1 along the kappa ray from the saved RubinRegime action CSVs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent / "action_h0_activation_plots" / "RubinRegime"
KAPPAS = ["0p18", "0p09", "0p07", "0p06"]

fig, ax = plt.subplots(figsize=(8.3, 4.25))
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(KAPPAS)))
for tag, color in zip(KAPPAS, colors):
    w, V = np.loadtxt(HERE / "actions" / f"a0s_sigma_{tag}.csv", delimiter=",", skiprows=1, unpack=True)
    ax.plot(w, V, color=color, lw=2.5, label=rf"$\kappa={tag.replace('p', '.')}$")
ax.axhline(0.0, color="0.75", lw=0.8)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-16, 12)
ax.set_xlabel(r"$w$", fontsize=22)
ax.set_ylabel(r"$V(w)$", fontsize=22)
ax.set_title(r"$\chi=1,\ \sigma_A^2=8/N$", fontsize=22)
ax.tick_params(labelsize=18)
ax.legend(loc="lower left", fontsize=18, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "V_action_chi1_kappa_ray.png", dpi=200)
