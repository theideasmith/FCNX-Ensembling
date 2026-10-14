#!/usr/bin/env python3
"""Plot fine mode-Laplace ρ scan around the L3 bump; overlay multiwell χ=N."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FINE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine"
COARSE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
MW = HERE / "action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/summary_deep_equilibrium.csv"


def load_mode(directory: Path) -> np.ndarray | None:
    rows = []
    for p in sorted(directory.glob("rho*_scfrac.npz")):
        z = np.load(p, allow_pickle=True)
        eps = float(z["eps"]) if "eps" in z.files else 1.0
        pop = np.asarray(z["pop"], dtype=float)
        pop1, pop3 = float(pop[0]), float(pop[1])
        rows.append(
            dict(
                rho=float(z["rho"]),
                L3=(eps - pop3) / eps,
                f_star=float(z["f_s"]),
                mu_b=float(z["mu_b"]),
                sig_b=float(z["sig_b"]),
                mu_s=float(z["mu_s"]),
                sig_s=float(z["sig_s"]),
                pop_he1=pop1,
                pop_he3=pop3,
            )
        )
    if not rows:
        return None
    rows.sort(key=lambda d: d["rho"])
    return np.array([tuple(d.values()) for d in rows], dtype=[(k, float) for k in rows[0]])


def load_mw(path: Path) -> np.ndarray:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if float(r["chi"]) == 200.0]
    rows.sort(key=lambda r: float(r["rho"]))
    keys = ("rho", "L3", "pi_out", "mu_in", "mu_out")
    return np.array(
        [tuple(float(r[k]) for k in keys) for r in rows],
        dtype=[(k, float) for k in keys],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=FINE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    mode = load_mode(args.dir)
    if mode is None:
        raise SystemExit(f"no rho*_scfrac.npz in {args.dir}")
    coarse = load_mode(COARSE)
    mw = load_mw(MW) if MW.is_file() else None

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharex=True)
    ax = axes[0, 0]
    ax.plot(mode["rho"], mode["L3"], "o-", color="C0", lw=2, ms=5, label="mode-Laplace (fine)")
    if coarse is not None:
        ax.plot(coarse["rho"], coarse["L3"], "s", color="0.4", ms=7, label="mode-Laplace (4-pt)")
    if mw is not None:
        m = (mw["rho"] >= mode["rho"].min() * 0.9) & (mw["rho"] <= mode["rho"].max() * 1.05)
        ax.plot(mw["rho"][m], mw["L3"][m], "--", color="C3", lw=1.8, label="multiwell χ=N")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel(r"$L_3=(\varepsilon-\mathrm{pop}_{He3})/\varepsilon$")
    ax.set_title("learnability bump?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(mode["rho"], mode["f_star"], "o-", color="C0", lw=2, ms=5)
    if coarse is not None:
        ax.plot(coarse["rho"], coarse["f_star"], "s", color="0.4", ms=7)
    if mw is not None:
        m = (mw["rho"] >= mode["rho"].min() * 0.9) & (mw["rho"] <= mode["rho"].max() * 1.05)
        ax.plot(mw["rho"][m], mw["pi_out"][m], "--", color="C3", lw=1.8, label=r"multiwell $\pi_{out}$")
        ax.legend(frameon=False, fontsize=8)
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel(r"$f^\star$ (specialist fraction)")
    ax.set_title("self-consistent outer mass")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(mode["rho"], mode["mu_b"], "o-", color="C2", lw=2, ms=5, label=r"bulk $\mu$")
    ax.plot(mode["rho"], mode["mu_s"], "o-", color="C1", lw=2, ms=5, label=r"spec $\mu$")
    ax.fill_between(
        mode["rho"],
        mode["mu_b"] - mode["sig_b"],
        mode["mu_b"] + mode["sig_b"],
        color="C2",
        alpha=0.15,
    )
    ax.fill_between(
        mode["rho"],
        mode["mu_s"] - mode["sig_s"],
        mode["mu_s"] + mode["sig_s"],
        color="C1",
        alpha=0.15,
    )
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel(r"well location")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("well centers (±σ band)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(mode["rho"], mode["pop_he1"], "o-", color="C4", lw=2, ms=5, label="pop He1")
    ax.plot(mode["rho"], mode["pop_he3"], "o-", color="C5", lw=2, ms=5, label="pop He3")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel("population mode coeff")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("induced He1 / residual He3")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    for a in axes.ravel():
        a.set_xscale("log")
        a.set_xlim(mode["rho"].min() * 0.9, mode["rho"].max() * 1.05)

    fig.suptitle(
        r"Pure-cubic $\chi=N$ mode-Laplace: fine $\rho$ scan around $0.03$"
        "\n"
        r"(bump in 4-pt $L_3$ was mode-Laplace; multiwell overlay for comparison)",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.out or (args.dir / "L3_fine_vs_rho.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")

    # also dump a merged summary csv
    sum_path = args.dir / "summary_fine.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["rho", "L3", "f_star", "mu_b", "sig_b", "mu_s", "sig_s", "pop_he1", "pop_he3"]
        )
        for row in mode:
            w.writerow([row[k] for k in mode.dtype.names])
    print(f"wrote {sum_path}  ({len(mode)} points)")
    print(f"{'rho':>8} {'L3':>8} {'f*':>8} {'mu_b':>8} {'mu_s':>8}")
    for row in mode:
        print(
            f"{row['rho']:8.4g} {row['L3']:8.4f} {row['f_star']:8.4f} "
            f"{row['mu_b']:8.3f} {row['mu_s']:8.3f}"
        )


if __name__ == "__main__":
    main()
