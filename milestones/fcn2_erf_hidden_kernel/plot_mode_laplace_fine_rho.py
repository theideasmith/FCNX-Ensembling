#!/usr/bin/env python3
"""Plot fine ρ learnability: matrix continuum multiwell + mode-Laplace overlay."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FINE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine"
COARSE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
MW_FINE = FINE / "summary_multiwell_fine.csv"


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


def load_mw(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows.sort(key=lambda r: float(r["rho"]))
    keys = ("rho", "L3", "pi_out", "mu_in", "mu_out", "sig_in", "sig_out", "dV_raw")
    out = []
    for r in rows:
        out.append(tuple(float(r[k]) for k in keys))
    return np.array(out, dtype=[(k, float) for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=FINE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    mode = load_mode(args.dir)
    coarse = load_mode(COARSE)
    mw = load_mw(args.dir / "summary_multiwell_fine.csv")
    if mw is None:
        mw = load_mw(MW_FINE)
    if mode is None and mw is None:
        raise SystemExit(f"no data in {args.dir}")

    rho_lo = min(
        ([mode["rho"].min()] if mode is not None else [])
        + ([mw["rho"].min()] if mw is not None else [])
        + ([coarse["rho"].min()] if coarse is not None else [])
    )
    rho_hi = max(
        ([mode["rho"].max()] if mode is not None else [])
        + ([mw["rho"].max()] if mw is not None else [])
        + ([coarse["rho"].max()] if coarse is not None else [])
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharex=True)

    ax = axes[0, 0]
    if mw is not None:
        ax.plot(mw["rho"], mw["L3"], "D-", color="C3", lw=2.2, ms=5, label="matrix continuum")
    if mode is not None:
        ax.plot(mode["rho"], mode["L3"], "o-", color="C0", lw=1.6, ms=4, label="mode-Laplace")
    if coarse is not None:
        ax.plot(coarse["rho"], coarse["L3"], "s", color="0.35", ms=7, label="mode 4-pt")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel(r"$L_3$")
    ax.set_title("learnability (no bump in continuum)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if mw is not None:
        ax.plot(mw["rho"], mw["pi_out"], "D-", color="C3", lw=2.2, ms=5, label=r"continuum $\pi_{out}$")
    if mode is not None:
        ax.plot(mode["rho"], mode["f_star"], "o-", color="C0", lw=1.6, ms=4, label=r"mode $f^\star$")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel("outer mass")
    ax.set_title("specialist / outer fraction")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if mw is not None:
        ax.plot(mw["rho"], mw["mu_in"], "D-", color="C2", lw=2.0, ms=5, label=r"continuum $\mu_{in}$")
        ax.plot(mw["rho"], mw["mu_out"], "D-", color="C1", lw=2.0, ms=5, label=r"continuum $\mu_{out}$")
    if mode is not None:
        ax.plot(mode["rho"], mode["mu_b"], "o-", color="C2", lw=1.2, ms=4, alpha=0.7, label=r"mode $\mu_b$")
        ax.plot(mode["rho"], mode["mu_s"], "o-", color="C1", lw=1.2, ms=4, alpha=0.7, label=r"mode $\mu_s$")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel("well location")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("well centers")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if mw is not None:
        ax.plot(mw["rho"], mw["dV_raw"], "D-", color="C3", lw=2.0, ms=5, label=r"$V_{out}-V_{in}$")
        ax.axhline(0.0, color="0.6", ls="--", lw=0.8)
    if mode is not None:
        ax.plot(mode["rho"], mode["pop_he1"], "o-", color="C4", lw=1.4, ms=4, label="mode pop He1")
        ax2 = ax.twinx()
        ax2.plot(mode["rho"], mode["pop_he3"], "o-", color="C5", lw=1.4, ms=4, label="mode pop He3")
        ax2.set_ylabel("mode pop He3", color="C5")
    ax.axvline(0.03, color="0.7", ls=":", lw=1)
    ax.set_ylabel(r"continuum $dV$")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("well depth gap / mode He coeffs")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    for a in axes.ravel():
        a.set_xscale("log")
        a.set_xlim(rho_lo * 0.9, rho_hi * 1.05)

    fig.suptitle(
        r"Pure-cubic $\chi=N$: fine $\rho$ scan around $0.03$"
        "\n"
        r"matrix continuum Laplace (primary) vs mode-Laplace specialist ansatz",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.out or (args.dir / "L3_fine_vs_rho.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")

    if mw is not None:
        print(f"\nmatrix continuum ({len(mw)} pts):")
        print(f"{'rho':>8} {'L3':>8} {'pi':>8} {'mu_in':>8} {'mu_out':>8}")
        for row in mw:
            print(
                f"{row['rho']:8.4g} {row['L3']:8.4f} {row['pi_out']:8.4f} "
                f"{row['mu_in']:8.3f} {row['mu_out']:8.3f}"
            )
        # monotonicity check
        L = mw["L3"]
        dL = np.diff(L)
        # mw sorted ascending rho; learnability should rise as rho falls
        # so as rho increases L3 should decrease → dL < 0
        n_up = int(np.sum(dL > 1e-6))
        print(f"\nL3 vs increasing ρ: {n_up}/{len(dL)} positive steps "
              f"(0 ⇒ strictly decreasing in ρ ⇒ no bump)")

    if mode is not None:
        sum_path = args.dir / "summary_fine.csv"
        with sum_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(mode.dtype.names))
            for row in mode:
                w.writerow([row[k] for k in mode.dtype.names])
        print(f"wrote {sum_path} ({len(mode)} mode points)")


if __name__ == "__main__":
    main()
