#!/usr/bin/env python3
"""Learnability vs α with outer-well occupation — shows L̃ can stay high without outer wells."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=float, default=10.0)
    args = ap.parse_args()

    rows = np.genfromtxt(args.summary, delimiter=",", names=True, dtype=None, encoding=None)
    P = np.asarray(rows["P"], dtype=float)
    ρp = np.asarray(rows["rho_p"], dtype=float)
    λ1 = np.asarray(rows["lam_lin"], dtype=float)
    λ2 = np.asarray(rows["lam_cub"], dtype=float)
    L2 = np.asarray(rows["L3"], dtype=float)
    L1 = λ1 / (λ1 + ρp)
    π_out = np.asarray(rows["pi_out"], dtype=float)
    alpha = np.log(P) / np.log(args.d)

    order = np.argsort(alpha)
    alpha, P, L1, L2, λ1, λ2, ρp, π_out = (
        a[order] for a in (alpha, P, L1, L2, λ1, λ2, ρp, π_out)
    )

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.8, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    # --- learnability ---
    ax.plot(alpha, L2, "o-", color="#1b9e77", lw=1.9, ms=7,
            label=r"$\tilde L_2=\lambda_2/(\lambda_2+\rho')$")
    ax.plot(alpha, L1, "s--", color="#d95f02", lw=1.3, ms=5,
            label=r"$\tilde L_1=\lambda_1/(\lambda_1+\rho')$")
    # shade outer-occupied mid-ramp
    outer = π_out > 0.5
    if np.any(outer):
        a0 = alpha[outer].min() - 0.05
        a1 = alpha[outer].max() + 0.05
        ax.axvspan(a0, a1, color="#7570b3", alpha=0.12, label="outer wells occupied")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("learnability")
    ax.set_title(
        r"Eigenframe $\tilde y=(0,\varepsilon)$: $\tilde L_2$ stays high after outer wells lift"
        "\n"
        r"(lab $y=U\tilde y\approx(0.2$–$0.3,\ 0.95$–$0.98)$; $L$ tracks $\lambda_2/\rho'$, not well width)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # --- λ₂ vs ρ' and π_out ---
    ax2.plot(alpha, λ2, "o-", color="#1b9e77", lw=1.6, ms=6, label=r"$\lambda_2$")
    ax2.plot(alpha, ρp, "^-", color="#e7298a", lw=1.4, ms=5, label=r"$\rho'=\kappa/(a_0 P)$")
    ax2b = ax2.twinx()
    ax2b.plot(alpha, π_out, "D--", color="#7570b3", lw=1.2, ms=5, label=r"$\pi_{\mathrm{out}}$")
    ax2b.set_ylim(-0.05, 1.15)
    ax2b.set_ylabel(r"$\pi_{\mathrm{out}}$", color="#7570b3")
    ax2.set_xlabel(rf"$\alpha$  ($P=d^{{\alpha}}$, $d={args.d:g}$)")
    ax2.set_ylabel(r"$\lambda_2,\ \rho'$")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    # combined legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved", out)

    # annotate CSV with regime
    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("alpha,P,rho_p,lam2,Ltilde2,pi_out,regime\n")
        for i in range(len(P)):
            regime = "outer" if π_out[i] > 0.5 else "inner"
            f.write(f"{alpha[i]:.10g},{P[i]:.10g},{ρp[i]:.10g},{λ2[i]:.10g},"
                    f"{L2[i]:.10g},{π_out[i]:.10g},{regime}\n")
    print("saved", csv)

if __name__ == "__main__":
    main()
