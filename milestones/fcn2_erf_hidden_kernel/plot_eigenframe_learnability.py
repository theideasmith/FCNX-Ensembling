#!/usr/bin/env python3
"""Plot eigenframe mode learnabilities L̃_k = λ_k / (λ_k + ρ')."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = np.genfromtxt(args.summary, delimiter=",", names=True, dtype=None, encoding=None)
    P = np.asarray(rows["P"], dtype=float)
    ρp = np.asarray(rows["rho_p"], dtype=float)
    λ1 = np.asarray(rows["lam_lin"], dtype=float)
    λ2 = np.asarray(rows["lam_cub"], dtype=float)
    L1 = λ1 / (λ1 + ρp)
    L2 = λ2 / (λ2 + ρp)
    # Prefer stored L3 if present (same as L2)
    if "L3" in rows.dtype.names:
        L2 = np.asarray(rows["L3"], dtype=float)

    order = np.argsort(P)
    P, L1, L2, λ1, λ2, ρp = (a[order] for a in (P, L1, L2, λ1, λ2, ρp))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(P, L2, "o-", color="#1b9e77", lw=1.8, ms=6,
            label=r"$\tilde L_2=\lambda_2/(\lambda_2+\rho')$  (2nd eigenmode)")
    ax.plot(P, L1, "s--", color="#d95f02", lw=1.4, ms=5,
            label=r"$\tilde L_1=\lambda_1/(\lambda_1+\rho')$  (1st eigenmode)")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$P$")
    ax.set_ylabel("learnability")
    ax.set_title(r"Eigenframe learnabilities (mixed He1–He3 modes), $\tilde y=(0,\varepsilon)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)

    # also write a clean learnability CSV
    out_csv = Path(args.out).with_suffix(".csv")
    with open(out_csv, "w") as f:
        f.write("P,rho_p,lam1,lam2,Ltilde1,Ltilde2\n")
        for i in range(len(P)):
            f.write(f"{P[i]:.10g},{ρp[i]:.10g},{λ1[i]:.10g},{λ2[i]:.10g},{L1[i]:.10g},{L2[i]:.10g}\n")
    print("saved", out_csv)

if __name__ == "__main__":
    main()
