#!/usr/bin/env python3
"""
Where is the RubinRegime problem highly learnable? Mean-field theory (readin_action_theory.py) only.

1. Along the α/β invariant ray of red_robin_alpha_beta_invariant_steepwell.py, started at our base point
   (d0=150, N0=700, P0=3000, sa0=8/700, κ0=0.07, σ_w²=0.5, ε=-0.3):
       d = d0 β^{1/2},  N = N0 β^{1/4},  P = P0 α^{3/4},  sa0 = sa0_0 β^{-3/4},  κ = κ0 (α/β)^{3/4},  α = β^γ.
   λ = κ/(P sa0) and Ĉ ∝ sa0 P²/(N d κ²) are invariant, so only β_th = d/2σ_w² (well depth) moves.
2. Off the ray, at the base d, N, sa0: L1 and L3 over (κ, P), i.e. over (λ, Ĉ).

    python theory_learnability_scan.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from readin_action_theory import Kernel, analyse, reduced_params, solve_loop

D0, N0, P0, SA0, K0, S0, EPS = 150, 700, 3000, 8 / 700, 0.07, 0.5, -0.3


def theory(d, N, P, sa0, kappa, s0=S0, eps=EPS, chi=1.0):
    T, beta, lam, Chat = reduced_params(N, d, s0, sa0, P, kappa, chi)
    K = Kernel(T, beta)
    D, res = solve_loop(Chat, lam, eps, K)
    out = analyse(D, lam, eps, K)
    return dict(L1=out["L1"], L3=out["L3"], D=D, lam=lam, Chat=Chat, res=res,
                nb=len(out["basins"]), basins=out["basins"])


def ray_point(beta, gamma=1.25):
    alpha = beta ** gamma
    return dict(d=D0 * beta ** 0.5, N=N0 * beta ** 0.25, P=P0 * alpha ** 0.75,
                sa0=SA0 * beta ** -0.75, kappa=K0 * (alpha / beta) ** 0.75)


def main():
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(exist_ok=True)

    print("1. α/β ray from the base point (γ = 1.25)")
    print(f"{'β':>6s} {'d':>6s} {'N':>6s} {'P':>8s} {'κ':>7s} {'λ':>8s} {'Ĉ':>8s} {'L1':>7s} {'L3':>7s}  basins")
    for beta in (1, 2, 4, 8, 16):
        r = ray_point(beta)
        t = theory(**r)
        print(f"{beta:>6g} {r['d']:>6.0f} {r['N']:>6.0f} {r['P']:>8.0f} {r['kappa']:>7.4f} {t['lam']:>8.4f} "
              f"{t['Chat']:>8.3f} {t['L1']:>7.4f} {t['L3']:>7.4f}  "
              + ", ".join(f"{b['basin']}={b['weight']:.2f}" for b in t["basins"]))

    print("\n2. (κ, P) at d=150, N=700, sa0=8/700")
    kappas = np.geomspace(0.005, 0.3, 19)
    Ps = np.geomspace(1500, 48000, 16)
    L1 = np.zeros((len(kappas), len(Ps)))
    L3 = np.zeros_like(L1)
    for i, k in enumerate(kappas):
        for j, P in enumerate(Ps):
            t = theory(D0, N0, P, SA0, k)
            L1[i, j], L3[i, j] = t["L1"], t["L3"]
    np.savez(out_dir / "theory_learnability_scan.npz", kappas=kappas, Ps=Ps, L1=L1, L3=L3)

    print(f"{'κ \\ P':>7s} " + " ".join(f"{P:>6.0f}" for P in Ps[::3]))
    for i in range(0, len(kappas), 2):
        print(f"{kappas[i]:>7.4f} " + " ".join(f"{L3[i, j]:>6.3f}" for j in range(0, len(Ps), 3)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, L, name in zip(axes, (L1, L3), ("L1", "L3")):
        im = ax.pcolormesh(Ps, kappas, L, shading="auto", cmap="viridis", vmin=0, vmax=1)
        cs = ax.contour(Ps, kappas, L, levels=[0.5, 0.8, 0.9, 0.95], colors="w", linewidths=1)
        ax.clabel(cs, fontsize=9)
        ax.plot(P0, K0, "r*", ms=14, label="current point")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$P$", fontsize=13)
        ax.set_ylabel(r"$\kappa$", fontsize=13)
        ax.set_title(f"theory {name}", fontsize=13)
        fig.colorbar(im, ax=ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(rf"$d={D0}$, $N={N0}$, $\sigma_w^2={S0}$, $\sigma_a^2=8/N$, $\epsilon={EPS}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "theory_learnability_scan.png", dpi=150)
    print(f"\nsaved {out_dir / 'theory_learnability_scan.png'}")


if __name__ == "__main__":
    main()
