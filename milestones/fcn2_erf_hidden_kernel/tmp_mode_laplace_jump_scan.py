#!/usr/bin/env python3
"""Fine mode-Laplace scan of the f* jump near ρ≈0.02.

For each ρ: dense g(f) at fixed f (find all roots), then the usual solve_sc_root
pick. Writes summary + per-ρ g(f) tables under mode_laplace_jump/.

Usage:
  ITERS=35 N_BISECT=8 python -u tmp_mode_laplace_jump_scan.py
  RHOS=0.022,0.02,0.018 FS=0.1,0.2,0.3,0.5,0.7 python -u tmp_mode_laplace_jump_scan.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_jump"

os.environ.setdefault("D", "16")
os.environ.setdefault("SELF", "full")
os.environ.setdefault("M_DAMP", "0.1")
os.environ.setdefault("FIX_TRACE", "1")
os.environ.setdefault("BULK_LO", "0.0")
os.environ.setdefault("BULK_HI", "1.0")
os.environ.setdefault("SPEC_LO", "1.0")
os.environ.setdefault("SPEC_HI", "5.5")
os.environ.setdefault("MU_B0", "0.45")
os.environ.setdefault("SIG_B0", "0.10")
os.environ.setdefault("MU_S0", "3.0")
os.environ.setdefault("SIG_S0", "0.25")
os.environ.setdefault("LAPLACE_DEV", os.environ.get("LAPLACE_DEV", "cuda:1"))

ITERS = int(os.environ.get("ITERS", "35"))
N_BISECT = int(os.environ.get("N_BISECT", "8"))
sys.argv = [sys.argv[0], "none", str(ITERS)]

import tmp_mode_laplace as T  # noqa: E402
import tmp_mode_laplace_pure_cubic_rho as PC  # noqa: E402


def parse_list(env: str, default: str) -> list[float]:
    raw = os.environ.get(env, default)
    return [float(x) for x in raw.split(",") if x.strip()]


def g_of_f(S, rho: float, f: float) -> dict:
    T.solve_sc(S, f"ρ={rho:g} f={f:.4f}", float(f), f_damp=0.0)
    dlogZ = float(T.LAST["dlogZ"])
    g = dlogZ - math.log(f / (1.0 - f))
    st = T.LAST["st"]
    _, pop = T.LAST["th"].disc()
    return dict(
        rho=rho,
        f=float(f),
        g=g,
        dlogZ=dlogZ,
        mu_b=float(st["b"]["mu"]),
        sig_b=float(st["b"]["sig"]),
        mu_s=float(st["s"]["mu"]),
        sig_s=float(st["s"]["sig"]),
        pop_he1=float(pop[0]),
        pop_he3=float(pop[1]),
        L3=1.0 - float(pop[1]),
        f_target=1.0 / (1.0 + math.exp(min(-dlogZ, 700.0))),
    )


def find_roots(rows: list[dict]) -> list[dict]:
    """Log-linear interpolate every g sign change."""
    roots = []
    for i in range(len(rows) - 1):
        g0, g1 = rows[i]["g"], rows[i + 1]["g"]
        if g0 == 0:
            roots.append({**rows[i], "f_star": rows[i]["f"], "bracket": "exact"})
            continue
        if g0 * g1 >= 0:
            continue
        f0, f1 = rows[i]["f"], rows[i + 1]["f"]
        t = g0 / (g0 - g1)
        f_star = math.exp(math.log(f0) + t * (math.log(f1) - math.log(f0)))
        # blend nearby observables
        blend = lambda k: (1 - t) * rows[i][k] + t * rows[i + 1][k]
        roots.append(
            dict(
                rho=rows[i]["rho"],
                f_star=f_star,
                L3=blend("L3"),
                mu_b=blend("mu_b"),
                mu_s=blend("mu_s"),
                pop_he3=blend("pop_he3"),
                g_lo=g0,
                g_hi=g1,
                f_lo=f0,
                f_hi=f1,
                bracket=f"[{f0:.3f},{f1:.3f}]",
            )
        )
    return roots


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rhos = parse_list(
        "RHOS",
        "0.024,0.022,0.021,0.0205,0.02,0.0195,0.019,0.0185,0.018,0.017,0.016,0.015,0.014",
    )
    fs = parse_list(
        "FS",
        "0.08,0.12,0.16,0.20,0.24,0.26,0.28,0.30,0.35,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85",
    )
    f_lo = float(os.environ.get("F_LO", "0.01"))
    f_hi = float(os.environ.get("F_HI", "0.85"))

    all_g_rows: list[dict] = []
    root_rows: list[dict] = []
    pick_rows: list[dict] = []

    for rho in rhos:
        print(f"\n######## ρ={rho:g} ########", flush=True)
        # specialist init farther out at lower ρ (matches pure-cubic script)
        if rho <= 0.03:
            os.environ["SPEC_HI"] = "5.5"
            os.environ["MU_S0"] = "3.2"
        else:
            os.environ["SPEC_HI"] = "4.5"
            os.environ["MU_S0"] = "2.8"
        PC.configure(rho, d=16, P=1000, N=200, chi=200, sa0=0.03, eps=1.0, he1=0.0, s0=1.0)
        S = torch.eye(T.NM, device=T.DEV, dtype=T.DT)

        rows = []
        for f in fs:
            row = g_of_f(S, rho, f)
            rows.append(row)
            all_g_rows.append(row)
            print(
                f"  f={row['f']:.3f}  g={row['g']:+8.3f}  L3={row['L3']:.4f}  "
                f"μb={row['mu_b']:.3f}  μs={row['mu_s']:.3f}",
                flush=True,
            )
        with (OUT / f"gscan_rho{str(rho).replace('.','p')}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        roots = find_roots(rows)
        print(f"  → {len(roots)} root(s):", flush=True)
        for k, rt in enumerate(roots):
            print(
                f"     branch{k}: f*≈{rt['f_star']:.4f}  L3≈{rt['L3']:.4f}  "
                f"μs≈{rt['mu_s']:.3f}  {rt['bracket']}",
                flush=True,
            )
            root_rows.append({**rt, "branch": k, "n_roots": len(roots)})

        # what the production bisector picks
        out_sc = T.solve_sc_root(
            S, f"jump-scan ρ={rho:g}", lo=f_lo, hi=f_hi, n_bisect=N_BISECT
        )
        if out_sc is None:
            f_star, tr, pop = T.solve_sc(S, f"jump-scan ρ={rho:g} fallback", 0.08)
            src = "fallback-iter"
        else:
            f_star, tr, pop = out_sc
            src = "bisect"
        st = T.LAST["st"]
        pick = dict(
            rho=rho,
            f_star=float(f_star),
            L3=1.0 - float(pop[1]),
            mu_b=float(st["b"]["mu"]),
            sig_b=float(st["b"]["sig"]),
            mu_s=float(st["s"]["mu"]),
            sig_s=float(st["s"]["sig"]),
            pop_he1=float(pop[0]),
            pop_he3=float(pop[1]),
            n_roots=len(roots),
            source=src,
        )
        pick_rows.append(pick)
        print(
            f"  PICK ({src}): f*={pick['f_star']:.4f}  L3={pick['L3']:.4f}  "
            f"μb={pick['mu_b']:.3f}  μs={pick['mu_s']:.3f}  n_roots={len(roots)}",
            flush=True,
        )

    # summaries
    with (OUT / "gscan_all.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_g_rows[0].keys()))
        w.writeheader()
        w.writerows(all_g_rows)
    if root_rows:
        with (OUT / "roots_vs_rho.csv").open("w", newline="") as fh:
            keys = list(root_rows[0].keys())
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(root_rows)
    with (OUT / "bisect_pick_vs_rho.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pick_rows[0].keys()))
        w.writeheader()
        w.writerows(pick_rows)

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))

    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for i, rho in enumerate(rhos):
        sub = [r for r in all_g_rows if abs(r["rho"] - rho) < 1e-12]
        c = cmap(i / max(len(rhos) - 1, 1))
        ax.plot([r["f"] for r in sub], [r["g"] for r in sub], "-o", ms=3, color=c, lw=1.4, label=f"{rho:g}")
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("f (specialist fraction)")
    ax.set_ylabel("g(f)")
    ax.set_title(r"$g(f)=\log(Z_s/Z_b)-\log(f/(1-f))$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-80, 80)

    ax = axes[0, 1]
    for b in sorted({r["branch"] for r in root_rows}):
        sub = [r for r in root_rows if r["branch"] == b]
        sub = sorted(sub, key=lambda r: r["rho"])
        ax.plot(
            [r["rho"] for r in sub],
            [r["f_star"] for r in sub],
            "o-",
            ms=5,
            lw=1.8,
            label=f"root branch {b}",
        )
    ax.plot(
        [r["rho"] for r in pick_rows],
        [r["f_star"] for r in pick_rows],
        "kx--",
        ms=7,
        lw=1.2,
        label="bisect pick",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$f^\star$")
    ax.set_title("all g(f)=0 roots vs bisect pick")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for b in sorted({r["branch"] for r in root_rows}):
        sub = sorted([r for r in root_rows if r["branch"] == b], key=lambda r: r["rho"])
        ax.plot([r["rho"] for r in sub], [r["L3"] for r in sub], "o-", ms=5, lw=1.8, label=f"root {b}")
    ax.plot(
        [r["rho"] for r in pick_rows],
        [r["L3"] for r in pick_rows],
        "kx--",
        ms=7,
        lw=1.2,
        label="bisect pick",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$L_3$")
    ax.set_title("learnability on each root")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for b in sorted({r["branch"] for r in root_rows}):
        sub = sorted([r for r in root_rows if r["branch"] == b], key=lambda r: r["rho"])
        ax.plot([r["rho"] for r in sub], [r["mu_s"] for r in sub], "o-", ms=5, lw=1.8, label=f"μs root {b}")
        ax.plot([r["rho"] for r in sub], [r["mu_b"] for r in sub], "s--", ms=4, lw=1.2, alpha=0.8, label=f"μb root {b}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("well μ")
    ax.set_title("well locations on each root")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Mode-Laplace $f^\star$ jump near $\rho\approx 0.02$: multi-root structure of $g(f)$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "jump_detail.png", dpi=160, bbox_inches="tight")
    print(f"\nsaved {OUT / 'jump_detail.png'}", flush=True)

    (OUT / "README.md").write_text(
        "# Mode-Laplace f* jump detail (ρ≈0.02)\n\n"
        "Dense `g(f)` scans across the ρ window where the specialist-fraction "
        "bisector jumps from `f*≈0.26` to `f*≳0.6`.\n\n"
        "- `gscan_all.csv` / `gscan_rho*.csv` — fixed-f solves\n"
        "- `roots_vs_rho.csv` — every sign-change root of `g(f)=0`\n"
        "- `bisect_pick_vs_rho.csv` — what `solve_sc_root` (F_LO..F_HI) selects\n"
        "- `jump_detail.png` — g(f) curves + branch L3 / f* / wells\n\n"
        "Reproduce:\n"
        "```bash\n"
        "ITERS=35 N_BISECT=8 python -u tmp_mode_laplace_jump_scan.py\n"
        "```\n"
    )
    print(f"done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
