"""Section 8 equilibration test (snapshot_history_crossings) on pure-cubic χ=N ρ-ray."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE / (
    "red_robin_pure_cubic_rho_ray"
    "_d16_N200_P1000_chiN_sa00.03_eps1_he10"
    "_lr0.01_ep60M_sched2_3_5"
)
OUT_DIR = HERE / "action_h0_activation_plots/PureCubicChiN/equilibration"
DIR_RE = re.compile(r"purecubic_rho(?P<rho>[\d.eE+-]+)_")

# Same hysteresis cut as Section 8 / snapshot_history_crossings.py
IN_T, OUT_T = 0.5, 1.0
OUTER_FRAC_CUT = 0.6


def analyze(run: Path, rho: float):
    snaps = sorted((run / "A_snapshots").glob("epoch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if len(snaps) < 3:
        raise FileNotFoundError(f"too few snaps in {run}")
    ep = np.array([int(p.stem.split("_")[-1]) for p in snaps])
    cfg = json.loads((run / "config.json").read_text())
    d = int(cfg["d"])
    W = np.stack([
        torch.load(p, map_location="cpu", weights_only=False)["W0"]
        .double().reshape(-1, d).numpy()
        for p in snaps
    ])
    w0 = W[:, :, 0]
    a = np.abs(w0)
    wp2 = (W[:, :, 1:] ** 2).sum(2)
    T, N = w0.shape

    fo = (a > OUTER_FRAC_CUT).mean(1)
    print(f"\n=== ρ={rho:g}  d={d} N={N}  snaps={T} epochs {ep[0]}..{ep[-1]} "
          f"(Δ={ep[1]-ep[0]}) ===", flush=True)
    print(f"  outer fraction (|w0|>{OUTER_FRAC_CUT}): "
          f"first 10 {fo[:10].mean():.4f}, last 10 {fo[-10:].mean():.4f}, "
          f"min {fo.min():.4f} max {fo.max():.4f}, "
          f"drift last-first {fo[-10:].mean()-fo[:10].mean():+.4f}", flush=True)

    state = np.where(a[0] < IN_T, 0, np.where(a[0] > OUT_T, 1, -1))
    i2o = o2i = 0
    for k in range(1, T):
        s = np.where(a[k] < IN_T, 0, np.where(a[k] > OUT_T, 1, -1))
        i2o += int(((state == 0) & (s == 1)).sum())
        o2i += int(((state == 1) & (s == 0)).sum())
        state = np.where(s >= 0, s, state)
    ever_in, ever_out = (a < IN_T).any(0), (a > OUT_T).any(0)
    both = int((ever_in & ever_out).sum())
    print(f"  crossings hysteresis ({IN_T}/{OUT_T}): "
          f"inner→outer {i2o}, outer→inner {o2i}; "
          f"neurons seen both: {both}/{N}", flush=True)

    # Also with cut adapted to outer-well location (mid between bulk/spec)
    # use OUT at 1.2 for clearer outer at low rho
    for out_t in (1.0, 1.5, 2.0):
        state = np.where(a[0] < IN_T, 0, np.where(a[0] > out_t, 1, -1))
        i2o2 = o2i2 = 0
        for k in range(1, T):
            s = np.where(a[k] < IN_T, 0, np.where(a[k] > out_t, 1, -1))
            i2o2 += int(((state == 0) & (s == 1)).sum())
            o2i2 += int(((state == 1) & (s == 0)).sum())
            state = np.where(s >= 0, s, state)
        print(f"  crossings ({IN_T}/{out_t}): i→o {i2o2}, o→i {o2i2}", flush=True)

    stay_in = (a < OUTER_FRAC_CUT).all(0)
    flips = int((np.sign(w0[1:, stay_in]) != np.sign(w0[:-1, stay_in])).sum()) if stay_in.any() else 0
    print(f"  always-inner neurons: {int(stay_in.sum())}; "
          f"sign flips of w0 between snaps: {flips}", flush=True)
    if stay_in.sum() > 5:
        xs = w0[:, stay_in]
        var = (xs - xs.mean()).var()
        lags = [1, 2, 5, 10, min(20, T - 1)]
        ac = []
        for l in lags:
            if l < T and var > 0:
                ac.append(float(((xs[l:] - xs.mean()) * (xs[:-l] - xs.mean())).mean() / var))
            else:
                ac.append(float("nan"))
        print("  autocorr w0 (always-inner), lag snaps: " +
              ", ".join(f"{l}:{c:.3f}" for l, c in zip(lags, ac)), flush=True)
        print(f"  inner |w_perp|^2: first10 {wp2[:10][:, stay_in].mean():.3f}, "
              f"last10 {wp2[-10:][:, stay_in].mean():.3f}", flush=True)

    # Split-half histogram KS / action agreement (second diagnostic)
    mid = T // 2
    h1, e = np.histogram(w0[:mid].ravel(), bins=40, density=True, range=(-5, 5))
    h2, _ = np.histogram(w0[mid:].ravel(), bins=e, density=True)
    # L1 distance between densities
    l1 = 0.5 * np.abs(h1 - h2).sum() * (e[1] - e[0])
    print(f"  split-half density L1/2 (TV): {l1:.4f}  "
          f"(0=identical first/second half of snaps)", flush=True)

    verdict = "EQUILIBRATED" if (i2o > 0 and o2i > 0 and abs(fo[-10:].mean() - fo[:10].mean()) < 0.03) else (
        "ONE-WAY / STALLED" if (i2o + o2i > 0 and min(i2o, o2i) == 0) else (
            "NO CROSSINGS" if i2o + o2i == 0 else "MIXED / DRIFTING"
        )
    )
    print(f"  verdict (Sec.8 style): {verdict}", flush=True)
    return dict(rho=rho, ep=ep, fo=fo, w0=w0, a=a, stay_in=stay_in,
                i2o=i2o, o2i=o2i, both=both, verdict=verdict, l1=l1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted((SWEEP / "models").iterdir()):
        m = DIR_RE.match(d.name)
        if m and (d / "A_snapshots").is_dir():
            runs.append((float(m.group("rho")), d))
    runs.sort(key=lambda t: -t[0])
    results = [analyze(p, rho) for rho, p in runs]

    fig, axes = plt.subplots(2, len(results), figsize=(3.6 * len(results), 6.2), squeeze=False)
    rng = np.random.default_rng(0)
    for ax_col, r in zip(axes.T, results):
        ep, w0, a, stay_in = r["ep"], r["w0"], r["a"], r["stay_in"]
        ax0, ax1 = ax_col
        nin = np.where(stay_in)[0]
        if nid := len(nin):
            for j in rng.choice(nin, min(25, nid), replace=False):
                ax0.plot(ep, w0[:, j], lw=0.5, color="0.55", alpha=0.7)
        outer_idx = np.where((a > OUTER_FRAC_CUT).any(0))[0]
        for j in outer_idx[:25]:
            ax0.plot(ep, w0[:, j], lw=0.7)
        ax0.set_title(rf"$\rho={r['rho']:.3g}$: {r['verdict']}"
                      f"\ni→o={r['i2o']}, o→i={r['o2i']}, both={r['both']}", fontsize=8)
        ax0.set_xlabel("epoch"); ax0.set_ylabel(r"$w_0$")
        ax0.axhline(OUT_T, color="C3", ls="--", lw=0.7); ax0.axhline(-OUT_T, color="C3", ls="--", lw=0.7)
        ax0.axhline(IN_T, color="C0", ls=":", lw=0.7); ax0.axhline(-IN_T, color="C0", ls=":", lw=0.7)
        ax1.plot(ep, r["fo"], color="k")
        ax1.axhline(r["fo"][:10].mean(), color="0.5", ls=":", lw=0.8, label="first10 mean")
        ax1.axhline(r["fo"][-10:].mean(), color="C3", ls="--", lw=0.8, label="last10 mean")
        ax1.set_xlabel("epoch"); ax1.set_ylabel(rf"outer frac $|w_0|>{OUTER_FRAC_CUT}$")
        ax1.set_title(f"TV(first/second half)={r['l1']:.3f}", fontsize=8)
        ax1.legend(fontsize=6, loc="best")
        ax1.set_ylim(0, max(0.2, 1.15 * float(r["fo"].max())))
    fig.suptitle(
        "Pure cubic χ=N ρ-ray: Section 8 equilibration test "
        f"(hysteresis {IN_T}/{OUT_T}; grey=always-inner, color=ever-outer)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "snapshot_history_crossings.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}", flush=True)

    # summary csv
    import csv
    with (OUT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "i2o", "o2i", "both", "fo_first10", "fo_last10", "tv_split", "verdict"])
        for r in results:
            w.writerow([r["rho"], r["i2o"], r["o2i"], r["both"],
                        float(r["fo"][:10].mean()), float(r["fo"][-10:].mean()),
                        r["l1"], r["verdict"]])
    print(f"wrote {OUT_DIR/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
