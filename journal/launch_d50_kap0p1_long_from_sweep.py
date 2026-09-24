#!/usr/bin/env python3
"""After d=50 T=0.2 (κ=0.1) 50k P-sweep: pick P≈He3 0.2 and 0.5, launch 50M runs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/akiva/FCNX-Ensembling/journal/LearningCubic_models")
LAUNCH_SH = Path(
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/run_d50_ep50M_schedule.sh"
)
STATE_PATH = OUT_DIR / "d50_kap0p1_long_launch_state.json"
TARGETS = (0.2, 0.5)


def completed_histories(T: float = 0.2, N: int = 400, epochs: int = 50_000):
    pat = f"langevin_mf_d50_N{N}_P*_chi{N}_lr0.01_T{T}_eps0.5_seed42_ep{epochs}_history.npz"
    rows = []
    for path in sorted(OUT_DIR.glob(pat)):
        # ..._P{P}_chi...
        name = path.name
        try:
            p_str = name.split("_P")[1].split("_")[0]
            P = int(p_str)
        except Exception:
            continue
        hist = dict(np.load(path))
        if int(hist["epoch"][-1]) < epochs:
            continue
        rows.append(
            {
                "P": P,
                "he3": float(hist["he3_test"][-1]),
                "test_mse": float(hist["test_mse"][-1]),
                "path": str(path),
            }
        )
    rows.sort(key=lambda r: r["P"])
    return rows


def pick_nearest(rows, target: float):
    """Prefer P with He3 closest to target among learnable (He3>0.05) points."""
    learnable = [r for r in rows if r["he3"] >= 0.05]
    pool = learnable if learnable else rows
    return min(pool, key=lambda r: abs(r["he3"] - target))


def analyze(rows):
    print("=== d=50 T=0.2 (κ=0.1) 50k He3 vs P ===", flush=True)
    for r in rows:
        flag = ""
        if r["he3"] >= 0.15:
            flag = "  << learnable"
        print(
            f"  P={r['P']:5d}  He3={r['he3']:+.4f}  test_MSE={r['test_mse']:.4f}{flag}",
            flush=True,
        )
    picks = {}
    used = set()
    for t in TARGETS:
        cand = pick_nearest(rows, t)
        # if same P picked twice, take next-best unused
        if cand["P"] in used:
            alts = sorted(rows, key=lambda r: abs(r["he3"] - t))
            for a in alts:
                if a["P"] not in used:
                    cand = a
                    break
        used.add(cand["P"])
        picks[t] = cand
        print(
            f"pick target He3≈{t}: P={cand['P']}  (measured He3={cand['he3']:+.4f})",
            flush=True,
        )
    return picks


def launch_tmux(P: int, device: str, session: str, T: float = 0.2, N: int = 400):
    env = os.environ.copy()
    env.update(
        {
            "P": str(P),
            "N": str(N),
            "CHI": str(N),
            "T": str(T),
            "DEVICE": device,
        }
    )
    # kill existing session with same name if any
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, capture_output=True)
    cmd = (
        f"cd /home/akiva/FCNX-Ensembling && "
        f"P={P} N={N} CHI={N} T={T} DEVICE={device} {LAUNCH_SH}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True, env=env)
    print(f"launched tmux session {session}: P={P} T={T} N={N} device={device}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-launch even if state exists")
    ap.add_argument("--device0", default="cuda:0")
    ap.add_argument("--device1", default="cuda:1")
    args = ap.parse_args()

    if STATE_PATH.exists() and not args.force and not args.dry_run:
        print(f"already launched (see {STATE_PATH}); pass --force to redo", flush=True)
        print(STATE_PATH.read_text(), flush=True)
        return 0

    rows = completed_histories()
    expected_P = {50, 100, 500, 1000, 2000, 3000, 8000, 16000}
    have = {r["P"] for r in rows}
    missing = sorted(expected_P - have)
    if missing:
        print(f"sweep incomplete; missing P={missing}", flush=True)
        return 2

    picks = analyze(rows)
    # assign: lower P -> device0, higher P -> device1
    ordered = sorted(picks.items(), key=lambda kv: kv[1]["P"])
    launches = []
    for i, (target, row) in enumerate(ordered):
        device = args.device0 if i == 0 else args.device1
        session = f"d50_ep50M_P{row['P']}_T0p2"
        launches.append(
            {
                "target_he3": target,
                "P": row["P"],
                "measured_he3": row["he3"],
                "device": device,
                "session": session,
            }
        )

    if args.dry_run:
        print("dry-run; would launch:", json.dumps(launches, indent=2))
        return 0

    for L in launches:
        launch_tmux(L["P"], L["device"], L["session"])

    state = {"rows": rows, "launches": launches}
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"wrote {STATE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
