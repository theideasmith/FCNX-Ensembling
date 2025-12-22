#!/usr/bin/env python3
"""
Poll eigenvalue outputs from train_fcn2_erf.py and push new epochs to a webhook.site URL.
Usage:
  WEBHOOK_URL="https://webhook.site/<your-id>" python post_eigs_to_webhook.py
Optional env vars:
  RUN_DIR: override the run directory (defaults to the chi=200, lr=5e-7 run).
  POLL_INTERVAL: seconds between polls (default: 30).
State is tracked in .webhook_posted_epochs.json inside the run directory to avoid re-sending.
"""

import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Dict, List, Set

RUN_DIR = pathlib.Path(
    os.environ.get(
        "RUN_DIR",
        pathlib.Path(__file__).parent / "d50_P200_N200_chi_200.0_lr_5e-07_T_2.0",
    )
)
EIG_FILE = RUN_DIR / "eigenvalues_over_time.json"
STATE_FILE = RUN_DIR / ".webhook_posted_epochs.json"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "30"))


def load_posted() -> Set[int]:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            try:
                data = json.load(f)
                return {int(e) for e in data}
            except Exception:
                return set()
    return set()


def save_posted(posted: Set[int]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(sorted(posted), f, indent=2)


def load_eigenvalues() -> Dict[int, List[float]]:
    if not EIG_FILE.exists():
        return {}
    with open(EIG_FILE, "r") as f:
        raw = json.load(f)
    eigs: Dict[int, List[float]] = {}
    for k, v in raw.items():
        try:
            eigs[int(k)] = v
        except ValueError:
            continue
    return eigs


def eigen_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": math.nan, "max": math.nan, "min": math.nan, "std": math.nan}
    n = float(len(vals))
    mean = sum(vals) / n
    variance = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "max": max(vals),
        "min": min(vals),
        "std": std,
    }


def post_epoch(epoch: int, vals: List[float]) -> None:
    if not WEBHOOK_URL:
        raise SystemExit("WEBHOOK_URL env var is required.")
    payload = {
        "run_dir": str(RUN_DIR),
        "run_name": RUN_DIR.name,
        "epoch": epoch,
        "eigenvalues": vals,
        "stats": eigen_stats(vals),
        "count": len(vals),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()  # consume to allow connection reuse


def main() -> None:
    posted = load_posted()
    print(f"Polling {EIG_FILE} every {POLL_INTERVAL}s; already posted {len(posted)} epochs.")
    while True:
        eigs = load_eigenvalues()
        if eigs:
            new_epochs = sorted([e for e in eigs.keys() if e not in posted])
            for epoch in new_epochs:
                try:
                    post_epoch(epoch, eigs[epoch])
                    posted.add(epoch)
                    save_posted(posted)
                    print(f"Posted epoch {epoch} with {len(eigs[epoch])} eigenvalues.")
                except urllib.error.URLError as e:
                    print(f"Failed to post epoch {epoch}: {e}")
                except Exception as e:
                    print(f"Error on epoch {epoch}: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
