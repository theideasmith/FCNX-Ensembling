"""
Shared run bookkeeping for the samplers in this milestone.

A run lives in runs/<run_id>/ with config.json, trace.csv, obs.csv, samples/,
checkpoint.pt and summary.json; lifecycle events go to runs/registry.jsonl (runs.py).
"""

import csv
import json
import math
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from runs import log_event

OBS_KEYS = ["w0_sq", "w0_abs", "w0_kurt", "frac_aligned", "wperp_sq", "A_sq_over_sa2", "train_loss", "L3",
            "L1_he1", "test_mse"]


def git_info():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=root, text=True).strip())
    except Exception:
        commit, dirty = None, None
    return {"commit": commit, "milestone_dirty": dirty}


class RunRecorder:
    def __init__(self, run_dir, cfg, start_iter=0):
        self.run_dir = Path(run_dir)
        self.runs_root = self.run_dir.parent
        self.cfg = cfg
        self.run_id = cfg["run_id"]
        self.start_iter = start_iter
        self._files = []

    @classmethod
    def create(cls, runs_root, run_id, args, kind):
        run_dir = Path(runs_root) / run_id
        (run_dir / "samples").mkdir(parents=True, exist_ok=False)
        cfg = {
            "run_id": run_id,
            "kind": kind,
            "created": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "command": " ".join([sys.executable] + sys.argv),
            "git": git_info(),
            "args": vars(args),
        }
        rec = cls(run_dir, cfg)
        rec.save_config()
        return rec

    @classmethod
    def load(cls, run_dir):
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        return cls(run_dir, cfg)

    def save_config(self):
        (self.run_dir / "config.json").write_text(json.dumps(self.cfg, indent=2))

    def open_tables(self, trace_header, obs_header, append):
        mode = "a" if append else "w"
        self._trace_f = open(self.run_dir / "trace.csv", mode, newline="")
        self._obs_f = open(self.run_dir / "obs.csv", mode, newline="")
        self._files = [self._trace_f, self._obs_f]
        self.trace_w = csv.writer(self._trace_f)
        self.obs_w = csv.writer(self._obs_f)
        if not append:
            self.trace_w.writerow(trace_header)
            self.obs_w.writerow(obs_header)

    def write_obs(self, it, burn, ob):
        """ob: dict of per-chain tensors."""
        C = next(iter(ob.values())).shape[0]
        for c in range(C):
            self.obs_w.writerow([it, c, int(it < burn)] + [f"{ob[k][c].item():.6g}" for k in OBS_KEYS])

    def save_sample(self, it, **tensors):
        torch.save({"iter": it, **{k: v.detach().cpu() for k, v in tensors.items()}},
                   self.run_dir / "samples" / f"iter_{it:08d}.pt")

    def flush(self):
        for f in self._files:
            f.flush()

    def close(self):
        for f in self._files:
            f.close()
        self._files = []

    def event(self, status, **fields):
        log_event(self.runs_root, self.run_id, status, **fields)

    def obs_summary(self):
        """Post-burn means ± stderr (pooled) and per-chain means from obs.csv."""
        with open(self.run_dir / "obs.csv") as f:
            obs = [r for r in csv.DictReader(f) if r["burn"] == "0"]
        chains = sorted({int(r["chain"]) for r in obs})

        def col(rows, k):
            return np.array([float(r[k]) for r in rows])

        pooled = {k: [float(col(obs, k).mean()), float(col(obs, k).std(ddof=1) / math.sqrt(max(len(obs), 1))),
                      len(obs)] for k in OBS_KEYS}
        per_chain = {k: [float(col([r for r in obs if int(r["chain"]) == c], k).mean()) for c in chains]
                     for k in OBS_KEYS}
        return pooled, per_chain, len(obs)

    def trace_column(self, key, from_iter=0):
        with open(self.run_dir / "trace.csv") as f:
            rows = list(csv.DictReader(f))
        return np.array([float(r[key]) for r in rows if int(r["iter"]) >= from_iter and r[key] != ""])

    def write_summary(self, summary):
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
