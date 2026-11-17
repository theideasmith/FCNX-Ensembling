#!/usr/bin/env python3
"""Collect (w0, q) read-in snapshots from a RUNNING chain started before run_fcn3.py had --snap-every: polls the run's
checkpoint.pt (rewritten every --checkpoint-every iterations) and appends each new state to snaps.npz in the run
folder, in the same format run_fcn3.py writes (iter, w0, q). Exits when summary.json appears.

    python snap_watch.py runs/<run> [runs/<run2> ...]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

runs = [Path(r) for r in sys.argv[1:]]
state = {}
for r in runs:
    sw = math.sqrt(json.loads((r / "config.json").read_text())["args"]["sigw"] ** 2
                   / json.loads((r / "config.json").read_text())["args"]["d"])
    old = dict(np.load(r / "snaps.npz")) if (r / "snaps.npz").exists() else {}
    state[r] = dict(sw=sw, last=-1 if not old else int(old["iter"][-1]), snaps={k: list(v) for k, v in old.items()})
while True:
    done = 0
    for r, s in state.items():
        try:
            ck = torch.load(r / "checkpoint.pt", map_location="cpu", weights_only=False)
        except Exception:
            continue                                     # caught mid-write: next poll
        if ck["iter"] != s["last"]:
            W = ck["ZW"].double() * s["sw"]
            for k, v in (("iter", ck["iter"]), ("w0", W[:, 0].float().numpy()),
                         ("q", (W[:, 1:] ** 2).sum(1).float().numpy())):
                s["snaps"].setdefault(k, []).append(v)
            s["last"] = ck["iter"]
            np.savez(r / "snaps.npz", **{k: np.asarray(v) for k, v in s["snaps"].items()})
            print(f"{r.name}: snapshot at iter {ck['iter']} ({len(s['snaps']['iter'])} total)", flush=True)
        done += (r / "summary.json").exists()
    if done == len(state):
        break
    time.sleep(60)
