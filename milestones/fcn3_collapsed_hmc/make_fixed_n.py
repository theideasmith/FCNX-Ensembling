#!/usr/bin/env python3
"""Start state with a chosen number of outer read-in neurons.

The outer count n never changes in a chain, so it can be set: take a chain's final state and return the excess outer
neurons (|w0| > 0.6) to the inner well by redrawing their incoming weights and outgoing column from the prior. The
result is written as <out>/checkpoint.pt for run_fcn3.py --init-from <out>.

    python make_fixed_n.py --from runs/<run> --n 26 --out runs/fixedn/<name>
"""
import argparse
import json
import math
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="src", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
src = Path(a.src)
cfg = json.loads((src / "config.json").read_text())["args"]
ck = torch.load(src / "checkpoint.pt", map_location="cpu", weights_only=False)
ZW, ZB = ck["ZW"].clone(), ck["ZB"].clone()
w0 = ZW[:, 0].double() * math.sqrt(cfg["sigw"] ** 2 / cfg["d"])
outer = torch.where(w0.abs() > 0.6)[0]
n0 = len(outer)
assert a.n <= n0, f"only {n0} outer neurons; this script only removes"
g = torch.Generator().manual_seed(a.seed)
drop = outer[torch.randperm(n0, generator=g)[: n0 - a.n]]
ZW[drop] = torch.randn(len(drop), ZW.shape[1], generator=g).to(ZW.dtype)
ZB[:, drop] = torch.randn(ZB.shape[0], len(drop), generator=g).to(ZB.dtype)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(ZW=ZW, ZB=ZB, step=ck.get("step", 0.03), iter=-1), out / "checkpoint.pt")
w0n = ZW[:, 0].double() * math.sqrt(cfg["sigw"] ** 2 / cfg["d"])
print(f"{src.name}: {n0} outer -> {int((w0n.abs() > 0.6).sum())} outer (target {a.n}); wrote {out}/checkpoint.pt")
