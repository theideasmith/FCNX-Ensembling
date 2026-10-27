#!/usr/bin/env python3
"""
Run registry for collapsed-HMC runs.

runs/registry.jsonl is append-only: one JSON event per line
(running / resumed / done / failed), written with O_APPEND so concurrent runs
don't clobber each other. The current state of a run is the fold of its events.

    python runs.py list [--group G] [--status done]
    python runs.py show <run_id or unique prefix/substring>
    python runs.py compare --group G          # or: compare <id> <id> ...
    python runs.py rhat --group G             # Gelman-Rubin R̂ across the group's chains

Statuses: running, resumed, done, failed, invalid (excluded from compare/rhat).
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def log_event(runs_root, run_id, status, **fields):
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "run_id": run_id, "status": status}
    rec.update({k: v for k, v in fields.items() if v is not None})
    line = (json.dumps(rec) + "\n").encode()
    fd = os.open(runs_root / "registry.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def load_runs(runs_root=RUNS_ROOT):
    path = Path(runs_root) / "registry.jsonl"
    runs = {}
    if not path.exists():
        return runs
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        r = runs.setdefault(ev["run_id"], {"started": ev["time"]})
        r.update(ev)
        r["updated"] = ev["time"]
    return runs


def resolve(runs, key):
    if key in runs:
        return key
    hits = [r for r in runs if r.startswith(key)] or [r for r in runs if key in r]
    if len(hits) != 1:
        raise SystemExit(f"'{key}' matches {len(hits)} runs: {hits[:10]}")
    return hits[0]


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def cmd_list(args):
    runs = load_runs(args.runs_root)
    rows = [r for r in runs.values()
            if (args.group is None or r.get("group") == args.group)
            and (args.status is None or r.get("status") == args.status)]
    rows.sort(key=lambda r: r["started"])
    cols = ["run_id", "status", "group", "init", "rng_seed", "wall_s", "U_post_burn", "L3", "w0_sq", "frac_aligned"]
    table = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) if table else len(c) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for t in table:
        print("  ".join(x.ljust(w) for x, w in zip(t, widths)))


def cmd_show(args):
    runs = load_runs(args.runs_root)
    rid = resolve(runs, args.run)
    run_dir = Path(args.runs_root) / rid
    print(json.dumps(runs[rid], indent=2))
    for name in ("config.json", "summary.json"):
        p = run_dir / name
        if p.exists():
            print(f"\n== {name}")
            print(p.read_text())


def cmd_compare(args):
    runs = load_runs(args.runs_root)
    ids = [resolve(runs, k) for k in args.runs] if args.runs else _group_ids(runs, args.group)
    if not ids:
        raise SystemExit("no runs selected")
    sums, names, lang = [], [], None
    for rid in ids:
        p = Path(args.runs_root) / rid / "summary.json"
        if not p.exists():
            print(f"(skipping {rid}: no summary.json, status={runs[rid].get('status')})")
            continue
        s = json.loads(p.read_text())
        sums.append(s.get("posterior") or s["hmc"])
        names.append(f"{runs[rid].get('init')}/r{runs[rid].get('rng_seed')}")
        lang = s.get("langevin") or lang
    if not sums:
        return
    cols = list(zip(names, sums))
    if lang:
        cols.append(("Langevin", lang["summary"]))
    keys = list(dict.fromkeys(k for s in sums for k in s))
    print(f"{'observable':>16s} " + " ".join(f"{n:>22s}" for n, _ in cols))
    for k in keys:
        print(f"{k:>16s} " + " ".join(f"{s[k][0]:>12.5g} ± {s[k][1]:<7.2g}" if k in s else f"{'-':>22s}"
                                      for _, s in cols))
    U = [runs[r].get("U_post_burn") for r in ids if runs[r].get("U_post_burn") is not None]
    if U:
        print(f"{'U_post_burn':>16s} " + " ".join(f"{u:>22.7g}" for u in U))


def _group_ids(runs, group):
    return sorted((r for r in runs if runs[r].get("group") == group and runs[r].get("status") != "invalid"),
                  key=lambda r: runs[r]["started"])


def _post_burn_series(run_dir):
    """{(chain, key): array} of post-burn observables, plus U/E from trace.csv."""
    out = {}
    with open(Path(run_dir) / "obs.csv") as f:
        for r in csv.DictReader(f):
            if r["burn"] != "0":
                continue
            for k, v in r.items():
                if k not in ("iter", "chain", "burn"):
                    out.setdefault((r["chain"], k), []).append(float(v))
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    burn = cfg["args"]["burn"]
    with open(Path(run_dir) / "trace.csv") as f:
        rows = list(csv.DictReader(f))
    ukey = "U" if rows and "U" in rows[0] else "E_over_T"
    for r in rows:
        if int(r["iter"]) >= burn:
            out.setdefault((r["chain"], "U"), []).append(float(r[ukey]))
    return out


def cmd_rhat(args):
    """Gelman-Rubin R̂ (split-chain) across all chains of all runs in the group."""
    import numpy as np
    runs = load_runs(args.runs_root)
    ids = _group_ids(runs, args.group)
    chains = {}
    for rid in ids:
        d = Path(args.runs_root) / rid
        if not (d / "obs.csv").exists():
            continue
        for (c, k), v in _post_burn_series(d).items():
            chains.setdefault(k, []).append((f"{runs[rid].get('init')}/r{runs[rid].get('rng_seed')}/c{c}", np.array(v)))
    print(f"{len(ids)} runs in group {args.group}")
    print(f"{'quantity':>16s} {'R_hat':>8s}   per-chain means")
    for k, series in chains.items():
        n = min(len(v) for _, v in series) // 2 * 2
        if n < 4 or len(series) < 2:
            print(f"{k:>16s} {'n/a':>8s}   (n={n}, chains={len(series)})")
            continue
        halves = [h for _, v in series for h in (v[-n:][: n // 2], v[-n:][n // 2:])]
        m, L = len(halves), n // 2
        means = np.array([h.mean() for h in halves])
        Wv = np.mean([h.var(ddof=1) for h in halves])
        Bv = L * means.var(ddof=1)
        rhat = math.sqrt(((L - 1) / L * Wv + Bv / L) / Wv) if Wv > 0 else float("inf")
        print(f"{k:>16s} {rhat:8.3f}   " + "  ".join(f"{name}={v.mean():.5g}" for name, v in series))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--group")
    p.add_argument("--status")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("run")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="*")
    p.add_argument("--group")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("rhat")
    p.add_argument("--group", required=True)
    p.set_defaults(fn=cmd_rhat)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
