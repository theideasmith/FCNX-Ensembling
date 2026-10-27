"""CLI arguments, initial states and Langevin comparison shared by the runners."""

import glob
import math
from pathlib import Path

import torch

from bookkeeping import OBS_KEYS
from collapsed_hmc_fcn2 import observables
from runs import RUNS_ROOT


def add_physics_args(ap):
    g = ap.add_argument_group("physics (must match the Langevin run)")
    g.add_argument("--d", type=int, required=True)
    g.add_argument("--P", type=int, required=True)
    g.add_argument("--N", type=int, required=True)
    g.add_argument("--T", type=float, required=True, help="Langevin temperature (T = 2κ)")
    g.add_argument("--chi", type=float, default=1.0)
    g.add_argument("--s0", type=float, default=1.0)
    g.add_argument("--sa0", type=float, required=True)
    g.add_argument("--eps", type=float, default=0.5)
    g.add_argument("--he1", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0, help="dataset_seed of the Langevin run")


def add_init_args(ap):
    g = ap.add_argument_group("initial state")
    g.add_argument("--init", choices=["prior", "small", "snapshot", "file"], default="prior",
                   help="prior: W0 ~ N(0, σ_W²); small: 0.1 σ_W (no features); snapshot: last Langevin snapshot")
    g.add_argument("--init-file", default=None, help="--init file: torch file with W0 (N, d)")
    g.add_argument("--rng-seed", type=int, default=1234, help="sampler RNG (init draw, momenta, proposals)")
    g.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    g.add_argument("--device", default="cuda:0")


def add_gibbs_args(ap):
    g = ap.add_argument_group("per-neuron Gibbs jumps (neuron_gibbs.py)")
    g.add_argument("--gibbs-every", type=int, default=0, help="sweep before every k-th iteration (0: off)")
    g.add_argument("--gibbs-cands", type=int, default=64, help="proposal candidates per neuron")
    g.add_argument("--gibbs-mode", choices=["w0", "full", "both"], default="w0")
    g.add_argument("--gibbs-omega", type=float, default=0.6, help="std of the w0 proposal N(0, ω²)")
    g.add_argument("--gibbs-sperp", type=float, default=None, help="std of w_perp proposal in 'full' mode (default σ_W)")
    g.add_argument("--gibbs-block", type=int, default=0,
                   help="vectorized block-MH w0 moves (fast_moves.BlockGibbs) with this initial block size "
                        "(adapted until --burn, then fixed); replaces the sequential w0 sweep (0: off)")
    g.add_argument("--gibbs-block-fixed", action="store_true", help="keep the block size fixed (no adaptation)")
    g.add_argument("--gibbs-graph", action="store_true",
                   help="exact sequential w0 sweep captured in a CUDA graph (fast_moves.GraphedNeuronGibbs); "
                        "M⁻¹ kept in the run dtype")


def add_bookkeeping_args(ap):
    b = ap.add_argument_group("bookkeeping")
    b.add_argument("--tag", default="run", help="short label used in the run_id")
    b.add_argument("--group", default=None, help="group name for runs meant to be compared")
    b.add_argument("--langevin-dir", default=None,
                   help="Langevin run dir (with A_snapshots/) for provenance, --init snapshot and comparison")
    b.add_argument("--obs-every", type=int, default=5)
    b.add_argument("--save-every", type=int, default=10, help="save W0/A samples every k post-burn iters (0: never)")
    b.add_argument("--checkpoint-every", type=int, default=25)
    b.add_argument("--runs-root", default=str(RUNS_ROOT))
    b.add_argument("--resume", default=None, help="run dir to resume; other args are taken from its config")


def snapshot_files(langevin_dir):
    return sorted(glob.glob(str(Path(langevin_dir) / "A_snapshots" / "epoch_*.pt")))


def initial_W(args, chains, N, d, sw, device, dtype):
    """Initial read-in weights (chains, N, d) and, for --init snapshot, the snapshot used."""
    if args.init == "file":
        W = torch.load(args.init_file)["W0"].to(device, dtype).reshape(N, d)
        return W.expand(chains, -1, -1).clone(), args.init_file
    if args.init == "snapshot":
        f = snapshot_files(args.langevin_dir)[-1]
        s = torch.load(f)
        return s["W0"].to(device, dtype).expand(chains, -1, -1).clone(), f
    W = torch.randn(chains, N, d, device=device, dtype=dtype) * sw
    return (0.1 * W if args.init == "small" else W), None


def langevin_comparison(model, langevin_dir, Xinf, Yinf, device, dtype):
    files = snapshot_files(langevin_dir) if langevin_dir else []
    if not files:
        return None
    vals = {k: [] for k in OBS_KEYS}
    for f in files:
        s = torch.load(f)
        ob = observables(model, s["W0"].to(device, dtype) / model.sw, s["A"].to(device, dtype), Xinf, Yinf)
        for k in OBS_KEYS:
            vals[k].append(ob[k].flatten().double().cpu())
    summary = {}
    for k in OBS_KEYS:
        v = torch.cat(vals[k])
        summary[k] = [v.mean().item(), v.std().item() / math.sqrt(len(v)), len(v)]
    return {"snapshot_dir": str(Path(langevin_dir) / "A_snapshots"), "n_snapshots": len(files), "summary": summary}


def gibbs_sweeps(sampler, W, args):
    """Run the configured Gibbs sweep(s) on each chain of W (chains, N, d)."""
    modes = ["w0", "full"] if args.gibbs_mode == "both" else [args.gibbs_mode]
    stats = {"changed": 0.0, "crossed": 0.0}
    out = []
    for c in range(W.shape[0]):
        w = W[c]
        for mode in modes:
            w, st = sampler.sweep(w, B=args.gibbs_cands, mode=mode)
            stats["changed"] += st["changed"] / (len(modes) * W.shape[0])
            stats["crossed"] += st["crossed"] / (len(modes) * W.shape[0])
        out.append(w)
    return torch.stack(out), stats


def print_table(summary):
    cols = [("sampler", summary["posterior"])]
    if summary.get("langevin"):
        cols.append((f"Langevin ({summary['langevin']['n_snapshots']} snaps)", summary["langevin"]["summary"]))
    print(f"{'observable':>16s} " + " ".join(f"{n:>26s}" for n, _ in cols))
    for k in OBS_KEYS:
        print(f"{k:>16s} " + " ".join(f"{s[k][0]:>14.5g} ± {s[k][1]:<9.2g}" for _, s in cols if k in s))
