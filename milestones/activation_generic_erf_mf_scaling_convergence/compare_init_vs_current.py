#!/usr/bin/env python3
"""
Compare H-eigenvalues at model initialization vs current checkpoint weights
for a given run directory (e.g., d10_P30_N50_chi50).

- Parses d, P, N, chi from the folder name
- Loads checkpoint `model.pt`
- Builds an "init" model with the same arch and ensemble size
- Uses the same dataset (seeded) to compute H_eig for init vs current
- Prints absolute and relative errors

Usage:
    python compare_init_vs_current.py --run d10_P30_N50_chi50 [--device cpu]
"""

import argparse
import re
from pathlib import Path
import sys
import torch

# Add lib to path
LIB_DIR = str(Path(__file__).parent.parent.parent / "lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from FCN3Network import FCN3NetworkActivationGeneric


def parse_run_dir(name: str):
    m = re.match(r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi(?P<chi>\d+)", name)
    if not m:
        raise ValueError(f"Could not parse run directory name: {name}")
    g = {k: int(v) for k, v in m.groupdict().items()}
    return g["d"], g["P"], g["N"], g["chi"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run folder name, e.g. d10_P30_N50_chi50")
    parser.add_argument("--device", default=None, help="cuda:0|cpu")
    args = parser.parse_args()

    base = Path(__file__).parent
    run_dir = base / args.run
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}")
        return

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    d, P, N, chi = parse_run_dir(run_dir.name)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    if not ckpt.exists():
        print(f"No checkpoint in {run_dir}")
        return

    state = torch.load(ckpt, map_location=device)
    # infer ensemble size from weights
    ens = state.get("W0", state.get("A")).shape[0]

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    print(d)
    # Build current model and load weights
    cur_model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
        device=device,
    ).to(device)
    cur_model.load_state_dict(state)
    cur_model.eval()

    # Build init model with same arch/ens (fresh random init)
    torch.manual_seed(42)
    init_model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
        device=device,
    ).to(device)
    init_model.eval()

    # Dataset: use fixed seed and same P,d
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device, dtype=torch.float32)
    z = X[:, 0]
    Y = z.view(-1, 1)  # He1 target column

    with torch.no_grad():
        h_init = init_model.H_eig(X, X)[1:]
        print(h_init)
        h_cur = cur_model.H_eig(X, X)[1:]
        print(h_cur)

    # Ensure scalar tensors
    h_init_val = h_init.mean().item() if torch.is_tensor(h_init) else float(h_init)
    h_cur_val = h_cur.mean().item() if torch.is_tensor(h_cur) else float(h_cur)

    abs_err = abs(h_cur_val - h_init_val)
    rel_err = abs_err / abs(h_init_val) if h_init_val != 0 else float("inf")

    print("=" * 70)
    print(f"Run: {run_dir.name}  (d={d}, P={P}, N={N}, chi={chi}, ens={ens})")
    print(f"Device: {device}")
    print("Dataset: seed=42, Y=He1")
    print("-- H eigenvalue (Rayleigh quotient along He1) --")
    print(f"Init:     {h_init_val:.10e}")
    print(f"Current:  {h_cur_val:.10e}")
    print(f"Abs err:  {abs_err:.10e}")
    print(f"Rel err:  {rel_err:.6%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
