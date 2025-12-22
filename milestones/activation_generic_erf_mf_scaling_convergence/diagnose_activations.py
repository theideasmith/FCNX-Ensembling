#!/usr/bin/env python3
"""
Diagnostic script for activation_generic_erf_mf_scaling_convergence.

Scans d*_P*_N*_chi* directories, loads models, and prints statistics of
preactivations/activations for each layer on the training-like dataset
(seed 42, same P and d as used in training).

Usage:
    python diagnose_activations.py [--device cuda:0]
"""

import argparse
import re
from pathlib import Path
import sys
import torch
import numpy as np

# Ensure lib is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric  # noqa: E402


def tensor_stats(name: str, t: torch.Tensor):
    t_np = t.detach().cpu().numpy()
    print(
        f"  {name}: mean={t_np.mean():.4e}, std={t_np.std():.4e}, "
        f"min={t_np.min():.4e}, max={t_np.max():.4e}"
    )


def parse_run_dir(run_dir: Path):
    m = re.match(r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi(?P<chi>\d+)", run_dir.name)
    if not m:
        return None
    vals = {k: int(v) for k, v in m.groupdict().items()}
    return vals["d"], vals["P"], vals["N"], vals["chi"]


def load_model(run_dir: Path, device: torch.device):
    parsed = parse_run_dir(run_dir)
    if parsed is None:
        print(f"[skip] could not parse directory name {run_dir.name}")
        return None, None, None, None, None
    d, P, N, chi = parsed

    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    if not ckpt.exists():
        print(f"[skip] no checkpoint in {run_dir}")
        return None, None, None, None, None

    state = torch.load(ckpt, map_location=device)
    # Infer ensemble count from checkpoint shape
    ens = state.get("W0", state.get("A")).shape[0]

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
        device=device,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    X = torch.randn(3000, d, device=device, dtype=torch.float32)
    print(model.H_eig_random_svd(X)[:d])
    return model, d, P, N, chi


def diagnose_run(run_dir: Path, device: torch.device):
    model, d, P, N, chi = load_model(run_dir, device)
    if model is None:
        return

    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)

    with torch.no_grad():
        h0_pre = model.h0_preactivation(X)           # (P, ens, n1)
        h0_act = model.h0_activation(X)              # (P, ens, n1)
        h1_pre = model.h1_preactivation(X)           # (P, ens, n2)
        h1_act = model.h1_activation(X)              # (P, ens, n2)
        out = model.forward(X)                       # (P, ens)

    print(f"\nRun: {run_dir.name}")
    print(f"d={d}, P={P}, N={N}, chi={chi}")

    # Collapse all dimensions for summary
    tensor_stats("h0_pre", h0_pre)
    tensor_stats("h0_act", h0_act)
    tensor_stats("h1_pre", h1_pre)
    tensor_stats("h1_act", h1_act)
    tensor_stats("output", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1, or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    base = Path(__file__).parent
    run_dirs = sorted([p for p in base.glob("d*_P*_N*_chi*") if p.is_dir()])
    if not run_dirs:
        print(f"No run directories found under {base}")
        return

    for rd in run_dirs:
        diagnose_run(rd, device)


if __name__ == "__main__":
    main()
