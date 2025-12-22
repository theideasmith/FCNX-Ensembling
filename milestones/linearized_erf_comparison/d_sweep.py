#!/usr/bin/env python3
"""
Replicates d_sweep for both linear and erf networks (with rescaled weights for erf).
Trains over a sweep of d, for both activations, and saves results in runs/.
"""
import subprocess
from pathlib import Path
import itertools

# Sweep parameters
Ds = [2, 4, 6, 8, 10]
chis = [1.0]  # Only SSC for now
activations = ["linear", "erf"]

base_dir = Path(__file__).parent
script = base_dir / "train_linearized_comparison.py"

for d, chi, activation in itertools.product(Ds, chis, activations):
    cmd = [
        "python", str(script),
        "--d", str(d),
        "--activation", activation,
        "--chi", str(chi),
        "--datasets", "20",
        "--ensembles", "4",
        "--epochs", "4000000",
        "--log-interval", "10000",
        "--device", "cuda:1",
        "--lr", "5e-6",
        "--kappa", "0.1",
        "--sigma-w-lin", "1.0"
    ]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
