import json
import numpy as np
import subprocess
from pathlib import Path
import tempfile

def fcn3_theory_solver(
    d,
    n1,
    P,
    chi,
    kappa,
    epsilon=0.0,
    n2=None,
    b=None,
    lr=1e-6,
    max_iter=6_000_000,
    anneal=True,
    anneal_steps=30000,
    tol=1e-12,
    precision=8,
    julia_path=None,
    script_path=None,
    quiet=True,
):
    """
    Interface to the FCN3 Julia theory solver (eos_fcn3erf.jl).
    Returns parsed JSON results from the Julia script.
    """
    if n2 is None:
        n2 = n1
    if b is None:
        b = 4.0 / (3.0 * np.pi)
    if julia_path is None:
        julia_path = "julia"
    if script_path is None:
        script_path = Path(__file__).parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            julia_path,
            str(script_path),
            "--d", str(d),
            "--kappa", str(kappa),
            "--epsilon", str(epsilon),
            "--P", str(P),
            "--n1", str(n1),
            "--n2", str(n2),
            "--chi", str(chi),
            "--b", str(b),
            "--lr", str(lr),
            "--max-iter", str(max_iter),
            "--anneal-steps", str(anneal_steps),
            "--tol", str(tol),
            "--precision", str(precision),
            "--to", str(tmp_path)
        ]
        if quiet:
            cmd.append("--quiet")
        if not anneal:
            cmd.append("--no-anneal")
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, "r") as f:
            result = json.load(f)
        return result
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
