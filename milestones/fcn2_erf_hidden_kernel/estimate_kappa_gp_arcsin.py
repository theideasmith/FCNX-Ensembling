

import torch
import argparse
import json
import subprocess
import tempfile
import os


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)

def main():
    parser = argparse.ArgumentParser(description="Compute kappa correction via GP arcsin kernel eigenvalues and Julia solver.")
    parser.add_argument('--n', type=int, default=1000, help='Number of samples')
    parser.add_argument('--d', type=int, default=100, help='Input dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--kappa_bare', type=float, required=True, help='Bare kappa value')
    parser.add_argument('--P', type=int, required=True, help='P value for Julia solver')
    parser.add_argument('--julia_script', type=str, default='../../julia_lib/self_consistent_kappa_solver.jl', help='Path to Julia solver script')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X = torch.randn(args.n, args.d)
    K = arcsin_kernel(X)
    eigvals = torch.linalg.eigvalsh(K).cpu() / args.n

    # Write eigenvalues and kappa_bare to a temp JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpf:
        json.dump({'eigenvalues': eigvals.tolist(), 'kappa_bare': float(args.kappa_bare)}, tmpf)
        tmp_json_path = tmpf.name

    # Call Julia solver
    julia_cmd = [
        'julia',
        args.julia_script,
        tmp_json_path,
        str(args.P)
    ]
    try:
        result = subprocess.run(julia_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Julia solver failed: {e.stderr}")
    finally:
        os.remove(tmp_json_path)

if __name__ == "__main__":
    main()
