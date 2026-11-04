"""Train vs population residual of the HMC posterior-mean predictor, and the readout's effective degrees
of freedom df = tr[Φ(ΦᵀΦ + (κ_eff/σ_A²) I)⁻¹Φᵀ] on the training set, for every available run group.
Ridge-type prediction: R = MS_train/MS_pop ≈ (1 - df/P)². CPU-capable (training data via make_data)."""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(24)
from collapsed_hmc_fcn2 import make_data, _he3
from diag_cavity_action import group_args, sample_files
from readin_action_theory import reduced_params
groups = sys.argv[1:]
print(f"{'group':>26s} {'d':>4s} {'N':>5s} {'P':>6s} {'κ':>6s} {'λ':>9s} | {'R=MS_tr/MS_pop':>14s} {'df':>7s} {'df/P':>6s} {'(1-df/P)²':>9s} {'N/P':>5s}")
for g in groups:
    a = group_args(g)
    if a is None or not sample_files(g, 1):
        print(f"{g:>26s}: no samples"); continue
    d, N, P = a["d"], a["N"], a["P"]
    X, Y, _, _ = (t.double() for t in make_data(d, P, a["eps"], a["he1"], a["seed"], "cpu"))
    gen = torch.Generator().manual_seed(2024)
    Xf = torch.randn(100000, d, dtype=torch.float64, generator=gen); Yf = Xf[:, 0] + a["eps"] * _he3(Xf[:, 0])
    sa2, hT = a["sa0"] / (N * a["chi"]), a["T"] / (2 * a["chi"])
    _, _, lam, _ = reduced_params(N, d, a["s0"], a["sa0"], P, a["T"] / 2, a["chi"])
    R, DF = [], []
    for f in sample_files(g, 3):
        W = torch.load(f)["W0"].double().reshape(-1, d)
        Phi = torch.erf(X @ W.T); G = Phi.T @ Phi
        M = G / hT + torch.eye(N, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        ms_tr = ((Y - Phi @ A) ** 2).mean().item()
        ms_po = np.mean([((Yf[c:c + 25000] - torch.erf(Xf[c:c + 25000] @ W.T) @ A) ** 2).mean().item() for c in range(0, 100000, 25000)])
        ev = torch.linalg.eigvalsh(G)
        DF.append((ev / (ev + hT / sa2)).sum().item()); R.append(ms_tr / ms_po)
    df = np.mean(DF); r = np.mean(R)
    print(f"{g:>26s} {d:>4d} {N:>5d} {P:>6d} {a['T']/2:>6.3f} {lam:>9.3e} | {r:>14.3f} {df:>7.0f} {df/P:>6.3f} {(1-df/P)**2:>9.3f} {N/P:>5.2f}", flush=True)
