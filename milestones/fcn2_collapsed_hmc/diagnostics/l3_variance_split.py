"""Law of total variance for the He3 coefficient of f over HMC snapshots:
Var(f3) = E_W[Var(f3|W)] (readout, exact Gaussian A|W) + Var_W(E[f3|W]) (feature fluctuations)."""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from collapsed_hmc_fcn2 import make_data, CollapsedFCN2, _he3
from diag_cavity_action import group_args, sample_files
from readin_action_theory import Kernel, reduced_params, solve_loop
print(f"{'κ':>5s} {'snaps':>5s} | {'sd readout (HMC)':>16s} {'sd features (HMC)':>17s} {'sd total':>8s} | {'sd theory (readout only)':>24s} | {'mean #aligned':>13s} {'sd #aligned':>11s}")
for g in sys.argv[1:]:
    a = group_args(g); dev = "cuda:0"
    X, Y, Xi, Yi = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    m = CollapsedFCN2(X, Y, a["N"], a["T"], a["s0"], a["sa0"], a["chi"])
    he1, he3 = Xi[:, 0], _he3(Xi[:, 0])
    norm3 = (Yi * he3).mean()
    means, vars_, nal = [], [], []
    thr = 2.5 * m.sw
    for f in sample_files(g, 60):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T); hT = m.T / 2
        M = Phi.T @ Phi / hT + torch.eye(m.N, device=dev, dtype=W.dtype) / m.sa2
        L = torch.linalg.cholesky(M)
        Abar = torch.cholesky_solve((Phi.T @ Y / hT)[:, None], L)[:, 0]
        Pi = torch.erf(Xi @ W.T)
        # L3 is linear in A: L3 = g·A with g = Pi^T (he3 - <.,he1> he1 correction) / (n norm3), as in observables()
        g1 = Pi.T @ he1 / len(he1)                     # (N,) He1 coefficient per feature
        res = Pi - he1[:, None] * g1[None, :]          # remove He1 part per feature
        gvec = res.T @ he3 / len(he3) / norm3          # (N,)
        means.append((gvec @ Abar).item())
        vars_.append((gvec @ torch.cholesky_solve(gvec[:, None], L)[:, 0]).item())   # gᵀ M⁻¹ g = Var(L3|W)
        nal.append((W[:, 0].abs() > thr).sum().item())
    T, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"]/2, a["chi"])
    K = Kernel(T, beta); D, _ = solve_loop(Chat, lam, a["eps"], K); J = K.stats(D)[3]
    Sig = (a["T"]/2) / a["P"] * J @ np.linalg.inv(J + lam*np.eye(2))
    ro, fe = np.sqrt(np.mean(vars_)), np.std(means)
    print(f"{a['T']/2:>5g} {len(means):>5d} | {ro:>16.4f} {fe:>17.4f} {np.hypot(ro, fe):>8.4f} | {np.sqrt(Sig[1,1])/abs(a['eps']):>24.4f} | {np.mean(nal):>13.1f} {np.std(nal):>11.1f}")
