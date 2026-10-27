"""Exact Hermite-degree power of the posterior-mean predictor f = Σ Ā_i erf(w_i·x), x ~ N(0, I):
degree-k power  = Σ_ij Ā_i Ā_j a_k(|w_i|) a_k(|w_j|) (ŵ_i·ŵ_j)^k,  teacher part = (Σ_i Ā_i a_k ŵ_i0^k)²,
a_k(s) = normalized-Hermite coefficient of erf(s z): a1 = 2s/√(π(1+2s²)), a3 = -√(8/(3π)) s³/(1+2s²)^{3/2}."""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files
dev = torch.device("cuda:1")
print(f"{'group':32s} {'d':>4s} | {'He1(x0)':>8s} {'lin ⊥':>7s} | {'He3(x0)':>8s} {'cubic ⊥':>8s} {'⊥/total cubic':>13s} | {'total f²':>8s} | {'aligned ŵ0²':>11s}")
for g in sys.argv[1:]:
    a = group_args(g)
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    acc = []
    for f in sample_files(g, 30):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        s = W.norm(dim=1); u = W / s[:, None]; G = u @ u.T
        a1 = 2 * s / torch.sqrt(np.pi * (1 + 2 * s * s)); a3 = -np.sqrt(8 / (3 * np.pi)) * s ** 3 / (1 + 2 * s * s) ** 1.5
        p1 = (A * a1) @ G @ (A * a1); p3 = (A * a3) @ G ** 3 @ (A * a3)
        t1 = ((A * a1) @ u[:, 0]) ** 2; t3 = ((A * a3) @ u[:, 0] ** 3) ** 2
        # total power E[f²] = Σ_ij A_i A_j (2/π) asin(2 w_i·w_j / sqrt((1+2|w_i|²)(1+2|w_j|²)))
        C = 2 * (W @ W.T) / torch.sqrt(torch.outer(1 + 2 * s * s, 1 + 2 * s * s))
        tot = A @ (2 / np.pi * torch.asin(C.clamp(-1, 1))) @ A
        al = W[:, 0].abs() > 2.5 * np.sqrt(a["s0"] / a["d"])
        acc.append([t1.item(), (p1 - t1).item(), t3.item(), (p3 - t3).item(), tot.item(), (u[al, 0] ** 2).mean().item()])
    m = np.mean(acc, 0)
    print(f"{g:32s} {a['d']:>4d} | {m[0]:>8.4f} {m[1]:>7.4f} | {m[2]:>8.4f} {m[3]:>8.4f} {m[3] / (m[2] + m[3]):>13.3f} | {m[4]:>8.4f} | {m[5]:>11.3f}")
