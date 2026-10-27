"""Which parts of f masquerade as He3(x0) on the training set?
f3_train = He3 coefficient of f on the training set (least squares on Hermite(x0) deg ≤ 7)
f3_pop   = exact population He3(x0) coefficient = Σ A_i a3(s_i) ŵ0_i³
gap = f3_train - f3_pop, split by building f's pieces exactly on the training points:
  lin⊥   : Σ A_i a1 (ŵ⊥_i·x⊥)                  (degree-1, orthogonal)
  deg3⊥  : degree-3 part minus its He3(x0) part   (orthogonal + mixed cubic)
  deg1x0 : He1(x0) part (should leak ~0 after LSQ), rest: degree ≥5."""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files, hermite_basis
dev = torch.device("cuda:1")
def he(z, k):
    return z if k == 1 else (z ** 3 - 3 * z) / 6 ** 0.5
print(f"{'group':32s} {'d':>4s} | {'f3_train':>9s} {'f3_pop':>8s} {'gap':>8s} | {'lin⊥':>8s} {'deg3⊥':>8s} {'deg≥5':>8s} | {'gap/f3_pop':>10s}")
for g in sys.argv[1:]:
    a = group_args(g)
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    H = hermite_basis(X[:, 0], (0, 1, 2, 3, 4, 5, 6, 7)); c3 = lambda v: torch.linalg.lstsq(H, v[:, None]).solution[3, 0].item()
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    rows = []
    for f in sample_files(g, 20):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        s = W.norm(dim=1); u = W / s[:, None]
        a1 = 2 * s / torch.sqrt(np.pi * (1 + 2 * s * s)); a3 = -np.sqrt(8 / (3 * np.pi)) * s ** 3 / (1 + 2 * s * s) ** 1.5
        z = X @ u.T                                         # ŵ_i·x  (P, N)
        f_all = Phi @ A
        f_deg1 = (he(z, 1) * (A * a1)).sum(1); f_deg3 = (he(z, 3) * (A * a3)).sum(1)
        f_lin_perp = (X[:, 1:] @ u[:, 1:].T * (A * a1)).sum(1)
        f3_pop = ((A * a3) * u[:, 0] ** 3).sum().item()
        f3_tr = c3(f_all)
        x0 = X[:, 0:1]; u0 = u[:, 0]; t = X[:, 1:] @ u[:, 1:].T; q = 1 - u0 ** 2; c = A * a3
        f32 = ((3 * u0 ** 2 * c) * (x0 ** 2 - 1) / 2 ** 0.5 * t).sum(1)       # He2(x0)·t
        f31 = ((3 * u0 * c) * x0 * (t ** 2 - q)).sum(1) / 6 ** 0.5            # x0·(t² - q)
        f30 = (c * (t ** 3 - 3 * q * t)).sum(1) / 6 ** 0.5                    # pure ⊥ cubic
        rows.append([f3_tr, f3_pop, c3(f_lin_perp), c3(f_deg3) - f3_pop, c3(f_all - f_deg1 - f_deg3), c3(f32), c3(f31), c3(f30)])
    m = np.mean(rows, 0); gap = m[0] - m[1]
    print(f"{'':37s}   deg3⊥ split: He2(x0)·t {m[5]:+.4f}   x0·(t²-q) {m[6]:+.4f}   pure ⊥ {m[7]:+.4f}   (sum {m[5]+m[6]+m[7]:+.4f})")
    print(f"{g:32s} {a['d']:>4d} | {m[0]:>9.4f} {m[1]:>8.4f} {gap:>+8.4f} | {m[2]:>+8.4f} {m[3]:>+8.4f} {m[4]:>+8.4f} | {gap / m[1]:>+10.3f}")
