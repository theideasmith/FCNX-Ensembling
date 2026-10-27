import sys, numpy as np, torch
from math import factorial, sqrt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files
from plot_readin_action import theory_curve
from readin_action_theory import reduced_params
from readin_action_theory_q import analyse_q, KernelQ
dev = torch.device("cuda:0")
def dfact(n): return 1 if n <= 0 else n * dfact(n - 2)
def a_k(s, k): return (2 / sqrt(np.pi)) * (-1) ** ((k - 1) // 2) * dfact(k - 1) / sqrt(factorial(k)) * s ** k / (1 + 2 * s * s) ** (k / 2)
groups = ["ray_k0p015_both_b1", "ray_k0p015_both_b1p41421", "ray_k0p015_both_b2", "ray_k0p015_both_b2p82843", "ray_k0p015_both_b4"]
out = []
for g in groups:
    a = group_args(g); d = a["d"]
    X, Y, _, _ = (t.double() for t in make_data(d, a["P"], a["eps"], a["he1"], a["seed"], dev))
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    fk = []
    files = sample_files(g, 60)
    for f in files:
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, d)
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        s = W.norm(dim=1); u0 = W[:, 0] / s
        fk.append([(A * a_k(s, k) * u0 ** k).sum().item() for k in (1, 3, 5, 7)])
        del Phi, M
    fk = np.array(fk); f1, f3, f5, f7 = fk.mean(0); se3 = fk[:, 1].std() / np.sqrt(len(fk))
    L3 = f3 / a["eps"]
    c = (a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2)
    T = 1 + 2 * a["s0"] * (d - 1) / d
    _, th0 = theory_curve(a, T, np.linspace(-1, 1, 11))
    q = analyse_q(*c, a["eps"], ks=(1, 3, 5, 7))
    _, _, lam, _ = reduced_params(*c)
    J = KernelQ(d, a["s0"], ks=(1, 3, 5, 7)).stats(q["D"])[2]
    fq = J @ np.linalg.solve(J + lam * np.eye(4), np.array([1, a["eps"], 0, 0]))
    out.append((d, L3, th0["L3"], q["L3"], -f5, -fq[2], -f7, -fq[3]))
    print(f"d={d:3d} ({len(files)} snaps): L3 HMC {L3:.4f}±{se3/abs(a['eps']):.4f}  orig {th0['L3']:.4f} ({th0['L3']/L3-1:+.1%})  corr {q['L3']:.4f} ({q['L3']/L3-1:+.1%})"
          f" | residual Δ5: HMC {-f5:+.4f} corr {-fq[2]:+.4f} | Δ7: HMC {-f7:+.4f} corr {-fq[3]:+.4f} | L1 HMC {f1:.4f}", flush=True)
o = np.array(out); ds = o[:, 0]
for name, y in (("L3 orig/HMC-1", o[:, 2] / o[:, 1] - 1), ("L3 corr/HMC-1", o[:, 3] / o[:, 1] - 1), ("Δ7 corr/HMC", o[:, 7] / o[:, 6])):
    print(f"trend {name:>14s}: " + " ".join(f"{v:+.3f}" for v in y))
