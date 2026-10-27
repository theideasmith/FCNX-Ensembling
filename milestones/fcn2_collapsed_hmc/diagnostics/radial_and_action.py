import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
torch.set_num_threads(24)
from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files
from neuron_gibbs import NeuronGibbs
from readin_action_theory import reduced_params
from readin_action_theory_q import KernelQ, solve_loop_q, he_coeff
from readin_action_theory_radial import e_k, rho_quadrature
g = "pscan_d150_P3000"; a = group_args(g); d, N, P = a["d"], a["N"], a["P"]; eps = a["eps"]
X, Y, _, _ = (t.double() for t in make_data(d, P, eps, a["he1"], a["seed"], "cpu"))
ng = NeuronGibbs(X, Y, N, a["T"], a["s0"], a["sa0"], a["chi"]); sa2, gam = ng.sa2, ng.g
T0, beta, lam, Chat = reduced_params(N, d, a["s0"], a["sa0"], P, a["T"] / 2)
q0 = a["s0"] * (d - 1) / d; thr = 2.5 * np.sqrt(a["s0"] / d)
rho, wr, psi = rho_quadrature(d, 64, 2)
KS = (1, 3, 5, 7); MS = (0, 1, 2)
def ckm(w0, q):          # (n,) -> dict (k,m) -> (n,)
    A = w0[:, None] / np.sqrt(1 + 2 * q[:, None] * rho[None, :])
    return {(k, m): (e_k(A, k) * (wr * psi[:, m])[None, :]).sum(1) for k in KS for m in MS}
# theory (corrected, He1..He7) self-consistent D
Kq = KernelQ(d, a["s0"], ks=KS); Dth, _ = solve_loop_q(Chat, lam, eps, Kq)
print(f"theory D (He1,3,5,7) = {np.round(Dth, 3)}")
grid = np.linspace(0.02, 1.8, 90)
acts = {"exact": [], "theory_HMC_D": [], "theory_own_D": []}
rad_share = []
t0 = time.time()
for f in sample_files(g, 2):
    W = torch.load(f)["W0"].double().reshape(-1, d)
    Phi, Minv, _, zf = ng._rebuild(W); Phi = Phi.double()
    A = zf                                                   # posterior-mean readout (z = M⁻¹ b)
    w0n = W[:, 0].numpy(); qn = (W[:, 1:] ** 2).sum(1).numpy()
    s = np.sqrt(w0n ** 2 + qn); u0 = w0n / s
    An = A.numpy()
    # ---- #1: exact f coefficients on He_k(x0) psi_m(rho) modes, residual, radial share of aligned neurons' drive
    C = ckm(np.abs(w0n), qn)
    sgn = np.sign(w0n)                                       # odd in w0
    f = {km: float((An * C[km] * sgn ** km[0]).sum()) for km in C}
    y = {km: 0.0 for km in C}; y[(1, 0)] = 1.0; y[(3, 0)] = eps
    Res = {km: y[km] - f[km] for km in C}
    al = np.abs(w0n) > thr
    Ca = ckm(np.full(al.sum(), 1.0), qn[al])                 # aligned neurons moved to w0 = 1, own q
    base = Ca[(1, 0)] * Res[(1, 0)] + Ca[(3, 0)] * Res[(3, 0)]
    he57 = sum(Ca[(k, 0)] * Res[(k, 0)] for k in (5, 7))
    radial = sum(Ca[(k, m)] * Res[(k, m)] for k in KS for m in (1, 2))
    rad_share.append((np.median(he57 / base), np.median(radial / base), {km: Res[km] for km in Res if km[1] > 0}))
    # ---- #2: per-neuron conditional actions on the grid, for a subsample incl. all aligned neurons
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.where(al)[0], rng.choice(np.where(~al)[0], 120, replace=False)])
    D_hmc = np.sqrt(Chat) * np.array([Res[(k, 0)] for k in KS])
    wg = torch.tensor(grid)
    Zc = zf[:, None] - Minv * (zf / torch.diagonal(Minv))[None, :]; Zc.fill_diagonal_(0.0)
    E = Y[:, None] - Phi @ Zc[:, idx]                        # cavity residuals of the chosen neurons (P, n)
    diagM = torch.diagonal(Minv)
    for jj, i in enumerate(idx):
        Fc = torch.erf(X[:, 0:1] * wg[None, :] + (X[:, 1:] @ W[i, 1:])[:, None])
        r = gam * (Fc.T @ E[:, jj])
        Cm = gam * (Phi.T @ Fc); Cm[i, :] = 0.0
        col = Minv[:, i].clone(); col[i] = 0.0
        quad = (Cm * (Minv @ Cm)).sum(0) - (col @ Cm) ** 2 / diagM[i]
        G = sa2 * (gam * (Fc * Fc).sum(0) + 1 / sa2 - quad) - 1
        prior = beta * grid ** 2
        U_ex = prior + 0.5 * np.log1p(G.numpy()) - 0.5 * sa2 * (r ** 2 / (1 + G)).numpy()
        cg = np.stack([he_coeff(grid, np.full_like(grid, qn[i]), k) for k in KS])
        U_hD = prior - beta * (D_hmc @ cg) ** 2
        U_tD = prior - beta * (Dth @ cg) ** 2
        for k_, U in (("exact", U_ex), ("theory_HMC_D", U_hD), ("theory_own_D", U_tD)):
            acts[k_].append((bool(al[i]), U))
    print(f"snapshot done ({time.time()-t0:.0f}s)", flush=True)
# ---- report #1
print("\n#1 exact radial-mode content (d=150):")
he57 = np.mean([r[0] for r in rad_share]); rad = np.mean([r[1] for r in rad_share])
print(f"   aligned neuron at w0=1: He5+He7 share of drive overlap {he57:+.4f}, radial (m=1,2) share {rad:+.4f}   [earlier MC estimate of radial: -0.055]")
big = sorted(rad_share[0][2].items(), key=lambda kv: -abs(kv[1]))[:4]
print("   largest radial residual components (k,m): " + ", ".join(f"{km}: {v:+.4f}" for km, v in big))
# ---- report #2: neuron-averaged normalized conditional densities -> outer well
print("\n#2 per-neuron conditional actions (own w_perp), averaged as normalized densities over aligned neurons:")
dw = grid[1] - grid[0]
for k_, L in acts.items():
    Us = np.array([U for (isal, U) in L if isal])
    Us = Us - Us.min(1, keepdims=True); p = np.exp(-Us); p /= p.sum(1, keepdims=True) * dw
    nlp = -np.log(p.mean(0)); j = 10 + np.argmin(nlp[10:])
    print(f"   {k_:>14s}: outer well at w0 = {grid[j]:.2f}")
