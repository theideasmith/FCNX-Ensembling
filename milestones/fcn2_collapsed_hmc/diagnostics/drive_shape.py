import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
torch.set_num_threads(24)
from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files
from neuron_gibbs import NeuronGibbs
from readin_action_theory import reduced_params
from readin_action_theory_radial import e_k, rho_quadrature
KS = (1, 3, 5, 7, 9); MS = (0, 1, 2); grid = np.array([0.4, 0.7, 1.0, 1.3, 1.6])
for g in sys.argv[1:]:
    a = group_args(g); d, N, P, eps = a["d"], a["N"], a["P"], a["eps"]
    X, Y, _, _ = (t.double() for t in make_data(d, P, eps, a["he1"], a["seed"], "cpu"))
    ng = NeuronGibbs(X, Y, N, a["T"], a["s0"], a["sa0"], a["chi"]); sa2, gam = ng.sa2, ng.g
    _, beta, lam, Chat = reduced_params(N, d, a["s0"], a["sa0"], P, a["T"] / 2)
    rho, wr, psi = rho_quadrature(d, 64, 2); thr = 2.5 * np.sqrt(a["s0"] / d)
    ckm = lambda w0, q: {(k, m): (e_k(w0[:, None] / np.sqrt(1 + 2 * q[:, None] * rho[None, :]), k) * (wr * psi[:, m])[None, :]).sum(1) for k in KS for m in MS}
    out = {k: [] for k in ("train", "trainG", "pop", "theoryHe17", "G")}
    for f in sample_files(g, 1):
        W = torch.load(f)["W0"].double().reshape(-1, d)
        Phi, Minv, _, zf = ng._rebuild(W); Phi = Phi.double()
        w0n = W[:, 0].numpy(); qn = (W[:, 1:] ** 2).sum(1).numpy(); An = zf.numpy()
        C = ckm(np.abs(w0n), qn)
        Res = {km: (1.0 if km == (1, 0) else eps if km == (3, 0) else 0.0) - float((An * C[km] * np.sign(w0n) ** km[0]).sum()) for km in C}
        Dh = np.sqrt(Chat) * np.array([Res[(k, 0)] for k in (1, 3, 5, 7)])
        al = np.where(np.abs(w0n) > thr)[0]
        Zc = zf[:, None] - Minv * (zf / torch.diagonal(Minv))[None, :]; Zc.fill_diagonal_(0.0)
        E = Y[:, None] - Phi @ Zc[:, al]; diagM = torch.diagonal(Minv); wg = torch.tensor(grid)
        Zal = Zc[:, al].numpy()                                  # refitted readout without neuron i (column per i)
        Rcav = {km: (1.0 if km == (1, 0) else eps if km == (3, 0) else 0.0) - Zal.T @ (C[km] * np.sign(w0n) ** km[0]) for km in C}
        for jj, i in enumerate(al):
            Fc = torch.erf(X[:, 0:1] * wg[None, :] + (X[:, 1:] @ W[i, 1:])[:, None])
            r = (gam * (Fc.T @ E[:, jj])).numpy()
            Cm = gam * (Phi.T @ Fc); Cm[i, :] = 0.0
            col = Minv[:, i].clone(); col[i] = 0.0
            G = (sa2 * (gam * (Fc * Fc).sum(0) + 1 / sa2 - ((Cm * (Minv @ Cm)).sum(0) - (col @ Cm) ** 2 / diagM[i])) - 1).numpy()
            ci = ckm(grid, np.full_like(grid, qn[i]))
            # population overlap with the cavity residual: add back neuron i's own contribution
            ov = sum(ci[km] * Rcav[km][jj] for km in ci)
            out["train"].append(0.5 * sa2 * r ** 2); out["trainG"].append(0.5 * sa2 * r ** 2 / (1 + G))
            out["pop"].append(0.5 * sa2 * (gam * P * ov) ** 2)
            cg = np.stack([ci[(k, 0)] for k in (1, 3, 5, 7)])
            out["theoryHe17"].append(beta * (Dh @ cg) ** 2); out["G"].append(G)
    m = {k: np.mean(v, 0) for k, v in out.items()}
    print(f"\n{g} (P={P}, {len(al)} aligned neurons; drives averaged over them, ratios to the theory form β(D_HMC·c)²)")
    print("   w0:                       " + " ".join(f"{w:>7.2f}" for w in grid))
    print("   theory form (nats):       " + " ".join(f"{x:>7.1f}" for x in m["theoryHe17"]))
    print("   population overlap / th:  " + " ".join(f"{x:>7.3f}" for x in m["pop"] / m["theoryHe17"]) + "   (exact mode sum, cavity readout)")
    print("   training, no G / th:      " + " ".join(f"{x:>7.3f}" for x in m["train"] / m["theoryHe17"]))
    print("   training, with G / th:    " + " ".join(f"{x:>7.3f}" for x in m["trainG"] / m["theoryHe17"]))
    print("   mean G:                   " + " ".join(f"{x:>7.3f}" for x in m["G"]))
