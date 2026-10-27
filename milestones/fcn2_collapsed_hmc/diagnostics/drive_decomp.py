"""Population-level decomposition of the exact per-neuron drive against the theory's β(√Ĉ Δ_pop·c)².
exact  = ½σ_A² (gP O_i)²/(1+G_i),  O_i = <φ_i(w0), e_{-i}>_pop
theory = ½σ_A² (gP c_i·Δ_pop)²       (identity ½σ_A² g² P² = βĈ checked algebraically)
factors: [G] 1/(1+G);  [cav] (c·Δcav_i)² vs (c·Δpop)²;  [modes] O_i² vs (c·Δcav_i)²."""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from collapsed_hmc_fcn2 import make_data, _he3
from diag_cavity_action import group_args, sample_files
from neuron_gibbs import NeuronGibbs
from readin_action_theory import features, reduced_params
dev = torch.device("cuda:0"); w0s = [0.6, 1.0, 1.3]; NT = 20000
for g in sys.argv[1:]:
    a = group_args(g)
    _, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    gen = torch.Generator(device=dev).manual_seed(4242)
    Xt = torch.randn(NT, a["d"], device=dev, dtype=torch.float64, generator=gen)
    he1 = Xt[:, 0]; he3 = _he3(Xt[:, 0])
    Yt = a["he1"] * he1 + a["eps"] * he3
    ng = NeuronGibbs(X, Y, a["N"], a["T"], a["s0"], a["sa0"], a["chi"]); gam = ng.g
    acc = {w0: {k: [] for k in ("theory", "noG_cavHe", "noG_full", "withG_full", "G", "orth_frac", "dcav3", "dpop3")} for w0 in w0s}
    for f in sample_files(g, 3):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi, Minv, _, zf = ng._rebuild(W); Phi = Phi.double()
        diagM = torch.diagonal(Minv)
        Zc = zf[:, None] - Minv * (zf / diagM)[None, :]; Zc.fill_diagonal_(0.0)
        Pt = torch.erf(Xt @ W.T)
        e_full = Yt - Pt @ zf                                          # population residual of the posterior mean
        Dpop = torch.stack([(e_full * he1).mean(), (e_full * he3).mean()]).cpu().numpy()   # He1, He3 coefficients
        E_po = Yt[:, None] - Pt @ Zc                                   # (NT, N) cavity residual functions
        Dcav1 = (E_po * he1[:, None]).mean(0); Dcav3 = (E_po * he3[:, None]).mean(0)
        q = (W[:, 1:] ** 2).sum(1).cpu().numpy()
        for w0 in w0s:
            Fpo = torch.erf(Xt[:, 0:1] * w0 + Xt[:, 1:] @ W[:, 1:].T)    # (NT, N)
            O = (Fpo * E_po).mean(0)
            c1, c3 = features(np.full(a["N"], w0), 1 + 2 * q)
            c1t = torch.tensor(c1, device=dev); c3t = torch.tensor(c3, device=dev)
            cDcav = c1t * Dcav1 + c3t * Dcav3
            cDpop = c1 * Dpop[0] + c3 * Dpop[1]
            Ftr = torch.erf(X[:, 0:1] * w0 + X[:, 1:] @ W[:, 1:].T)
            C = gam * (Phi.T @ Ftr); C.fill_diagonal_(0.0)
            quad = (C * (Minv @ C)).sum(0) - ((Minv * C).sum(0)) ** 2 / diagM
            G = ng.sa2 * (gam * (Ftr * Ftr).sum(0) + 1 / ng.sa2 - quad) - 1
            k = 0.5 * ng.sa2 * (gam * a["P"]) ** 2
            A = acc[w0]
            A["theory"].append(k * np.mean(cDpop ** 2))
            A["noG_cavHe"].append(k * (cDcav ** 2).mean().item())
            A["noG_full"].append(k * (O ** 2).mean().item())
            A["withG_full"].append(k * (O ** 2 / (1 + G)).mean().item())
            A["G"].append(G.mean().item())
            A["orth_frac"].append(((O - cDcav) / cDcav).median().item())
            A["dcav3"].append(Dcav3.mean().item()); A["dpop3"].append(Dpop[1])
    print(f"\n{g}  (d={a['d']} N={a['N']} χ={a['chi']:g} P={a['P']} κ={a['T']/2:g})   theory drive uses HMC's own Δ_pop")
    print(f"   {'w0':>4s} | {'exact/theory':>12s} = {'1/(1+G)':>8s} × {'cavity':>7s} × {'other modes':>11s} | {'mean G':>6s} {'median (O-c·Δcav)/c·Δcav':>24s} | {'Δ3 cav':>7s} {'Δ3 pop':>7s}")
    for w0 in w0s:
        A = {k: np.mean(v) for k, v in acc[w0].items()}
        tot = A["withG_full"] / A["theory"]; fG = A["withG_full"] / A["noG_full"]
        fc = A["noG_cavHe"] / A["theory"]; fm = A["noG_full"] / A["noG_cavHe"]
        print(f"   {w0:>4.1f} | {tot:>12.3f} = {fG:>8.3f} × {fc:>7.3f} × {fm:>11.3f} | {A['G']:>6.3f} {A['orth_frac']:>24.3f} | {A['dcav3']:>7.4f} {A['dpop3']:>7.4f}")
