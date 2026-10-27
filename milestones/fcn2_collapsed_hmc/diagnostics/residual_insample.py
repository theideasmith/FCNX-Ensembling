import sys, numpy as np, torch
from math import factorial, sqrt
from numpy.polynomial.hermite_e import hermeval
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
torch.set_num_threads(24)
from collapsed_hmc_fcn2 import make_data, _he3
from diag_cavity_action import group_args, sample_files
from neuron_gibbs import NeuronGibbs
from readin_action_theory_radial import e_k, rho_quadrature
KMAX = 13
def Hmat(x0): return np.stack([hermeval(x0, np.eye(k + 1)[k]) / sqrt(factorial(k)) for k in range(KMAX + 1)], 1)
for g in sys.argv[1:]:
    a = group_args(g); d, N, P, eps = a["d"], a["N"], a["P"], a["eps"]
    X, Y, _, _ = (t.double() for t in make_data(d, P, eps, a["he1"], a["seed"], "cpu"))
    ng = NeuronGibbs(X, Y, N, a["T"], a["s0"], a["sa0"], a["chi"])
    W = torch.load(sample_files(g, 1)[0])["W0"].double().reshape(-1, d)
    _, Minv, _, zf = ng._rebuild(W)
    w0n = W[:, 0].numpy(); al = np.where(np.abs(w0n) > 2.5 * np.sqrt(a["s0"] / d))[0]
    Zc = (zf[:, None] - Minv * (zf / torch.diagonal(Minv))[None, :]); Zc.fill_diagonal_(0.0); Zal = Zc[:, al]
    gen = torch.Generator().manual_seed(31337)
    Xf = torch.randn(200000, d, dtype=torch.float64, generator=gen)
    yf = Xf[:, 0] + eps * _he3(Xf[:, 0])
    def resid(Xs, ys):          # (n, n_al) cavity residual functions evaluated on the sample
        out = []
        for c in range(0, Xs.shape[0], 25000):
            out.append(ys[c:c + 25000, None] - torch.erf(Xs[c:c + 25000] @ W.T) @ Zal)
        return torch.cat(out).numpy()
    Etr, Efr = resid(X, Y), resid(Xf, yf)
    Htr, Hfr = Hmat(X[:, 0].numpy()), Hmat(Xf[:, 0].numpy())
    # (a) plain sample averages <e He_k>, averaged over aligned cavities, with standard errors
    def avg(E, H):
        prods = (E.mean(1)[:, None] * H)                    # cavity-averaged residual times He_k
        return prods.mean(0), prods.std(0) / np.sqrt(len(H))
    mtr, setr = avg(Etr, Htr); mfr, sefr = avg(Efr, Hfr)
    # (b) least squares on He0..He13
    ltr = np.linalg.lstsq(Htr, Etr.mean(1), rcond=None)[0]; lfr = np.linalg.lstsq(Hfr, Efr.mean(1), rcond=None)[0]
    # exact population mode sum (He_k(x0) part only, k odd ≤ 9) for reference
    rho, wr, psi = rho_quadrature(d, 64, 0); qn = (W[:, 1:] ** 2).sum(1).numpy(); sg = np.sign(w0n)
    ck = lambda k: (e_k(np.abs(w0n)[:, None] / np.sqrt(1 + 2 * qn[:, None] * rho[None, :]), k) * wr[None, :]).sum(1) * sg ** k
    exact = {k: ((1.0 if k == 1 else eps if k == 3 else 0.0) - Zal.numpy().T @ ck(k)).mean() for k in (1, 3, 5, 7, 9)}
    print(f"\n{g} (P={P}, {len(al)} aligned cavities): He_k(x0) coefficient of the cavity residual")
    print(f"   {'k':>3s} | {'train <e He_k>':>17s} | {'fresh <e He_k>':>17s} | {'train lstsq':>11s} {'fresh lstsq':>11s} | {'exact':>7s} | train-fresh in s.e.")
    for k in (1, 3, 5, 7, 9, 11):
        z = (mtr[k] - mfr[k]) / np.hypot(setr[k], sefr[k])
        ex = f"{exact[k]:+7.4f}" if k in exact else "   -   "
        print(f"   {k:>3d} | {mtr[k]:+8.4f} ± {setr[k]:.4f} | {mfr[k]:+8.4f} ± {sefr[k]:.4f} | {ltr[k]:+11.4f} {lfr[k]:+11.4f} | {ex} | {z:+6.1f}")
    print(f"   residual mean square: train {np.mean(Etr.mean(1) ** 2):.4f}  fresh {np.mean(Efr.mean(1) ** 2):.4f}")
