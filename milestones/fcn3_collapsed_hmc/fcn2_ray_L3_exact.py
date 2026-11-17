#!/usr/bin/env python3
"""FCN2 learnabilities along the kappa0 = 0.015 ray from every saved sample of the listed chains.

For each sample (fcn2_collapsed_hmc/runs/<chain>/samples/iter_*.pt, all post-burn samples) the readout is replaced by
its posterior mean given the read-in weights, abar = (Phi^T Phi / v + I / s_a2)^-1 Phi^T y / v, and the population
He_k(x0) coefficients of that predictor are taken in closed form, fbar_k = sum_j abar_j a_k(|w_j|) (w_j0/|w_j|)^k
(the estimator of fcn2_collapsed_hmc/chains.chain_fbar, there on 15 samples per chain). L_k = fbar_k / y_k, averaged
over the samples of a chain.

    python fcn2_ray_L3_exact.py   ->  plots/cache/fcn2_ray_L3_exact.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FCN2 = HERE.parent / "fcn2_collapsed_hmc"
sys.path.insert(0, str(FCN2))
from analyze_large_d import a_k          # noqa: E402
from collapsed_hmc_fcn2 import make_data  # noqa: E402

CHAINS = {
    75: ["20261029-173200_larged_d75_d75_P750_N1500_file_r1", "20261029-173200_larged_d75_d75_P750_N1500_prior_r1",
         "20261029-173200_larged_d75_d75_P750_N1500_small_r1"],
    150: ["20261027-105544_pscan_P3000_d150_P3000_N1500_file_r1", "20261027-105544_pscan_P3000_d150_P3000_N1500_prior_r1",
          "20261027-105544_pscan_P3000_d150_P3000_N1500_small_r1"],
    300: ["20261101-171022_mix_d300_prior_d300_P12000_N1500_prior_r1",
          "20261101-171811_mix_d300_theory_d300_P12000_N1500_file_r1",
          "20261101-172602_mix_d300_overaligned_d300_P12000_N1500_file_r1"],
    600: ["20261101-173859_w0q_d600_prior_d600_P48000_N1500_prior_r1",
          "20261101-175730_w0q_d600_overaligned_d600_P48000_N1500_file_r1"],
    800: ["20261101-181623_w0q_d800_prior_d800_P85333_N1500_prior_r1",
          "20261101-184840_w0q_d800_overaligned_d800_P85333_N1500_file_r1"],
    1200: ["20261101-192054_w0q_d1200_prior_d1200_P192000_N1500_prior_r1"],
}
KS = (1, 3, 5, 7, 9, 11)


@torch.no_grad()
def main():
    dev = torch.device("cuda:0")
    out = {}
    for d, chains in CHAINS.items():
        out[d] = {}
        X = Y = None
        for name in chains:
            run = FCN2 / "runs" / name
            a = json.loads((run / "config.json").read_text())["args"]
            if X is None:
                X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev)
                X, Y = X.double(), Y.double()
            sa2, v = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
            files = sorted(glob.glob(str(run / "samples" / "iter_*.pt")))
            fk = []
            for f in files:
                W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
                Phi = torch.erf(X @ W.T)
                M = Phi.T @ Phi / v + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
                A = torch.linalg.solve(M, Phi.T @ Y / v)
                s = W.norm(dim=1)
                u0 = W[:, 0] / s
                fk.append([float((A * a_k(s, k) * u0 ** k).sum()) for k in KS])
                del Phi, M
            fk = np.array(fk)
            L1, L3 = fk[:, 0] / a["he1"], fk[:, 1] / a["eps"]
            out[d][name] = dict(n=len(files), L1=float(L1.mean()), L3=float(L3.mean()), L3_sd=float(L3.std()),
                                init=a["init"], tag=a.get("tag"))
            print(f"d={d} {name}: {len(files)} samples, L1 {L1.mean():.4f}, L3 {L3.mean():.4f} (sd over samples "
                  f"{L3.std():.4f})", flush=True)
        del X, Y
        torch.cuda.empty_cache()
    (HERE / "plots" / "cache" / "fcn2_ray_L3_exact.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
