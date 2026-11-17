#!/usr/bin/env python3
"""The perpendicular part of the pull on each hidden neuron, measured in the sampler's networks.

The theory's pull on a hidden neuron is 1/2 g~ <t | erf o h>^2 with t = (y - f_bar)/lambda (lambda = kappa/P) restricted to
the x0 tower, because its mean output has no perpendicular part. In a network it has one, f_bar_perp = f_bar minus its
x0 tower (He1..He11). Per hidden neuron k of saved networks (population averages over fresh inputs):
    s_k  = t . e_k                          x0 part   (t from the network's own population residual in He1..He11)
    c_k  = <t_perp | erf o h_k>,            t_perp = -f_bar_perp / lambda   (the full field)
    c0_k = c_k + a_bar_k |e_perp,k|^2/lambda   the same with neuron k's own contribution to f_bar_perp removed
and the x0 tower h_k (He1..He11), each neuron's readout mean a_bar_k (training set). Monte Carlo noise in c is
estimated from two independent halves of the inputs. Saved to plots/cache/perp_pull_P{P}.npz for perp_surrogate.py.

    PYTHONPATH=.:../fcn3_matrix_theory python measure_perp_pull.py [--Ps 24000 48000] [--nets 3] [--nx-log2 19]
"""
import argparse
import json

import numpy as np
import torch

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import hermite_norm
from fcn3_sampler import make_data

HERE = po.HERE


@torch.no_grad()
def one_net(W, B, abar, ks, y, lam, nx_log2, gen, chunk=2 ** 15):
    dev = W.device
    N2 = B.shape[0]
    ab = torch.tensor(abar, device=dev, dtype=torch.float64)
    nch = 2 ** nx_log2 // chunk
    seeds = torch.randint(0, 2 ** 31, (nch,), generator=torch.Generator().manual_seed(int(gen.initial_seed())))

    def batch(i):
        g = torch.Generator(device=dev).manual_seed(int(seeds[i]))
        X = torch.randn(chunk, W.shape[1], device=dev, dtype=torch.float64, generator=g)
        E = torch.erf(torch.erf(X @ W.T) @ B.T)                   # (chunk, N2)
        return hermite_norm(ks, X[:, 0]).T, E                      # (chunk, S), (chunk, N2)

    # pass 1: tower coefficients of every neuron and of f_bar, and E erf(h)^2
    e = torch.zeros(N2, len(ks), device=dev, dtype=torch.float64)
    e2 = torch.zeros(N2, device=dev, dtype=torch.float64)
    for i in range(nch):
        Ph, E = batch(i)
        e += E.T @ Ph
        e2 += (E * E).sum(0)
    e /= nch * chunk
    e2 /= nch * chunk
    fn = ab @ e                                                    # f_bar_n
    t = (torch.tensor(y, device=dev, dtype=torch.float64) - fn) / lam
    # pass 2: c_k from two halves
    c = torch.zeros(2, N2, device=dev, dtype=torch.float64)
    fperp2 = 0.0
    for i in range(nch):
        Ph, E = batch(i)
        fp = E @ ab - Ph @ fn                                      # f_bar_perp at these inputs
        c[i % 2] += fp @ E
        fperp2 += float((fp * fp).sum())
    c = -c / (nch * chunk / 2) / lam
    eperp2 = (e2 - (e * e).sum(1)).clamp_min(0)
    ch = c.mean(0)
    c0 = ch + ab * eperp2 / lam
    return dict(s=(e @ t).cpu().numpy(), c=ch.cpu().numpy(), c0=c0.cpu().numpy(), ca=c[0].cpu().numpy(),
                cb=c[1].cpu().numpy(), e=e.cpu().numpy(), t=t.cpu().numpy(), fn=fn.cpu().numpy(),
                fperp_rms=np.sqrt(fperp2 / (nch * chunk)), eperp2=eperp2.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[24000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--nx-log2", type=int, default=19)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    gen = torch.Generator().manual_seed(11)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        p = po._params(P)
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        y = np.array(p.y())
        out = {k: [] for k in ("s", "c", "c0", "ca", "cb", "h", "abar", "net", "eperp2", "e")}
        nets_t, nets_f, rms = [], [], []
        n = 0
        for r0 in [r for r in groups[P] if (r["path"] / "nets").exists()]:
            args = json.loads((r0["path"] / "config.json").read_text())["args"]
            X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
            X, Y = X.float(), Y.float()
            v, sa2 = args["kappa"] / args["chi"], args["sigA"] ** 2 / (args["N2"] * args["chi"])
            for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(f, map_location="cpu", weights_only=False)
                W = (nt["sw"] * nt["ZW"]).to(dev).double()
                B = (nt["sb"] * nt["ZB"]).to(dev).double()
                abar, _ = hr.network_parts(X, Y, W.float(), B.float(), v, sa2, ks)
                r = one_net(W, B, abar, ks, y, p.lam, a.nx_log2, torch.Generator().manual_seed(1000 + n))
                for k in ("s", "c", "c0", "ca", "cb", "eperp2", "e"):
                    out[k].append(r[k])
                out["h"].append(hr.het_h(W, B, ks))
                out["abar"].append(abar)
                out["net"].append(np.full(len(abar), n))
                nets_t.append(r["t"])
                nets_f.append(r["fn"])
                rms.append(r["fperp_rms"])
                n += 1
            del X, Y
            torch.cuda.empty_cache()
        o = {k: np.concatenate(v) for k, v in out.items()}
        np.savez(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz", **o, t=np.array(nets_t), fn=np.array(nets_f),
                 fperp_rms=np.array(rms), gt=p.gt, lam=p.lam, ks=np.array(ks))
        gt = p.gt
        sig = np.cov(o["ca"], o["cb"])[0, 1]
        print(f"\nP = {P}: {n} networks; f_bar_perp rms {np.mean(rms):.4f}; population residual (y - f_bar)_n "
              f"{np.round(np.mean(nets_t, 0) * p.lam, 4)}")
        print(f"   rms over neurons: s {np.sqrt((o['s'] ** 2).mean()):.4g}   c {np.sqrt((o['c'] ** 2).mean()):.4g}   "
              f"c (noise-free part) {np.sqrt(max(sig, 0)):.4g}   c0 {np.sqrt((o['c0'] ** 2).mean()):.4g}")
        reg = hr.regions(o["h"][:, 0], o["h"][:, 1])
        for k, m in reg.items():
            if not m.any():
                continue
            x0 = 0.5 * gt * o["s"][m] ** 2
            full = 0.5 * gt * (o["s"][m] + o["c"][m]) ** 2
            cav = 0.5 * gt * (o["s"][m] + o["c0"][m]) ** 2
            print(f"   {k:9s} ({m.sum() / n:7.1f}/net): pull, nats: x0 part {x0.mean():9.2f}   x0 + perp {full.mean():9.2f}"
                  f"   x0 + perp without self {cav.mean():9.2f};   corr(s, c) {np.corrcoef(o['s'][m], o['c'][m])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
