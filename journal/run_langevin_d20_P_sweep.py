#!/usr/bin/env python3
"""Langevin MF P-sweep over d, 50k epochs, same hyps as LearningCubic.ipynb cell 9."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/akiva/FCNX-Ensembling/lib")
from FCN2Network import FCN2NetworkActivationGeneric
from kappa_eff_solver import compute_kappa_eff

OUT_DIR = "/home/akiva/FCNX-Ensembling/journal/LearningCubic_models"
os.makedirs(OUT_DIR, exist_ok=True)

INV_SQRT_6 = 1.0 / np.sqrt(6)

N_ld = 400
eps_ld = 0.5
chi_ld = float(N_ld)
s0_ld = 1.0
# None → classic readout: σ_A² = 1/(N χ), wd_A = N χ T_eff (same as sa0=1).
# Set (e.g. via --sa0) → σ_A² = sa0/(N χ), wd_A = (N χ / sa0) T_eff.
# Steepwell probes use chi=1 so this reduces to sa0/N.
sa0_ld = None
T_ld = 1.0
ens_ld = 1
base_lr_ld = 0.01
test_P_ld = 2000
seed_ld = 42
N_PER_D = None  # if set, N = chi = N_PER_D * d (overrides N_ld / chi_ld)

D_SWEEP_LD = [5, 10, 20, 30]
P_SWEEP_LD = [50, 100, 500, 1000, 2000, 3000, 8000, 16000]
EPOCHS_SWEEP_LD = 50_000
LOG_SWEEP_LD = 2_000
JULIA_VGA = Path("/home/akiva/FCNX-Ensembling/julia_lib/fcn2_vga_erf.jl")
JULIA_CLASSICAL = Path("/home/akiva/FCNX-Ensembling/julia_lib/compute_fcn2_erf_cubic_eigs.jl")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return device


def f_target(x, eps):
    vx = x[:, 0]
    return vx + eps * INV_SQRT_6 * (vx**3 - 3 * vx)


def He3(vx):
    return (vx**3 - 3 * vx) * INV_SQRT_6


def gen(P, d, eps, device):
    x = torch.randn(P, d, device=device)
    return x, f_target(x, eps).unsqueeze(1)


def langevin_lr_divisor(epoch, epochs):
    if epoch <= epochs // 3:
        return 1.0
    if epoch <= (2 * epochs) // 3:
        return 3.0
    return 5.0


def mean_mse(model, x, y):
    f = model(x)
    return torch.mean((f - y) ** 2).item()


def he3_ratio(model, x, y):
    f = model(x)
    he1 = x[:, :1]
    he3 = He3(x[:, 0]).unsqueeze(1)
    inner_f_he1 = torch.mean(f * he1)
    inner_y_he3 = torch.mean(y * he3)
    f_res = f - inner_f_he1 * he1
    num = torch.mean(f_res * he3)
    if inner_y_he3.abs() <= 1e-8:
        return 0.0
    return (num / inner_y_he3).item()


def N_chi_for(d):
    """Hidden width and χ. Default N=400, χ=N; with N_PER_D set, N=χ=N_PER_D·d."""
    if N_PER_D:
        N = int(N_PER_D) * int(d)
        return N, float(N)
    return int(N_ld), float(chi_ld)


def scaling_for_d(d, chi=None, N=None):
    """Return (sigma_w0, wd_W0, sigma_a, wd_A, t_eff).

    σ_A² = sa0/(N χ) with sa0=1 when sa0_ld is None (classic MF).
    wd_A = (1/σ_A²) · T_eff = (N χ / sa0) · T_eff.
    """
    if chi is None or N is None:
        N_def, chi_def = N_chi_for(d)
        if N is None:
            N = N_def
        if chi is None:
            chi = chi_def
    t_eff = T_ld / float(chi)
    sigma_w0 = s0_ld / d
    wd_W0 = (1.0 / sigma_w0) * t_eff
    sa0 = 1.0 if sa0_ld is None else float(sa0_ld)
    sigma_a = sa0 / (float(N) * float(chi))
    wd_A = (1.0 / sigma_a) * t_eff
    return sigma_w0, wd_W0, sigma_a, wd_A, t_eff


def sweep_stem():
    """Filename prefix for sweep plots/summaries (matches N/chi convention)."""
    if N_PER_D:
        return f"N{N_PER_D}d"
    return f"N{N_ld}"


def _prior_tag():
    """Extra filename fragment when s0/sa0 differ from classic defaults."""
    parts = []
    if abs(float(s0_ld) - 1.0) > 1e-12:
        parts.append(f"s0{s0_ld:g}")
    if sa0_ld is not None:
        parts.append(f"sa0{float(sa0_ld):g}")
    return ("_" + "_".join(parts)) if parts else ""


def sweep_basename(suffix):
    eps_tag = "" if abs(float(eps_ld) - 0.5) < 1e-12 else f"_eps{eps_ld}"
    return f"langevin_mf_{sweep_stem()}_P_sweep_ep{EPOCHS_SWEEP_LD}{eps_tag}{_prior_tag()}{suffix}"


def langevin_tag(d, P, epochs=None, N=None, chi=None):
    if epochs is None:
        epochs = EPOCHS_SWEEP_LD
    if N is None or chi is None:
        N, chi = N_chi_for(d)
    chi_tag = int(chi) if float(chi) == int(chi) else f"{chi:g}"
    return (
        f"langevin_mf_d{d}_N{N}_P{P}_chi{chi_tag}"
        f"_lr{base_lr_ld}_T{T_ld}_eps{eps_ld}{_prior_tag()}_seed{seed_ld}_ep{epochs}"
    )


def train_langevin_mf_one(d, P, device, epochs=None, log_every=None):
    if epochs is None:
        epochs = EPOCHS_SWEEP_LD
    if log_every is None:
        log_every = LOG_SWEEP_LD
    N, chi = N_chi_for(d)
    sigma_w0, wd_W0, sigma_a_run, wd_A_run, t_eff = scaling_for_d(d, chi=chi, N=N)
    tag = langevin_tag(d, P, epochs=epochs, N=N, chi=chi)
    hist_path = os.path.join(OUT_DIR, f"{tag}_history.npz")
    ckpt_path = os.path.join(OUT_DIR, f"{tag}_model.pt")

    torch.manual_seed(seed_ld)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_ld)
    x_tr, y_tr = gen(P, d, eps_ld, device)
    x_te, y_te = gen(test_P_ld, d, eps_ld, device)

    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=P,
        ens=ens_ld,
        activation="erf",
        weight_initialization_variance=(sigma_w0, sigma_a_run),
        device=device,
    ).to(device)
    model.train()

    epochs_log, tr_mse, te_mse, he3s, lrs = [], [], [], [], []
    start_epoch = 1
    if os.path.exists(hist_path):
        hist = dict(np.load(hist_path))
        last = int(hist["epoch"][-1])
        if last >= epochs:
            print(f"  skip d={d} P={P}: loaded complete {os.path.basename(hist_path)}", flush=True)
            return hist, ckpt_path
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            epochs_log = [int(x) for x in hist["epoch"]]
            tr_mse = [float(x) for x in hist["train_mse"]]
            te_mse = [float(x) for x in hist["test_mse"]]
            he3s = [float(x) for x in hist["he3_test"]]
            lrs = [float(x) for x in hist["lr"]]
            start_epoch = last + 1
            print(
                f"  resume d={d} P={P} from epoch {last}  {os.path.basename(hist_path)}",
                flush=True,
            )
        else:
            print(
                f"  incomplete hist without ckpt; restarting d={d} P={P}",
                flush=True,
            )

    last_div = langevin_lr_divisor(max(start_epoch - 1, 1), epochs) if start_epoch > 1 else None
    sa0_eff = 1.0 if sa0_ld is None else float(sa0_ld)
    sa0_desc = f"{sa0_eff:g}" + ("" if sa0_ld is not None else " (classic default)")
    print(
        f"  Langevin d={d} P={P}  epochs={epochs}  N={N} chi={chi:g}  "
        f"s0={s0_ld:g} sigmaW0={sigma_w0:.4g}  sa0={sa0_desc} sigmaA={sigma_a_run:.4g}  "
        f"wd_W0={wd_W0:.4g} wd_A={wd_A_run:.4g}  T_eff={t_eff:.4g}  device={device}",
        flush=True,
    )

    def snap(epoch, lr):
        with torch.no_grad():
            epochs_log.append(epoch)
            tr_mse.append(mean_mse(model, x_tr, y_tr))
            te_mse.append(mean_mse(model, x_te, y_te))
            he3s.append(he3_ratio(model, x_te, y_te))
            lrs.append(lr)

    def dump():
        np.savez(
            hist_path,
            epoch=np.asarray(epochs_log),
            train_mse=np.asarray(tr_mse),
            test_mse=np.asarray(te_mse),
            he3_test=np.asarray(he3s),
            lr=np.asarray(lrs),
            P=np.asarray(P),
            d=np.asarray(d),
            N=np.asarray(N),
            chi=np.asarray(chi),
            s0=np.asarray(float(s0_ld)),
            sa0=np.asarray(float(sa0_ld) if sa0_ld is not None else np.nan),
            sigmaW0=np.asarray(sigma_w0),
            sigmaA=np.asarray(sigma_a_run),
            T=np.asarray(float(T_ld)),
        )
        torch.save(model.state_dict(), ckpt_path)

    if start_epoch == 1:
        snap(0, base_lr_ld / P)
        print(
            f"    epoch {0:6d}  train={tr_mse[-1]:.4f}  test={te_mse[-1]:.4f}  He3={he3s[-1]:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        div = langevin_lr_divisor(epoch, epochs)
        if div != last_div:
            print(f"    schedule epoch {epoch}: lr -> lr0/{div:g}", flush=True)
            last_div = div
        lr = (base_lr_ld / div) / P
        noise_scale = np.sqrt(2.0 * lr * t_eff)
        output = model(x_tr)
        loss = torch.sum((output - y_tr) ** 2)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                wd = wd_W0 if name == "W0" else (wd_A_run if name == "A" else 0.0)
                param.add_(-lr * param.grad)
                if wd:
                    param.add_(-lr * wd * param.data)
                param.add_(torch.randn_like(param) * noise_scale)
        if epoch % log_every == 0 or epoch == epochs:
            snap(epoch, lr)
            if epoch % (log_every * 5) == 0 or epoch == epochs:
                print(
                    f"    epoch {epoch:6d}  train={tr_mse[-1]:.4f}  "
                    f"test={te_mse[-1]:.4f}  He3={he3s[-1]:.4f}",
                    flush=True,
                )
            dump()

    hist = {
        "epoch": np.asarray(epochs_log),
        "train_mse": np.asarray(tr_mse),
        "test_mse": np.asarray(te_mse),
        "he3_test": np.asarray(he3s),
        "lr": np.asarray(lrs),
        "P": np.asarray(P),
        "d": np.asarray(d),
    }
    dump()
    print(f"    saved {os.path.basename(hist_path)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hist, ckpt_path


def gpr_cache_path(d, P):
    return os.path.join(OUT_DIR, f"{langevin_tag(d, P)}_gpr.npz")


def vga_cache_path(d, P, offdiag=False):
    suffix = "_vga_offdiag.json" if offdiag else "_vga.json"
    return os.path.join(OUT_DIR, f"{langevin_tag(d, P)}{suffix}")


def classical_cache_path(d, P):
    return os.path.join(OUT_DIR, f"{langevin_tag(d, P)}_classical.json")


def _extract_vga_target(payload):
    if not isinstance(payload, dict):
        return {}
    vga = payload.get("vga", payload)
    if isinstance(vga, dict) and "target" in vga and isinstance(vga["target"], dict):
        return vga["target"]
    if isinstance(vga, dict) and "lJ3" in vga:
        return vga
    return {}


def _extract_classical_target(payload):
    if not isinstance(payload, dict):
        return {}
    if "target" in payload and isinstance(payload["target"], dict):
        return payload["target"]
    return payload


def _he3_from_lJ3(target, kappa, P, prefer_reported=True):
    if prefer_reported:
        he3 = target.get("learnability3")
        if he3 is not None and np.isfinite(float(he3)):
            return float(he3)
    lJ3 = target.get("lJ3", target.get("lJ3T"))
    if lJ3 is not None and np.isfinite(float(lJ3)):
        return float(lJ3) / (float(lJ3) + float(kappa) / float(P))
    he3 = target.get("learnability3")
    return float(he3) if he3 is not None and np.isfinite(float(he3)) else float("nan")


def _run_julia_theory(cmd, cache_path, label, d, P, kappa):
    payload = None
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            payload = json.load(f)
        cached_kappa = payload.get("parameters", {}).get("kappa")
        if cached_kappa is not None and abs(float(cached_kappa) - float(kappa)) > 1e-8:
            payload = None
    if payload is not None:
        return payload
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cmd = list(cmd) + ["--to", str(tmp_path), "--quiet"]
        print(f"    {label} solve d={d} P={P} kappa={float(kappa):.4g}", flush=True)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if proc.stderr:
            err_tail = proc.stderr.strip().splitlines()[-5:]
            if err_tail:
                print("      julia stderr: " + " | ".join(err_tail), flush=True)
        with open(tmp_path, "r") as f:
            payload = json.load(f)
        with open(cache_path, "w") as f:
            json.dump(payload, f)
        return payload
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "")[-1500:]
        print(f"    {label} failed d={d} P={P}: {exc}\n{err}", flush=True)
        return None
    except Exception as exc:
        print(f"    {label} failed d={d} P={P}: {exc}", flush=True)
        return None
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def vga_he3_for_P(d, P, kappa, advanced=True, offdiag=True):
    """VGA cubic learnability λ3 / (λ3 + κ/P), same solver as fcn2_erf milestones.

    ``advanced=True`` (default) uses exact Gauss–Hermite GMM entropy
    (``fcn2_vga_erf.jl --advanced``). ``offdiag=True`` rotates the teacher
    into the He1–He3 kernel eigenbasis each residual.
    """
    N, chi = N_chi_for(d)
    cmd = [
        "julia",
        str(JULIA_VGA),
        "--d", str(int(d)),
        "--n1", str(int(N)),
        "--P", str(int(P)),
        "--chi", str(float(chi)),
        "--kappa", str(float(kappa)),
        "--epsilon", str(float(eps_ld)),
        "--s0", str(float(s0_ld)),
    ]
    if sa0_ld is not None:
        cmd.extend(["--sa0", str(float(sa0_ld))])
    if advanced:
        cmd.append("--advanced")
    if offdiag:
        cmd.append("--offdiag")
    cache_path = vga_cache_path(d, P, offdiag=offdiag)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        params = cached.get("parameters", {})
        cached_kappa = params.get("kappa")
        cached_adv = bool(params.get("advanced", False))
        cached_off = bool(params.get("offdiag", False))
        kappa_ok = cached_kappa is not None and abs(float(cached_kappa) - float(kappa)) <= 1e-8
        if not kappa_ok or cached_adv != bool(advanced) or cached_off != bool(offdiag):
            os.remove(cache_path)
    label = "VGA offdiag" if offdiag else "VGA"
    payload = _run_julia_theory(cmd, cache_path, label, d, P, kappa)
    if payload is None:
        return float("nan")
    return _he3_from_lJ3(_extract_vga_target(payload), kappa, P, prefer_reported=True)


def classical_he3_for_P(d, P, kappa):
    """Gaussian Laplace (non-VGA) cubic learnability from FCS2Erf_Cubic.jl.

    Uses the hidden-kernel eigenvalue λ_{J3} so the overlay is on the same
    formula as VGA: λ3 / (λ3 + κ/P).
    """
    N, chi = N_chi_for(d)
    cmd = [
        "julia",
        str(JULIA_CLASSICAL),
        "--d", str(int(d)),
        "--n1", str(int(N)),
        "--P", str(int(P)),
        "--chi", str(float(chi)),
        "--kappa", str(float(kappa)),
        "--epsilon", str(float(eps_ld)),
        "--anneal_steps", "3000",
    ]
    payload = _run_julia_theory(cmd, classical_cache_path(d, P), "classical", d, P, kappa)
    if payload is None:
        return float("nan")
    return _he3_from_lJ3(_extract_classical_target(payload), kappa, P, prefer_reported=False)


def vga_he3_curve(d, P_list, device):
    out = []
    for P in P_list:
        _he3_g, _mse_g, kappa, _lab = gpr_he3_for_P(d, P, device)
        he3_v = vga_he3_for_P(d, P, kappa)
        out.append(he3_v)
    return np.asarray(out, dtype=float)


def vga_he3_from_gpr_rows(d, gpr_rows, workers=4):
    """Offdiag VGA He3 using κ already stored in gpr_rows (no GPU)."""
    def one(row):
        P, _he3, _mse, kappa, _lab = row
        return int(P), vga_he3_for_P(d, int(P), float(kappa))

    by_P = {}
    n = max(len(gpr_rows), 1)
    workers = max(1, min(int(workers), n))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, row) for row in gpr_rows]
        for fut in as_completed(futs):
            P, he3 = fut.result()
            by_P[P] = he3
    return np.asarray([by_P[int(row[0])] for row in gpr_rows], dtype=float)


def classical_he3_curve(d, P_list, device):
    out = []
    for P in P_list:
        _he3_g, _mse_g, kappa, _lab = gpr_he3_for_P(d, P, device)
        he3_c = classical_he3_for_P(d, P, kappa)
        out.append(he3_c)
    return np.asarray(out, dtype=float)


def gpr_he3_for_P(d, P, device):
    cache_path = gpr_cache_path(d, P)
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return float(z["he3"]), float(z["mse"]), float(z["sigma2"]), str(z["lab"])

    torch.manual_seed(seed_ld)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_ld)
    X_tr, y_tr = gen(P, d, eps_ld, device)
    X_te, y_te = gen(test_P_ld, d, eps_ld, device)
    y_tr = y_tr.squeeze(-1)
    y_te = y_te.squeeze(-1)
    kappa_bare = float(T_ld) / 2.0
    try:
        sigma2 = float(
            compute_kappa_eff(
                d=int(d),
                P=int(P),
                kappa_bare=kappa_bare,
                n1=int(N_chi_for(d)[0]),
                chi=float(N_chi_for(d)[1]),
                device=device,
                verbose=False,
            )
        )
        lab = "kappa_eff"
    except Exception as exc:
        print(f"    kappa_eff failed for d={d} P={P} ({exc}); using kappa_bare", flush=True)
        sigma2, lab = kappa_bare, "kappa_bare"
    d_in = X_tr.shape[1]

    def K(X, X2=None):
        if X2 is None:
            X2 = X
        gram = (X @ X2.T) / d_in
        sx = torch.sqrt(1.0 + 2.0 * torch.sum(X * X, dim=1) / d_in)
        sy = torch.sqrt(1.0 + 2.0 * torch.sum(X2 * X2, dim=1) / d_in)
        arg = torch.clamp(2.0 * gram / (sx[:, None] * sy[None, :]), -1.0 + 1e-6, 1.0 - 1e-6)
        return (2.0 / torch.pi) * torch.arcsin(arg)

    with torch.no_grad():
        eye = torch.eye(P, device=device, dtype=X_tr.dtype)
        alpha = torch.linalg.solve(K(X_tr) + sigma2 * eye, y_tr)
        f_te = K(X_te, X_tr) @ alpha
        he1 = X_te[:, 0]
        he3 = He3(X_te[:, 0])
        f_res = f_te - torch.mean(f_te * he1) * he1
        denom = torch.mean(y_te * he3)
        he3_val = (torch.mean(f_res * he3) / denom).item() if denom.abs() > 1e-8 else float("nan")
        mse = torch.mean((f_te - y_te) ** 2).item()
    np.savez(cache_path, he3=he3_val, mse=mse, sigma2=sigma2, lab=np.asarray(lab))
    return he3_val, mse, sigma2, lab


def collect_from_disk(dims, P_list, device):
    """Load completed Langevin histories and (re)compute GPR overlays."""
    by_d = {}
    for d in dims:
        hists, gpr_rows = {}, []
        complete = True
        for P in P_list:
            tag = langevin_tag(d, P)
            hist_path = os.path.join(OUT_DIR, f"{tag}_history.npz")
            if not os.path.exists(hist_path):
                complete = False
                break
            hist = dict(np.load(hist_path))
            if int(hist["epoch"][-1]) < EPOCHS_SWEEP_LD:
                complete = False
                break
            hists[P] = hist
            he3_g, mse_g, s2, lab = gpr_he3_for_P(d, P, device)
            gpr_rows.append((P, he3_g, mse_g, s2, lab))
        if complete:
            by_d[d] = (hists, gpr_rows)
    return by_d


def P_to_alpha(P, d):
    """α such that P = d^α."""
    return np.log(np.asarray(P, dtype=float)) / np.log(float(d))


ALPHA_GUIDES = (1.0, 1.5, 2.0, 2.5, 3.0)


def _style_alpha_axis(ax, xmin=None, xmax=None):
    ax.set_xlabel(r"$\alpha$  ($P = d^{\alpha}$)")
    ax.grid(True, alpha=0.3)
    for a in ALPHA_GUIDES:
        ax.axvline(a, color="0.75", ls=":", lw=0.9, zorder=0)
    if xmin is not None and xmax is not None:
        pad = 0.08 * max(xmax - xmin, 0.5)
        ax.set_xlim(xmin - pad, xmax + pad)
    ticks = sorted(set(ALPHA_GUIDES) | set(np.round(np.arange(1.0, 6.5, 0.5), 10)))
    ax.set_xticks([t for t in ticks if xmin is None or xmax is None or (xmin - 0.2 <= t <= xmax + 0.2)])


def plot_all_d(by_d, P_list, show=False, vga_by_d=None, class_by_d=None):
    P_arr = np.asarray(P_list, dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    alpha_lo, alpha_hi = np.inf, -np.inf
    for i, d in enumerate(sorted(by_d)):
        hists, gpr_rows = by_d[d]
        color = f"C{i}"
        x = P_to_alpha(P_arr, d)
        alpha_lo = min(alpha_lo, float(np.min(x)))
        alpha_hi = max(alpha_hi, float(np.max(x)))
        he3_ld = np.asarray([hists[P]["he3_test"][-1] for P in P_list])
        mse_ld = np.asarray([hists[P]["test_mse"][-1] for P in P_list])
        he3_gpr = np.asarray([r[1] for r in gpr_rows])
        mse_gpr = np.asarray([r[2] for r in gpr_rows])
        ax[0].plot(x, he3_ld, "o-", color=color, lw=2, label=f"Langevin d={d}")
        ax[0].plot(x, he3_gpr, "s--", color=color, lw=1.4, alpha=0.85, label=f"GPR d={d}")
        if vga_by_d is not None and d in vga_by_d:
            ax[0].plot(
                x,
                vga_by_d[d],
                "^:",
                color=color,
                lw=1.8,
                ms=5,
                label=f"VGA offdiag d={d}",
            )
        if class_by_d is not None and d in class_by_d:
            ax[0].plot(
                x,
                class_by_d[d],
                "D-.",
                color=color,
                lw=1.5,
                ms=4.5,
                alpha=0.9,
                label=f"classical d={d}",
            )
        ax[1].plot(x, mse_ld, "o-", color=color, lw=2, label=f"Langevin d={d}")
        ax[1].plot(x, mse_gpr, "s--", color=color, lw=1.4, alpha=0.85, label=f"GPR d={d}")
    ax[0].set_ylabel("test He3")
    ax[0].set_title(r"cubic learnability vs $\alpha$")
    ax[0].legend(fontsize=6.5, ncol=2, loc="lower right")
    ax[1].set_ylabel("test MSE")
    ax[1].set_title(r"test MSE vs $\alpha$")
    ax[1].legend(fontsize=8, ncol=2)
    _style_alpha_axis(ax[0], alpha_lo, alpha_hi)
    _style_alpha_axis(ax[1], alpha_lo, alpha_hi)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, sweep_basename(".png"))
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig_path


def plot_train_test_mse_per_d(by_d, P_list, show=False):
    """One figure per d: train and test MSE vs epoch, all P overlaid."""
    cmap = plt.cm.viridis
    paths = []
    for d in sorted(by_d):
        hists, gpr_rows = by_d[d]
        gpr_mse = {int(row[0]): float(row[2]) for row in gpr_rows}
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
        nP = max(len(P_list) - 1, 1)
        for i, P in enumerate(P_list):
            hist = hists[P]
            color = cmap(i / nP)
            ep = np.asarray(hist["epoch"])
            ax[0].plot(ep, hist["train_mse"], color=color, lw=1.8, label=rf"$P={P}$")
            ax[1].plot(ep, hist["test_mse"], color=color, lw=1.8, label=rf"$P={P}$")
            if P in gpr_mse and np.isfinite(gpr_mse[P]):
                ax[1].axhline(gpr_mse[P], color=color, ls=":", lw=1.2, alpha=0.85)
        for a in ax:
            a.axvline(EPOCHS_SWEEP_LD / 3, color="0.6", ls="--", lw=0.8)
            a.axvline(2 * EPOCHS_SWEEP_LD / 3, color="0.6", ls="--", lw=0.8)
            a.set_xlabel("epoch")
            a.grid(True, alpha=0.3)
            a.set_xlim(0, EPOCHS_SWEEP_LD)
        ax[0].set_ylabel("MSE")
        ax[0].set_title(f"train MSE  d={d}")
        ax[1].set_title(f"test MSE  d={d}  (dotted = GPR)")
        ax[0].legend(fontsize=7, ncol=2, loc="upper right")
        fig.suptitle(f"Langevin train/test MSE vs epoch, all $P$  (d={d})")
        fig.tight_layout()
        fig_path = os.path.join(
            OUT_DIR, f"langevin_mf_{sweep_stem()}_d{d}_train_test_mse_vs_epoch.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}", flush=True)
        if show:
            plt.show()
        else:
            plt.close(fig)
        paths.append(fig_path)
    return paths


def plot_collapse(by_d, P_list, show=False, vga_by_d=None, class_by_d=None):
    """He3 vs α where P = d^α, so the learnability onset is readable on the x-axis."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    dims = sorted(by_d)
    P_arr = np.asarray(P_list, dtype=float)
    alpha_lo, alpha_hi = np.inf, -np.inf
    for i, d in enumerate(dims):
        hists, gpr_rows = by_d[d]
        x = P_to_alpha(P_arr, d)
        alpha_lo = min(alpha_lo, float(np.min(x)))
        alpha_hi = max(alpha_hi, float(np.max(x)))
        he3_ld = np.asarray([hists[P]["he3_test"][-1] for P in P_list])
        he3_gpr = np.asarray([r[1] for r in gpr_rows])
        ax.plot(x, he3_ld, "o-", color=f"C{i}", lw=2, label=f"Langevin d={d}")
        ax.plot(x, he3_gpr, "s--", color=f"C{i}", lw=1.4, alpha=0.85, label=f"GPR d={d}")
        if vga_by_d is not None and d in vga_by_d:
            ax.plot(
                x,
                vga_by_d[d],
                "^:",
                color=f"C{i}",
                lw=1.8,
                ms=5,
                label=f"VGA offdiag d={d}",
            )
        if class_by_d is not None and d in class_by_d:
            ax.plot(
                x,
                class_by_d[d],
                "D-.",
                color=f"C{i}",
                lw=1.5,
                ms=4.5,
                alpha=0.9,
                label=f"classical d={d}",
            )
    ax.set_ylabel("test He3")
    ax.set_title(r"cubic learnability vs $\alpha$")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    _style_alpha_axis(ax, alpha_lo, alpha_hi)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, sweep_basename("_collapse.png"))
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig_path


HE3_THRESH = 0.5


def crossing_P(P_arr, he3, thresh=HE3_THRESH):
    """Smallest P where He3 exceeds thresh, log-interpolated from the previous point."""
    P_arr = np.asarray(P_arr, dtype=float)
    he3 = np.asarray(he3, dtype=float)
    above = he3 > thresh
    if not np.any(above):
        return np.nan
    i = int(np.argmax(above))
    if i == 0:
        return float(P_arr[0])
    P0, P1 = P_arr[i - 1], P_arr[i]
    h0, h1 = he3[i - 1], he3[i]
    if h1 == h0:
        return float(P1)
    t = float(np.clip((thresh - h0) / (h1 - h0), 0.0, 1.0))
    return float(np.exp(np.log(P0) + t * (np.log(P1) - np.log(P0))))


def fit_log_exponent(d, P_c):
    d = np.asarray(d, dtype=float)
    P_c = np.asarray(P_c, dtype=float)
    mask = np.isfinite(P_c) & (P_c > 0) & (d > 0)
    d, P_c = d[mask], P_c[mask]
    if len(d) < 2:
        return np.nan, np.nan, np.nan, d, P_c
    logd, logP = np.log(d), np.log(P_c)
    alpha, logC = np.polyfit(logd, logP, 1)
    pred = alpha * logd + logC
    ss_res = np.sum((logP - pred) ** 2)
    ss_tot = np.sum((logP - np.mean(logP)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(alpha), float(np.exp(logC)), float(r2), d, P_c


def threshold_P_by_model(by_d, P_list, thresh=HE3_THRESH):
    dims = np.asarray(sorted(by_d), dtype=float)
    P_arr = np.asarray(P_list, dtype=float)
    out = {}
    getters = {
        "Langevin": lambda hists, gpr_rows: np.asarray(
            [hists[P]["he3_test"][-1] for P in P_list]
        ),
        "GPR": lambda hists, gpr_rows: np.asarray([r[1] for r in gpr_rows]),
    }
    for name, getter in getters.items():
        P_c = np.array([crossing_P(P_arr, getter(*by_d[int(d)]), thresh) for d in dims])
        alpha, C, r2, d_fit, P_fit = fit_log_exponent(dims, P_c)
        out[name] = {
            "d": dims,
            "P_c": P_c,
            "alpha": alpha,
            "C": C,
            "r2": r2,
            "d_fit": d_fit,
            "P_fit": P_fit,
            "getter": getter,
        }
    return out


def plot_he3_threshold_scaling(by_d, P_list, show=False, thresh=HE3_THRESH):
    """(d, P) frontier where He3 crosses thresh, plus log-log exponent."""
    stats = threshold_P_by_model(by_d, P_list, thresh=thresh)
    P_arr = np.asarray(P_list, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    d_line = np.geomspace(4, 40, 80)
    colors = {"Langevin": "C0", "GPR": "C2"}
    for ax, name in zip(axes, ("Langevin", "GPR")):
        getter = stats[name]["getter"]
        color = colors[name]
        first = True
        for d in sorted(by_d):
            he3 = getter(*by_d[d])
            below = he3 <= thresh
            above = ~below
            if np.any(below):
                ax.scatter(
                    np.full(int(below.sum()), d),
                    P_arr[below],
                    s=36,
                    facecolors="none",
                    edgecolors="0.55",
                    linewidths=1.2,
                    zorder=3,
                    label=rf"He$_3\leq {thresh}$" if first else None,
                )
            if np.any(above):
                ax.scatter(
                    np.full(int(above.sum()), d),
                    P_arr[above],
                    s=42,
                    c=color,
                    zorder=4,
                    label=rf"He$_3>{thresh}$" if first else None,
                )
            first = False
        st = stats[name]
        finite = np.isfinite(st["P_c"])
        ax.scatter(
            st["d"][finite],
            st["P_c"][finite],
            marker="*",
            s=160,
            c="k",
            zorder=5,
            label=rf"$P_c$ (He$_3={thresh}$)",
        )
        for d, Pc, ok in zip(st["d"], st["P_c"], finite):
            if not ok:
                ax.annotate(
                    "no cross",
                    (d, P_arr[-1]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="0.4",
                )
        if np.isfinite(st["alpha"]):
            ax.plot(
                d_line,
                st["C"] * d_line ** st["alpha"],
                "--",
                color="k",
                lw=1.6,
                label=rf"$\alpha={st['alpha']:.2f}$  ($R^2={st['r2']:.2f}$)",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$d$")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(sorted(by_d))
        ax.set_xticklabels([str(d) for d in sorted(by_d)])
    axes[0].set_ylabel(r"$P$")
    fig.suptitle(rf"He$_3>{thresh}$ frontier in $(d,P)$")
    fig.tight_layout()
    tag = f"{thresh:g}".replace(".", "p")
    fig_path = os.path.join(OUT_DIR, sweep_basename(f"_he3_threshold_{tag}.png"))
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}", flush=True)
    print(
        f"\nHe3 > {thresh} crossing  (log-interpolated P_c;  log P_c = α log d + c)",
        flush=True,
    )
    print(f"{'model':<10} {'d':>4} {'P_c':>10}  {'alpha':>8}  {'R^2':>6}", flush=True)
    for name, st in stats.items():
        alpha_s = f"{st['alpha']:.3f}" if np.isfinite(st["alpha"]) else "nan"
        r2_s = f"{st['r2']:.3f}" if np.isfinite(st["r2"]) else "nan"
        for d, Pc in zip(st["d"], st["P_c"]):
            pc_s = f"{Pc:10.1f}" if np.isfinite(Pc) else f"{'no cross':>10}"
            print(f"{name:<10} {int(d):4d} {pc_s}  {alpha_s:>8}  {r2_s:>6}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig_path, stats


def by_d_from_summary(path):
    """Rebuild plot inputs from a saved sweep summary (no GPU / VGA)."""
    z = np.load(path)
    P_list = [int(p) for p in z["P"]]
    dims = [int(d) for d in z["d"]]
    by_d, vga_by_d, class_by_d = {}, {}, {}
    for d in dims:
        hists, gpr_rows = {}, []
        he3_ld = np.asarray(z[f"he3_langevin_d{d}"])
        mse_ld = np.asarray(z[f"mse_langevin_d{d}"])
        he3_g = np.asarray(z[f"he3_gpr_d{d}"])
        mse_g = np.asarray(z[f"mse_gpr_d{d}"])
        s2 = np.asarray(z[f"sigma2_d{d}"])
        for i, P in enumerate(P_list):
            hists[P] = {
                "he3_test": np.asarray([he3_ld[i]]),
                "test_mse": np.asarray([mse_ld[i]]),
            }
            gpr_rows.append((P, float(he3_g[i]), float(mse_g[i]), float(s2[i]), "summary"))
        by_d[d] = (hists, gpr_rows)
        if f"he3_vga_d{d}" in z.files:
            vga_by_d[d] = np.asarray(z[f"he3_vga_d{d}"])
        if f"he3_classical_d{d}" in z.files:
            class_by_d[d] = np.asarray(z[f"he3_classical_d{d}"])
    return by_d, P_list, vga_by_d, class_by_d


def configure_from_summary_name(path):
    """Set N/epoch globals from e.g. langevin_mf_N30d_P_sweep_ep1000000_summary.npz."""
    global N_PER_D, N_ld, EPOCHS_SWEEP_LD
    name = os.path.basename(path)
    rest = name
    for prefix in ("langevin_mf_",):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
    rest = rest.replace("_summary.npz", "")
    stem, _, ep_part = rest.partition("_P_sweep_ep")
    if stem.endswith("d") and stem[1:-1].isdigit():
        N_PER_D = int(stem[1:-1])
    elif stem.startswith("N") and stem[1:].isdigit():
        N_PER_D = None
        N_ld = int(stem[1:])
    if ep_part.split("_")[0].isdigit():
        EPOCHS_SWEEP_LD = int(ep_part.split("_")[0])


def plot_from_summary(summary_path, show=False, recompute_vga=False, vga_workers=4):
    configure_from_summary_name(summary_path)
    by_d, P_list, vga_by_d, class_by_d = by_d_from_summary(summary_path)
    print(
        f"plot from {os.path.basename(summary_path)}  "
        f"stem={sweep_stem()}  epochs={EPOCHS_SWEEP_LD}  d={sorted(by_d)}  P={P_list}",
        flush=True,
    )
    if recompute_vga:
        print("recomputing VGA offdiag (fcn2_vga_erf.jl --advanced --offdiag, kappa=kappa_eff)", flush=True)
        for d in sorted(by_d):
            _hists, gpr_rows = by_d[d]
            vga_by_d[d] = vga_he3_from_gpr_rows(d, gpr_rows, workers=vga_workers)
            print(
                f"  d={d}  VGA offdiag He3="
                + " ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in vga_by_d[d]),
                flush=True,
            )
        save_summary(by_d, P_list, vga_by_d=vga_by_d, class_by_d=class_by_d)
    plot_all_d(by_d, P_list, show=show, vga_by_d=vga_by_d, class_by_d=class_by_d)
    plot_collapse(by_d, P_list, show=show, vga_by_d=vga_by_d, class_by_d=class_by_d)
    plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.1)
    plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.3)
    plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.5)
    print_tables(by_d, P_list, vga_by_d=vga_by_d, class_by_d=class_by_d)


def save_summary(by_d, P_list, vga_by_d=None, class_by_d=None):
    payload = {"P": np.asarray(P_list, dtype=float), "d": np.asarray(sorted(by_d), dtype=int)}
    for d in sorted(by_d):
        hists, gpr_rows = by_d[d]
        payload[f"he3_langevin_d{d}"] = np.asarray([hists[P]["he3_test"][-1] for P in P_list])
        payload[f"mse_langevin_d{d}"] = np.asarray([hists[P]["test_mse"][-1] for P in P_list])
        payload[f"he3_gpr_d{d}"] = np.asarray([r[1] for r in gpr_rows])
        payload[f"mse_gpr_d{d}"] = np.asarray([r[2] for r in gpr_rows])
        payload[f"sigma2_d{d}"] = np.asarray([r[3] for r in gpr_rows])
        if vga_by_d is not None and d in vga_by_d:
            payload[f"he3_vga_d{d}"] = np.asarray(vga_by_d[d])
        if class_by_d is not None and d in class_by_d:
            payload[f"he3_classical_d{d}"] = np.asarray(class_by_d[d])
    summary_path = os.path.join(OUT_DIR, sweep_basename("_summary.npz"))
    np.savez(summary_path, **payload)
    print(f"saved {summary_path}", flush=True)
    return summary_path


def print_tables(by_d, P_list, vga_by_d=None, class_by_d=None):
    for d in sorted(by_d):
        hists, gpr_rows = by_d[d]
        print(f"\nd={d}", flush=True)
        print(
            f"{'P':>6}  {'He3_LD':>8}  {'He3_GPR':>8}  {'He3_VGA':>8}  {'He3_cls':>8}  "
            f"{'MSE_LD':>8}  {'MSE_GPR':>8}",
            flush=True,
        )
        vga = vga_by_d.get(d) if vga_by_d is not None else None
        cls = class_by_d.get(d) if class_by_d is not None else None
        for i, row in enumerate(gpr_rows):
            P = int(P_list[i])
            h_g, m_g = float(row[1]), float(row[2])
            h_v = float(vga[i]) if vga is not None else float("nan")
            h_c = float(cls[i]) if cls is not None else float("nan")
            print(
                f"{P:6d}  {hists[P]['he3_test'][-1]:8.4f}  {h_g:8.4f}  {h_v:8.4f}  {h_c:8.4f}  "
                f"{hists[P]['test_mse'][-1]:8.4f}  {m_g:8.4f}",
                flush=True,
            )


def main(dims=None, P_list=None, show=False, plot_only=False, train_only=False):
    device = get_device()
    dims = list(dims if dims is not None else D_SWEEP_LD)
    P_list = list(P_list if P_list is not None else P_SWEEP_LD)
    if N_PER_D:
        n_desc = f"N={N_PER_D}*d (chi=N)"
    else:
        n_desc = f"N={N_ld} chi={chi_ld:g}"
    sa0_desc = f"sa0={float(sa0_ld):g}" if sa0_ld is not None else "sa0=classic(1/(Nχ))"
    print(
        f"{n_desc} s0={s0_ld:g} {sa0_desc}  T={T_ld} lr0={base_lr_ld}  eps={eps_ld}  "
        f"epochs={EPOCHS_SWEEP_LD}  d={dims}  P={P_list}  device={device}",
        flush=True,
    )
    trained = {}
    if not plot_only:
        for d in dims:
            hists, gpr_rows = {}, []
            for P in P_list:
                hist, _ = train_langevin_mf_one(d, P, device)
                hists[P] = hist
                if train_only:
                    print(
                        f"  d={d} P={P:5d}  Langevin He3={hist['he3_test'][-1]:.4f}  "
                        f"test MSE={hist['test_mse'][-1]:.4f}",
                        flush=True,
                    )
                    continue
                he3_g, mse_g, s2, lab = gpr_he3_for_P(d, P, device)
                gpr_rows.append((P, he3_g, mse_g, s2, lab))
                print(
                    f"  d={d} P={P:5d}  Langevin He3={hist['he3_test'][-1]:.4f}  "
                    f"test MSE={hist['test_mse'][-1]:.4f}  "
                    f"GPR He3={he3_g:.4f}  GPR MSE={mse_g:.4f}  σ²={s2:.4g} ({lab})",
                    flush=True,
                )
            trained[d] = (hists, gpr_rows)
    if train_only:
        return

    by_d = collect_from_disk(dims, P_list, device)
    by_d.update(trained)
    if by_d:
        vga_by_d = {}
        class_by_d = {}
        print("VGA cubic learnability (fcn2_vga_erf.jl --advanced --offdiag, kappa=kappa_eff)", flush=True)
        for d in sorted(by_d):
            vga_by_d[d] = vga_he3_curve(d, P_list, device)
            print(
                f"  d={d}  VGA He3="
                + " ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in vga_by_d[d]),
                flush=True,
            )
        print(
            "Classical Laplace cubic learnability (compute_fcn2_erf_cubic_eigs.jl, kappa=kappa_eff)",
            flush=True,
        )
        for d in sorted(by_d):
            class_by_d[d] = classical_he3_curve(d, P_list, device)
            print(
                f"  d={d}  classical He3="
                + " ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in class_by_d[d]),
                flush=True,
            )
        save_summary(by_d, P_list, vga_by_d=vga_by_d, class_by_d=class_by_d)
        plot_all_d(by_d, P_list, show=show, vga_by_d=vga_by_d, class_by_d=class_by_d)
        plot_collapse(by_d, P_list, show=show, vga_by_d=vga_by_d, class_by_d=class_by_d)
        plot_train_test_mse_per_d(by_d, P_list, show=show)
        plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.1)
        plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.3)
        plot_he3_threshold_scaling(by_d, P_list, show=show, thresh=0.5)
        print_tables(by_d, P_list, vga_by_d=vga_by_d, class_by_d=class_by_d)
    else:
        print("no completed (d, P) runs to plot", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Langevin MF P-sweep. Default priors are classic (χ=N, σ_A²=1/(Nχ)). "
            "Pass --sa0/--chi/--N/--s0/--T to match train_fcn2_erf_sigma_a steepwell hyps."
        )
    )
    parser.add_argument("--dims", type=str, default="5,10,20,30")
    parser.add_argument("--P", type=str, default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--T", type=float, default=None, help="Langevin temperature (T=2κ)")
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Hidden width (default 400, or N_PER_D·d with --n-per-d)",
    )
    parser.add_argument(
        "--chi",
        type=float,
        default=None,
        help="χ in T_eff=T/χ (default=N classic; steepwell sigma_a uses 1)",
    )
    parser.add_argument(
        "--s0",
        type=float,
        default=None,
        help="Read-in prior scale; σ_W0²=s0/d (default 1)",
    )
    parser.add_argument(
        "--sa0",
        type=float,
        default=None,
        help=(
            "Readout prior scale: σ_A²=sa0/(N·χ), wd_A=(N·χ/sa0)·T_eff. "
            "Omit for classic sa0=1 (σ_A²=1/(N·χ))."
        ),
    )
    parser.add_argument("--n-per-d", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument(
        "--from-summary",
        type=str,
        default="",
        help="Regenerate plots from a *_summary.npz (skips GPU).",
    )
    parser.add_argument(
        "--recompute-vga",
        action="store_true",
        help="With --from-summary, rerun offdiag VGA and overwrite he3_vga in the npz.",
    )
    args = parser.parse_args()
    if args.n_per_d:
        N_PER_D = int(args.n_per_d)
    if args.N is not None:
        N_ld = int(args.N)
        if args.chi is None and args.sa0 is None and args.n_per_d is None:
            # Keep classic coupling χ=N unless user opted into sigma_a / explicit chi.
            chi_ld = float(N_ld)
    if args.chi is not None:
        chi_ld = float(args.chi)
    if args.s0 is not None:
        s0_ld = float(args.s0)
    if args.sa0 is not None:
        sa0_ld = float(args.sa0)
    if args.epochs is not None:
        EPOCHS_SWEEP_LD = int(args.epochs)
    if args.eps is not None:
        eps_ld = float(args.eps)
    if args.T is not None:
        T_ld = float(args.T)
    if args.log_every is not None:
        LOG_SWEEP_LD = int(args.log_every)
    if args.from_summary:
        plot_from_summary(args.from_summary, recompute_vga=args.recompute_vga)
        raise SystemExit(0)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    P_list = (
        [int(x) for x in args.P.split(",") if x.strip()] if args.P else None
    )
    main(
        dims=dims,
        P_list=P_list,
        plot_only=args.plot_only,
        train_only=args.train_only,
    )
