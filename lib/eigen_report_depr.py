import os
import json
import subprocess
import tempfile
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from FCN2Network import FCN2NetworkActivationGeneric
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from julia_theory import compute_fcn2_erf_cubic_eigs, compute_fcn2_erf_eigs
from spectrum_svd import randomized_svd_spectrum, qb_projection_eigenvalues
from compute_h3_projections import compute_h3_projections_streaming
from tqdm import tqdm
import re
import hashlib
import datetime

def extract_full_spectrum(res, method):
    """
    Returns sorted eigenvalues (descending) as 1D numpy array, or None.
    """
    if method == "h_proj":
        return None  # no raw spectrum here
    vals = res.get(method)
    if vals is None:
        return None
    arr = np.array(vals)
    if arr.ndim != 1 or arr.size == 0:
        return None
    return np.sort(arr)[::-1]

def load_model_from_run(run_dir: Path, device):
    d = None
    N = None
    P = None
    chi = 1.0
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            d = int(cfg.get("d")) if cfg.get("d") is not None else None
            P = int(cfg.get("P", 0)) or None
            N = int(cfg.get("N")) if cfg.get("N") is not None else None
            chi = float(cfg.get("chi", 1.0))
        except Exception:
            pass

    if d is None or N is None:
        m = re.match(
            r"^d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi_(?P<chi>[-+]?\d*\.?\d+)(?:_lr_[^_]+)?(?:_T_[-+]?\d*\.?\d+)?(?:_seed_\d+)?(?:_eps_[-+]?\d*\.?\d+)?$",
            run_dir.name,
        )
        if m:
            try:
                d = int(m.group("d"))
                P = int(m.group("P")) if m.group("P") else None
                N = int(m.group("N"))
                chi = float(m.group("chi"))
            except Exception:
                pass

    state_paths = [
        run_dir / "model_final.pt",
        run_dir / "model.pt",
        run_dir / "checkpoint.pt",
    ]
    sd = None
    for sp in state_paths:
        if sp.exists():
            sd_raw = torch.load(sp, map_location=device)
            sd = sd_raw["model_state_dict"] if isinstance(sd_raw, dict) and "model_state_dict" in sd_raw else sd_raw
            break
    if sd is None or not isinstance(sd, dict):
        raise FileNotFoundError(f"No loadable model state found in {run_dir}")
    if "W0" not in sd:
        raise KeyError(f"State dict missing 'W0' in {run_dir}")

    w_shape = sd["W0"].shape
    ens = int(w_shape[0])
    n1 = int(w_shape[1])
    d_from_weights = int(w_shape[2])
    if d is None:
        d = d_from_weights
    if N is None:
        N = n1

    sigma_W0_sq = 1.0 / d
    sigma_A_sq = 1.0 / (N * chi)
    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=None,
        ens=ens,
        activation="erf",
        weight_initialization_variance=(sigma_W0_sq, sigma_A_sq),
        device=device,
    ).to(device)
    model.load_state_dict(sd)
    model = model.double()
    model.eval()
    return model

def parse_run_params(run_dir: Path):
    d = None
    P = None
    N = None
    chi = None
    T = None
    epsilon = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            d = int(cfg.get("d")) if cfg.get("d") is not None else None
            P = int(cfg.get("P")) if cfg.get("P") is not None else None
            N = int(cfg.get("N")) if cfg.get("N") is not None else None
            chi = float(cfg.get("chi")) if cfg.get("chi") is not None else None
            T = float(cfg.get("temperature")) if cfg.get("temperature") is not None else None
            epsilon = float(cfg.get("eps")) if cfg.get("eps") is not None else None
        except Exception:
            pass

    if d is None or N is None or P is None or chi is None:
        m = re.match(
            r"^d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi_(?P<chi>[-+]?\d*\.?\d+)(?:_lr_[^_]+)?(?:_T_(?P<T>[-+]?\d*\.?\d+))?(?:_seed_\d+)?(?:_eps_(?P<eps>[-+]?\d*\.?\d+))?",
            run_dir.name,
        )
        if m:
            d = int(m.group("d")) if d is None else d
            P = int(m.group("P")) if P is None else P
            N = int(m.group("N")) if N is None else N
            chi = float(m.group("chi")) if chi is None else chi
            if T is None and m.group("T") is not None:
                T = float(m.group("T"))
            if epsilon is None and m.group("eps") is not None:
                epsilon = float(m.group("eps"))

    kappa = T / 2.0 if T is not None else (1.0 / chi if chi else None)
    return d, P, N, chi, kappa, epsilon

def get_model_H_eig(model: FCN2NetworkActivationGeneric, d: int, device: torch.device):
    if hasattr(model, 'H_eig'):
        X = torch.randn(10000, d, device=device)
        Y = X
        eigvals_lin = model.H_eig(X, Y)
        Y3 = (X ** 3 - 3.0 * X)
        eigvals_he3 = model.H_eig(X, Y3)
        if isinstance(eigvals_lin, torch.Tensor):
            eigvals_lin = eigvals_lin.detach().sort(descending=True).values.cpu().numpy()
        if isinstance(eigvals_he3, torch.Tensor):
            eigvals_he3 = eigvals_he3.detach().sort(descending=True).values.cpu().numpy()
        eigvals = np.concatenate([eigvals_lin, eigvals_he3])
        return eigvals
    return None

def eigen_report(train_runs: List[str], out_dir: str = "eigen_report"):

    # Create unique report directory
    model_dirs_sorted = sorted([str(Path(run).resolve()) for run in train_runs])
    model_dirs_str = json.dumps(model_dirs_sorted, sort_keys=True)
    model_dirs_hash = hashlib.md5(model_dirs_str.encode('utf-8')).hexdigest()[:10]

    base_reports_dir = Path(out_dir).parent if out_dir else Path('.')
    report_dir = base_reports_dir / f"eigen_report_{model_dirs_hash}"

    report_json = report_dir / "model_dirs.json"
    if report_json.exists():
        with open(report_json, 'r') as f:
            existing = json.load(f)
        if sorted(existing.get("model_dirs", [])) == model_dirs_sorted:
            print(f"Reusing existing report directory: {report_dir}")
            out_dir = str(report_dir)
        else:
            out_dir = str(base_reports_dir / f"eigen_report_{model_dirs_hash}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(report_dir, exist_ok=True)
        with open(report_json, 'w') as f:
            json.dump({"model_dirs": model_dirs_sorted}, f, indent=2)
        out_dir = str(report_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = {}
    agg = {m: {"lH1_target": [], "lH1_perp": [], "lH3_target": [], "lH3_perp": []}
           for m in ["h_proj", "svd_eigs", "qb_proj_eigs", "H_eig"]}

    for run in tqdm(train_runs, desc="Processing runs"):
        run_dir = Path(run)
        print(f"\n{run_dir.name}")
        model = load_model_from_run(run_dir, device)
        d, P, N, chi, kappa, eps = parse_run_params(run_dir)

        print("  → h3/h1 projections")
        stats = compute_h3_projections_streaming(model, d, P_total=1000, batch_size=1000, device=device)

        h1_target = stats.get("h3", {}).get("target", {}).get("second_moment")
        h1_perp   = stats.get("h3", {}).get("perp",   {}).get("second_moment")

        print("  → randomized SVD spectrum")
        svd_eigs = randomized_svd_spectrum(model, d, device, N=700, k=500, p=10, chunk_size=4096)

        print("  → QB Hermite projection eigenvalues")
        qb_proj = qb_projection_eigenvalues(model, d, device, N=700, k=500, p=25, chunk_size=4096)
        qb_eigs = qb_proj.get("all_eigenvalues") if qb_proj else None

        print("  → model.H_eig (if available)")
        H_eig = get_model_H_eig(model, d, device)

        print("  → theory (bare kappa)")
        print("Epsilon: ", eps)
        theory_bare = compute_fcn2_erf_cubic_eigs(d, N, P, chi, kappa, epsilon=eps or 0.0)


            # --- Kappa correction: run self-consistent solver ---
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        # Use arcsin kernel eigenvalues for kappa bare
        eigenvalues = svd_eigs.tolist()
        with open(eig_json, "w") as f2:
            json.dump({"eigenvalues": eigenvalues, "kappa_bare": kappa}, f2)
        sc_cmd = [
            "julia", '/home/akiva/FCNX-Ensembling/julia_lib/self_consistent_kappa_solver.jl',
            eig_json, str(P)
        ]
        try:
            sc_out = subprocess.check_output(sc_cmd, text=True)
            match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
            if match:
                kappa_eff = float(match.group(1))
                print(f"  Found kappa_eff: {kappa_eff:.6f} for {run_dir}")
            else:
                print(f"Warning: could not parse kappa_eff from self-consistent solver output for {run_dir}.")
                kappa_eff = kappa
        except Exception as e:
            print(f"Self-consistent solver failed for {run_dir}: {e}")
            kappa_eff = kappa
        finally:
            try:
                os.remove(eig_json)
            except Exception:
                pass
        print("  → theory (effective kappa)")
        theory_eff = compute_fcn2_erf_cubic_eigs(d, N, P, chi, kappa_eff, epsilon=eps or 0.0)

        # Aggregate summary values
        if svd_eigs is not None:
            arr = np.array(svd_eigs)
            agg["svd_eigs"]["lH1_target"].append(arr[0])
            agg["svd_eigs"]["lH1_perp"].append(arr[1:d])
            agg["svd_eigs"]["lH3_target"].append(arr[d])
            agg["svd_eigs"]["lH3_perp"].append(arr[d+1:])

        if qb_eigs is not None:
            arr = np.array(qb_eigs)
            agg["qb_proj_eigs"]["lH1_target"].append(arr[0])
            agg["qb_proj_eigs"]["lH1_perp"].append(arr[1:d])
            agg["qb_proj_eigs"]["lH3_target"].append(arr[d])
            agg["qb_proj_eigs"]["lH3_perp"].append(arr[d+1:])

        if H_eig is not None:
            arr = np.array(H_eig)
            agg["H_eig"]["lH1_target"].append(arr[0])
            agg["H_eig"]["lH1_perp"].append(arr[1:d])
            agg["H_eig"]["lH3_target"].append(arr[d])
            agg["H_eig"]["lH3_perp"].append(arr[d+1:])

        agg["h_proj"]["lH1_target"].append(h1_target)
        agg["h_proj"]["lH1_perp"].append(h1_perp)

        all_results[run] = {
            "h_proj": stats,
            "svd_eigs": svd_eigs.tolist() if svd_eigs is not None else None,
            "qb_proj_eigs": qb_eigs.tolist() if qb_eigs is not None else None,
            "H_eig": H_eig.tolist() if H_eig is not None else None,
            "theory_bare": theory_bare,
            "theory_eff": theory_eff
        }

        print("Results summary:")
        print(f"  h_proj lH1_target: {h1_target}")
        print(f"  h_proj lH1_perp:   {h1_perp}")
        if svd_eigs is not None:
            print(f"  svd_eigs lH1_target: {svd_eigs[0]}")
            print(f"  svd_eigs lH1_perp:   {svd_eigs[1:d].mean()}")
        if qb_eigs is not None:
            print(f"  qb_proj_eigs lH1_target: {qb_eigs[0]}")
            print(f"  qb_proj_eigs lH1_perp:   {qb_eigs[1:d].mean()}")
        if H_eig is not None:
            print(f"  H_eig lH1_target: {H_eig[0]}")
            print(f"  H_eig lH1_perp:   {H_eig[1:d].mean()}")
        if theory_bare:
            print(f"  theory_bare lJ1_target: {theory_bare['target'].get('lJ1T')}")
            print(f"  theory_bare lJ1_perp:   {theory_bare['perpendicular'].get('lJ1P')}")
            print(f"  theory_bare lJ3_target: {theory_bare['target'].get('lJ3T')}")
            print(f"  theory_bare lJ3_perp:   {theory_bare['perpendicular'].get('lJ3P')}")
        if theory_eff:
            print(f"  theory_eff lJ1_target: {theory_eff['target'].get('lJ1T')}")
            print(f"  theory_eff lJ1_perp:   {theory_eff['perpendicular'].get('lJ1P')}")
            print(f"  theory_eff lJ3_target: {theory_eff['target'].get('lJ3T')}")
            print(f"  theory_eff lJ3_perp:   {theory_eff['perpendicular'].get('lJ3P')}")
        

        with open(os.path.join(out_dir, f"eigen_report_{run_dir.name}.json"), "w") as f:
            json.dump(all_results[run], f, indent=2)

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ───────────────────────────────────────────────
    # Aggregate full spectrum statistics (across runs)
    # ───────────────────────────────────────────────

    spectra_by_method = {"svd_eigs": [], "qb_proj_eigs": [], "H_eig": []}

    for run in train_runs:
        res = all_results[run]
        d_run, _, _, _, _, _ = parse_run_params(Path(run))
        if d_run is None:
            continue
        for method in spectra_by_method:
            spectrum = extract_full_spectrum(res, method)
            if spectrum is not None:
                spectra_by_method[method].append(spectrum)

    agg_spectrum_stats = {}
    for method, run_spectra in spectra_by_method.items():
        if not run_spectra:
            continue
        stack = np.stack(run_spectra)  # (n_runs, 2*d)
        mean_spec = np.mean(stack, axis=0)
        std_spec  = np.std(stack, axis=0)

        agg_spectrum_stats[method] = {
            "mean": mean_spec,
            "std":  std_spec,
            "n_runs": len(run_spectra),
            "indices": np.arange(len(mean_spec)),
            "d": d_run   # assume same d across runs
        }

    method_names = {
        "svd_eigs": "Randomized SVD",
        "qb_proj_eigs": "QB Hermite Proj",
        "H_eig": "Model .H_eig"
    }

    # ───────────────────────────────────────────────
    # Per-method full spectrum bar plot (aggregate across runs)
    # ───────────────────────────────────────────────

    ref_res = all_results[train_runs[0]] if train_runs else {}

    for method in ["svd_eigs", "qb_proj_eigs", "H_eig"]:
        print(f"Plotting full spectrum for method: {method}")
        if method not in agg_spectrum_stats:
            continue
        stats = agg_spectrum_stats[method]
        d = stats["d"]

        fig, ax = plt.subplots(figsize=(14, 7))

        x = stats["indices"]
        ax.bar(x, stats["mean"], yerr=stats["std"], capsize=3,
               color='C0', alpha=0.78, error_kw=dict(elinewidth=0.9, capthick=0.9))

        # ── BOTH theories ───────────────────────────────────────
        for theory_dict, color, label_prefix in [
            (ref_res.get("theory_bare"), 'k', "bare κ"),
            (ref_res.get("theory_eff"),  'r', "eff κ")
        ]:
            print("theory_dict:", theory_dict)
            if not theory_dict:
                continue
            target = theory_dict.get("target", {})
            perp   = theory_dict.get("perpendicular", {})

            if "lJ1T" in target:
                ax.axhline(target["lJ1T"], color=color, ls="--", lw=1.5, alpha=0.92,
                           label=f"{label_prefix} H1 target")
            if "lJ1P" in perp:
                ax.axhline(perp["lJ1P"], color=color, ls=":", lw=1.2, alpha=0.75,
                           label=f"{label_prefix} H1 perp")

            if "lJ3T" in target:
                ax.axhline(target["lJ3T"], color=color, ls="--", lw=1.5, alpha=0.92,
                           label=f"{label_prefix} H3 target")
            if "lJ3P" in perp:
                ax.axhline(perp["lJ3P"], color=color, ls=":", lw=1.2, alpha=0.75,
                           label=f"{label_prefix} H3 perp")

        ax.set_yscale("log")
        ax.set_xlabel("Eigenvalue index (sorted descending)")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Full spectrum – {method_names[method]}   ({stats['n_runs']} runs)\n"
                     "mean ± std across runs – no degeneracy averaging")
        ax.axvline(d - 0.5, color="0.65", ls="--", lw=0.9, alpha=0.6)
        ax.legend(fontsize=9.5, ncol=2, loc="upper right", framealpha=0.92)

        plt.tight_layout()
        plot_name = os.path.join(out_dir, f"full_spectrum_bar_{method}_across_runs.png")
        print("Saving plot to:", plot_name)
        plt.savefig(plot_name, dpi=150)
        plt.close()

    print(f"\nAll plots and JSON saved to:\n  {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate eigenvalue report for FCN2 runs.")
    parser.add_argument('run_dirs', nargs='+', help='List of run directories')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    run_dirs = [d for d in args.run_dirs if not d.startswith('--')]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = f"eigen_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eigen_report(run_dirs, out_dir=out_dir)