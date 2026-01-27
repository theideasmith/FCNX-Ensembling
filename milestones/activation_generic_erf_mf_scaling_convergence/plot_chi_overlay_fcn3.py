python #!/usr/bin/env python3
"""
Plot chi overlay for FCN3 models using h1_preactivation and lH* eigenvalues.
Computes empirical and theory eigenvalues for a list of FCN3 runs.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from fcn3_theory_solver import fcn3_theory_solver

# --- Model loading ---
def load_fcn3_model(run_dir, device):
    config_path = Path(run_dir) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    d = int(cfg["d"])
    N = int(cfg["N"])
    P = int(cfg["P"])
    ens = int(cfg.get("ens", 1))
    activation = cfg.get("activation", "erf")
    from FCN3Network import FCN3NetworkActivationGeneric
    model_path = None
    for fname in ["model_final.pt", "model.pt", "checkpoint.pt"]:
        p = Path(run_dir) / fname
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {run_dir}")
    model = FCN3NetworkActivationGeneric(d, N, N, P, ens=ens, activation=activation, device=device).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.double()
    model.eval()
    return model, cfg

# --- Empirical eigenvalue computation ---
def compute_empirical_eigenvalues(model, d, P_total=200000, batch_size=100000, device="cpu", perp_dim=1):
    dtype = torch.float32
    ens = model.ens
    n1 = model.n1
    model.to(dtype)
    proj3_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj3_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    num_batches = P_total // batch_size
    remainder = P_total % batch_size
    with torch.no_grad():
        for i in tqdm(range(num_batches + (1 if remainder > 0 else 0)), desc="Batches"):
            bs = batch_size if i < num_batches else remainder
            if bs == 0:
                break
            X_batch = torch.randn(bs, d, dtype=dtype, device=device)
            x0 = X_batch[:, 0]
            x_perp = X_batch[:, perp_dim]
            phi3_target = (x0**3 - 3.0 * x0)
            phi3_perp = (x_perp**3 - 3.0 * x_perp)
            phi1_target = x0
            phi1_perp = x_perp
            a1 = model.h1_preactivation(X_batch)
            proj3_target_sum += torch.einsum('pqn,p->qn', a1, phi3_target) / P_total
            proj3_perp_sum += torch.einsum('pqn,p->qn', a1, phi3_perp) / P_total
            proj1_target_sum += torch.einsum('pqn,p->qn', a1, phi1_target) / P_total
            proj1_perp_sum += torch.einsum('pqn,p->qn', a1, phi1_perp) / P_total
            torch.cuda.empty_cache()
            del X_batch, x0, x_perp, phi3_target, phi3_perp, phi1_target, phi1_perp, a1
    proj3_target = proj3_target_sum
    proj3_perp = proj3_perp_sum
    proj1_target = proj1_target_sum
    proj1_perp = proj1_perp_sum
    eig_target1 = proj1_target.var().cpu().numpy().flatten()
    eig_perp1 = proj1_perp.var().cpu().numpy().flatten()
    eig_target3 = proj3_target.var().cpu().numpy().flatten()
    eig_perp3 = proj3_perp.var().cpu().numpy().flatten()
    d_minus_1 = d - 1
    eigenvalues = np.concatenate([
        eig_target1, np.tile(eig_perp1, d_minus_1),
        eig_target3, np.tile(eig_perp3, d**3 - 1)
    ]).flatten()
    return {
        "h3_target": eig_target3.tolist(),
        "h3_perp": eig_perp3.tolist(),
        "h1_target": eig_target1.tolist(),
        "h1_perp": eig_perp1.tolist(),
        "all": eigenvalues.tolist(),
    }

# --- Main plotting routine ---
def main():
    parser = argparse.ArgumentParser(description="Plot chi overlay for FCN3 runs (h1_preactivation, lH* eigenvalues)")
    parser.add_argument("--runs", nargs="*", default=[], help="Run directories to process")
    parser.add_argument("--P_total", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--perp_dim", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    if not args.runs:
        print("No runs specified. Use --runs to provide run directories.")
        return
    labels = []
    h3_target_vars = []
    h3_perp_vars = []
    h1_target_vars = []
    h1_perp_vars = []
    theory_h3_target = None
    theory_h3_perp = None
    theory_h1_target = None
    theory_h1_perp = None
    for run in args.runs:
        run_dir = Path(run)
        model, cfg = load_fcn3_model(run_dir, device)
        d = int(cfg["d"])
        N = int(cfg["N"])
        P = int(cfg["P"])
        chi = float(cfg.get("chi", 1.0))
        kappa = float(cfg.get("kappa", 1.0 / chi))
        epsilon = float(cfg.get("eps", 0.03))
        # Empirical eigenvalues
        emp = compute_empirical_eigenvalues(model, d, P_total=args.P_total, batch_size=args.batch_size, device=device, perp_dim=args.perp_dim)
        h3_target_vars.append(np.mean(emp["h3_target"]))
        h3_perp_vars.append(np.mean(emp["h3_perp"]))
        h1_target_vars.append(np.mean(emp["h1_target"]))
        h1_perp_vars.append(np.mean(emp["h1_perp"]))
        labels.append(run_dir.name)
        # Theory (lH* eigenvalues)
        theory = fcn3_theory_solver(
            d=d,
            n1=N,
            P=P,
            chi=chi,
            kappa=kappa,
            epsilon=epsilon,
            quiet=True
        )
        target_block = theory.get("target", {})
        perp_block = theory.get("perpendicular", {})
        theory_h3_target = target_block.get("lH3T", theory_h3_target)
        theory_h1_target = target_block.get("lH1T", theory_h1_target)
        theory_h3_perp = perp_block.get("lH3P", theory_h3_perp)
        theory_h1_perp = perp_block.get("lH1P", theory_h1_perp)
    # Plot
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes[0, 0].bar(x, h3_target_vars, color="#4C78A8")
    if theory_h3_target is not None:
        axes[0, 0].axhline(theory_h3_target, color="#E45756", linestyle="--", label="Theory lH3T")
    axes[0, 0].set_title("He3 Projections (target)")
    axes[0, 0].set_ylabel("Variance")
    axes[0, 0].legend()
    axes[0, 1].bar(x, h3_perp_vars, color="#4C78A8")
    if theory_h3_perp is not None:
        axes[0, 1].axhline(theory_h3_perp, color="#E45756", linestyle="--", label="Theory lH3P")
    axes[0, 1].set_title("He3 Projections (perp)")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].legend()
    axes[1, 0].bar(x, h1_target_vars, color="#72B7B2")
    if theory_h1_target is not None:
        axes[1, 0].axhline(theory_h1_target, color="#E45756", linestyle="--", label="Theory lH1T")
    axes[1, 0].set_title("He1 Projections (target)")
    axes[1, 0].set_ylabel("Variance")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend()
    axes[1, 1].bar(x, h1_perp_vars, color="#72B7B2")
    if theory_h1_perp is not None:
        axes[1, 1].axhline(theory_h1_perp, color="#E45756", linestyle="--", label="Theory lH1P")
    axes[1, 1].set_title("He1 Projections (perp)")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].legend()
    fig.tight_layout()
    out_plot = args.out or (Path(args.runs[0]).parent / "h_projections_summary_fcn3.png")
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot: {out_plot}")
    # Save a JSON config with the image path and models used
    summary_json = Path(out_plot).with_suffix(".json")
    summary = {"image": str(out_plot.name), "models": args.runs}
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary config: {summary_json}")

if __name__ == "__main__":
    main()
