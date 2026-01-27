import numpy as np
import torch
import json
from pathlib import Path
from typing import List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
def load_fcn2_model(model_dir, device):
    """Load FCN2 model from directory."""
    config_path = Path(model_dir) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    d = int(cfg["d"])
    N = int(cfg["N"])
    P = int(cfg["P"])
    ens = int(cfg.get("ens", 1))
    activation = cfg.get("activation", "erf")
    from FCN2Network import FCN2NetworkActivationGeneric
    model_path = None
    for fname in ["model_final.pt", "model.pt", "checkpoint.pt"]:
        p = Path(model_dir) / fname
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    model = FCN2NetworkActivationGeneric(d, N, P, ens=ens, activation=activation, device=device).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(torch.float32)
    model.eval()
    return model, cfg

def compute_mse_over_ensemble(models: List[str], device="cpu", n_test=10000, n_seeds=10):
    """
    Compute dataset-averaged MSE for train and test loss over FCN2 ensemble.
    Returns dict with train_loss, test_loss, bias_train, bias_test.
    """
    train_losses = []
    test_losses = []
    bias_train = []
    bias_test = []
    for model_dir in models:
        model, cfg = load_fcn2_model(model_dir, device)
        d = int(cfg["d"])
        P = int(cfg["P"])
        ens = model.ens
        # For each dataset seed
        train_mse_seeds = []
        test_mse_seeds = []
        bias_train_seeds = []
        bias_test_seeds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Train set
            X_train = torch.randn(P, d, dtype=torch.float32, device=device)
            y_train = torch.randn(P, dtype=torch.float32, device=device)
            # Test set
            X_test = torch.randn(n_test, d, dtype=torch.float32, device=device)
            y_test = torch.randn(n_test, dtype=torch.float32, device=device)
            # Ensemble predictions
            with torch.no_grad():
                f_train = model(X_train)
                f_test = model(X_test)
            # If output is (P, ens), average over axis=1; if (ens, P), average over axis=0
            if f_train.shape[0] == P:
                fbar_train = f_train.mean(dim=1)
            else:
                fbar_train = f_train.mean(dim=0)
            if f_test.shape[0] == n_test:
                fbar_test = f_test.mean(dim=1)
            else:
                fbar_test = f_test.mean(dim=0)
            # Ensure shapes match
            assert fbar_train.shape == y_train.shape, f"fbar_train {fbar_train.shape} vs y_train {y_train.shape}"
            assert fbar_test.shape == y_test.shape, f"fbar_test {fbar_test.shape} vs y_test {y_test.shape}"
            # MSE
            train_mse = torch.mean((fbar_train - y_train) ** 2).item()
            test_mse = torch.mean((fbar_test - y_test) ** 2).item()
            train_mse_seeds.append(train_mse)
            test_mse_seeds.append(test_mse)
            # Bias: mean squared discrepancy, averaged over dataset, then over seeds
            bias_train_seeds.append(torch.sum((fbar_train - y_train) ** 2).item())
            bias_test_seeds.append(torch.sum((fbar_test - y_test) ** 2).item())
        train_losses.append(np.mean(train_mse_seeds))
        test_losses.append(np.mean(test_mse_seeds))
        bias_train.append(np.mean(bias_train_seeds))
        bias_test.append(np.mean(bias_test_seeds))
    return {
        "train_loss": float(np.mean(train_losses)),
        "test_loss": float(np.mean(test_losses)),
        "bias_train": float(np.mean(bias_train)),
        "bias_test": float(np.mean(bias_test)),
    }

def output_lI(model, X):
    """Compute eigenvalues of the model output Gram matrix using model.K_eig(X, X)."""
    with torch.no_grad():
        f = model.forward(X)  # (P, ens)
        P_actual = f.shape[0]
        K_per_ens = torch.einsum(
            'up,vp->uvp',
            f, f,

        ).mean(dim = 2)   # (P, P)
        return torch.linalg.eigvalsh(K_per_ens).cpu().sort(descending=True).values.numpy() / P_actual


def output_theory_eigs(kappa, P, chi, lJ):
    """
    Compute theory output eigenvalues as:
        eig = kappa / (P * chi) * lJ / (lJ + kappa / P)
    lJ: array-like of lJ values (e.g., [lJ1, lJ3])
    Returns: numpy array of output eigenvalues
    """
    lJ = np.array(lJ)
    eig = kappa / (P * chi) * lJ / (lJ + kappa / P) + ((lJ ) / (lJ + kappa / P))**2
    return eig

def compute_theoretical_loss(lH, d, P, kappa):
    """Theoretical EK Loss based on user formula."""
    denom = lH + (kappa / P)
    
    bias_theo = ((kappa / P) / denom) ** 2
    var_theo = (kappa / P) * (lH / denom)
    
    return bias_theo, var_theo

def streaming_feature_variances(model, d, P_total, batch_size=10000, device="cuda:1", seed=0):
    """
    Compute the variance (over ensemble index) of the model output projections onto d orthogonal linear target functions,
    using a streaming (batched) approach for large P_total.
    Args:
        model: FCN2 model
        d: input dimension
        P_total: total number of samples (e.g., 1e8)
        batch_size: number of samples per batch
        device: torch device
        seed: random seed for reproducibility
    Returns:
        proj_var: (d,) numpy array, variance over ensemble index for each linear direction
    """
    torch.manual_seed(seed)
    # Linear eigenfunctions: standard basis vectors in R^d
    # For each direction, project model output onto X[:,i]
    sum_proj = None  # (d, ens)
    sum_proj2 = None  # (d, ens)
    n_seen = 0
    for start in range(0, P_total, batch_size):
        end = min(start + batch_size, P_total)
        n_batch = end - start
        X = torch.randn(n_batch, d, dtype=torch.float32, device=device)
        # Model output: (n_batch, ens)
        with torch.no_grad():
            f = model(X)  # (n_batch, ens)
        # Projections: for each direction i, project f onto X[:,i]
        # proj[i, q] = sum_{u} f[u, q] * X[u, i]
        proj = torch.einsum('uq,ui->iq', f, X)  # (d, ens)
        if sum_proj is None:
            sum_proj = torch.zeros_like(proj)
            sum_proj2 = torch.zeros_like(proj)
        sum_proj += proj
        sum_proj2 += proj ** 2
        n_seen += n_batch
    # Compute mean and variance over all samples
    mean_proj = sum_proj / n_seen
    mean_proj2 = sum_proj2 / n_seen
    var_proj = mean_proj2 - mean_proj ** 2  # (d, ens)
    # Variance over ensemble index for each direction
    proj_var = var_proj.var(dim=1).cpu().numpy()  # (d,)
    return proj_var

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute dataset-averaged MSE for FCN2 ensemble.")
    parser.add_argument("--models", nargs="+", required=True, help="List of FCN2 model directories")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    result = compute_mse_over_ensemble(args.models, device=args.device, n_test=args.n_test, n_seeds=args.n_seeds)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    # --- Gram matrix eigenvalue comparison ---
    # Example: use first model, large dataset
    model_dir = args.models[0]
    model, cfg = load_fcn2_model(model_dir, args.device)
    d = int(cfg["d"])
    P =  int(cfg["P"])
    torch.manual_seed(1)
    X_big = torch.randn(P, d, dtype=torch.float32, device=args.device)
    Y1 = X_big[:, 0]
    Y3 = (X_big[:, 0]**3 - 3 * X_big[:, 0]) 

    y_big = Y1  + 0.03 * Y3 
    J_eigs = model.H_eig(X_big, X_big)[:100]
    
    # Compute network output via f = H @ [H + (kappa / P) I]^{-1} y
    kappa = 8.0

    chi = 10.
    H = model.H_Kernel(X_big, avg_ens=True)
    I = torch.eye(P, dtype=torch.float32, device=args.device)

    # Compute via torch
    H_reg_inv = torch.linalg.inv(H + (kappa) * I)
    f_out = model(X_big).mean(dim=1)# H @ H_reg_inv @ y_big # (P,)

    lJ1T = 0.00790392
    lJ3T = 1.82751e-5
    f_pred_ek = lJ1T / (lJ1T + kappa / P)*Y1 + (lJ3T / (lJ3T + kappa / P)) * Y3
    f_pred_ek_np = f_pred_ek.detach().cpu().numpy()
    f_out_np = f_out.detach().cpu().numpy()


    # Plot f_pred_ek against f_out and save
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(f_out_np, f_pred_ek_np, alpha=0.5)
    # Fit y=x line
    x = np.linspace(min(f_out_np.min(), f_pred_ek_np.min()), max(f_out_np.max(), f_pred_ek_np.max()), 100)
    plt.plot(x, x, color='red', linestyle='--', label='y=x')
    plt.legend()
    plt.xlabel("f_out (network output)")
    plt.ylabel("f_pred_ek (theory prediction)")
    plt.title("Network Output vs Theory Prediction")
    plt.grid()
    plt.savefig("fcn2_output_vs_theory.png")

    # Plot histogram of projections <f | Y1> and <f | Y3>
    with torch.no_grad():
        f_model = model(X_big)  # (P, ens)
    

    proj_Y1 = torch.einsum('up,u->p', f_model, Y1) / P # (ens,)
    proj_Y3 = torch.einsum('up,u->p', f_model, Y3) / P  # (ens,)
    print("Norm of y_big:", torch.norm(y_big).item())
    proj = torch.einsum('up,u->p', f_model, y_big).mean()  / P
    proj_pred = lJ1T / (lJ1T + kappa / P) + 0.03 * (lJ3T / (lJ3T + kappa / P))

    print("Projection onto Y1: ", proj_Y1.cpu().numpy())
    print("Theory of projection onto Y1: ", lJ1T / (lJ1T + kappa / P))
    # Print proj and proj_pred
    print("Projection onto target: ", proj.cpu().numpy())
    print("Predicted projection:", proj_pred)


    
    # Two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(proj_Y1.cpu().numpy(), bins=30, alpha=0.7, color='blue')
    ax[0].set_title("Projection onto Y1")
    ax[0].set_xlabel("Projection Value")
    ax[0].set_ylabel("Frequency")
    ax[0].grid()

    ax[1].hist(proj_Y3.cpu().numpy(), bins=30, alpha=0.7, color='green')
    ax[1].set_title("Projection onto Y3")
    ax[1].set_xlabel("Projection Value")
    ax[1].set_ylabel("Frequency")
    ax[1].grid()
    plt.legend()

    plt.title("Projections of Model Output onto Y1 and Y3")
    plt.grid()
    plt.savefig("fcn2_projections_Y1_Y3.png")

