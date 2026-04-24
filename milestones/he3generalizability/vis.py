import torch
import numpy as np
import matplotlib.pyplot as plt
from adam import ThreeLayerNet, D, HIDDEN_SIZE, P_VALUES, SAVE_DIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============== Data ==============
def generate_eval_data(P: int, d: int) -> tuple:
    """Generate evaluation dataset with fixed seed 400, same size as training P."""
    torch.manual_seed(400)
    Xprimt = torch.randn((12000, d), device=DEVICE)
    x0primt = Xprimt[:, 0]
    x0primt_np = x0primt.detach().cpu().numpy()
    y_np = (x0primt + 0.03 * (x0primt**3 - 3 * x0primt)).reshape(-1, 1).cpu().numpy().astype(np.float32)
    return Xprimt, x0primt_np, y_np


# ============== Load model ==============
def load_model(P: int, save_dir: str, d: int, hidden_size: int) -> ThreeLayerNet:
    model = ThreeLayerNet(d, hidden_size).to(DEVICE)
    path = f"{save_dir}/model_P{P}.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ============== Compute normalized inner product ==============
def compute_nip(model: ThreeLayerNet, Xprimt: torch.Tensor, x0primt_np: np.ndarray, y_np: np.ndarray) -> float:
    """
    Projects predictions and targets onto the subspace orthogonal to x0primt,
    then computes their normalized inner product.

    h3_component = predictions - (predictions · x0primt / P) * x0primt
    h3           = y            - (y · x0primt / P) * x0primt
    NIP          = (h3_component · h3) / (||h3_component|| * ||h3||)
    """
    P = len(Xprimt)

    with torch.no_grad():
        predictions = model(Xprimt).squeeze().cpu().numpy()  # shape (P,)

    y_flat = y_np.flatten()  # shape (P,)

    # Project out the x0primt component
    h3_component = predictions - (predictions @ x0primt_np / P) * x0primt_np
    h3           = y_flat      - (x0primt_np @ y_flat      / P) * x0primt_np

    # Normalized inner product
    nip = (h3_component @ h3) / (np.linalg.norm(h3_component) * np.linalg.norm(h3))
    return float(nip)


# ============== Main ==============
def evaluate_all(p_values, d, hidden_size, save_dir):
    nips = []

    for P in p_values:
        print(f"Evaluating P={P}...")
        Xprimt, x0primt_np, y_np = generate_eval_data(P, d)
        model = load_model(P, save_dir, d, hidden_size)
        nip = compute_nip(model, Xprimt, x0primt_np, y_np)
        nips.append(nip)
        print(f"  NIP = {nip:.4f}")

    return nips


def plot_nip(p_values, nips, save_path="h3_nip.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, nips, marker='o', linewidth=2, markersize=7)
    ax.set_xlabel("P (training set size)", fontsize=13)
    ax.set_ylabel("$\\langle h_3^{\\text{component}}, h_3 \\rangle$", fontsize=13)
    ax.set_title("He3 Mode-Test Alignment vs dataset size (d=20, N=128, FCN2 Erf)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(1, color='green', linestyle='--', linewidth=0.8, label='perfect alignment')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    nips = evaluate_all(P_VALUES, D, HIDDEN_SIZE, SAVE_DIR)
    plot_nip(P_VALUES, nips)