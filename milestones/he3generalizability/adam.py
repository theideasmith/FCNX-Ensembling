import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os


# ============== Hyperparameters ==============
D: int = 20
HIDDEN_SIZE: int = 128
EPOCHS: int = 5000
LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 256
P_VALUES: List[int] = [20, 200, 500, 1000, 2500, 3000, 3500, 4000, 5000]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR: str = "weights"          # directory where model weights are saved
ANALYSIS_P: int = 5000               # P value used for He1 kernel projection analysis
# =============================================


# ============== Model ==============
class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.erf =torch.erf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.erf(self.fc1(x))
        x = self.fc2(x)
        return x


# ============== Data ==============
def generate_data(P: int, d: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((P, d)).astype(np.float32)
    x0 = x_np[:, 0]
    y_np = (x0 + 0.03 * (x0**3 - 3 * x0)).reshape(-1, 1).astype(np.float32)
    X = torch.from_numpy(x_np).to(DEVICE)
    Y = torch.from_numpy(y_np).to(DEVICE)
    return X, Y


# ============== Training ==============
def train(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    verbose: bool = True,
) -> List[float]:
    P = len(X)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(P)
        X_shuf, Y_shuf = X[perm], Y[perm]
        epoch_loss = 0.0

        for i in range(0, P, batch_size):
            batch_x = X_shuf[i:i + batch_size]
            batch_y = Y_shuf[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        epoch_loss /= P
        losses.append(epoch_loss)

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:4d} | MSE Loss: {epoch_loss:.2e}")

    return losses


# ============== Saving ==============
def save_model(model: nn.Module, P: int, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_P{P}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Weights saved → {path}")
    return path


# ============== Post-training kernel analysis ==============
def load_saved_model(P: int, d: int, hidden_size: int, save_dir: str) -> nn.Module:
    path = os.path.join(save_dir, f"model_P{P}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved model not found: {path}")

    model = ThreeLayerNet(d, hidden_size).to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_preactivation_kernel(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preact = model.fc1(X)  # shape: [P, H]
        hidden_size = preact.shape[1]
        kernel = (preact @ preact.T) / hidden_size
    return kernel


def compute_he1_projection_eigenvalue(
    P: int,
    d: int,
    hidden_size: int,
    save_dir: str,
    seed: int = 42,
) -> float:
    model = load_saved_model(P, d, hidden_size, save_dir)
    X, _ = generate_data(P, d, seed=seed)

    K = compute_preactivation_kernel(model, X)
    x0 = X[:, 0].reshape(P, 1)

    with torch.no_grad():
        # He1 direction sandwich: x[:,0]^T K x[:,0]
        he1_sandwich = (x0.T @ K @ x0).squeeze()
        he1_eigenvalue_div_P = he1_sandwich / P

    return float(he1_eigenvalue_div_P.item())


# ============== Run experiment across P values ==============
def run_experiment(
    p_values: List[int],
    d: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_dir: str = "weights",
) -> Dict[int, List[float]]:
    all_losses: Dict[int, List[float]] = {}

    for P in p_values:
        print(f"\n{'='*50}")
        print(f"Training with P = {P}")
        print(f"{'='*50}")

        X, Y = generate_data(P, d)
        model = ThreeLayerNet(d, hidden_size).to(DEVICE)
        losses = train(model, X, Y, epochs, batch_size, learning_rate)
        all_losses[P] = losses

        print(f"  Final MSE: {losses[-1]:.2e}")
        save_model(model, P, save_dir)

    return all_losses


# ============== Plotting ==============
def plot_losses(all_losses: Dict[int, List[float]], save_path: str = "loss_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for P, losses in all_losses.items():
        axes[0].plot(losses, label=f"P={P}")
        axes[1].semilogy(losses, label=f"P={P}")

    for ax, title in zip(axes, ["MSE Loss (linear)", "MSE Loss (log scale)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()


# ============== Entry point ==============
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    all_losses = run_experiment(
        p_values=P_VALUES,
        d=D,
        hidden_size=HIDDEN_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR,
    )

    he1_eval = compute_he1_projection_eigenvalue(
        P=ANALYSIS_P,
        d=D,
        hidden_size=HIDDEN_SIZE,
        save_dir=SAVE_DIR,
    )
    print(f"\nHe1 projected preactivation-kernel eigenvalue / P (P={ANALYSIS_P}): {he1_eval:.6e}")

    plot_losses(all_losses)