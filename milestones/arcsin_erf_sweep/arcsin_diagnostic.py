import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math


# --- 1. SGLD with 1/fan-in weight regularization ---
class SGLDOptimizer(optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics with 1/fan_in weight regularization.
    
    Adds a prior term -λ * w / fan_in to the gradient of every nn.Linear layer.
    """
    def __init__(self, params, lr=1e-3, temperature=1.0, weight_decay_lambda=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if temperature < 0.0:
            raise ValueError(f"Invalid temperature: {temperature}")
        if weight_decay_lambda < 0.0:
            raise ValueError(f"Invalid weight_decay_lambda: {weight_decay_lambda}")

        defaults = dict(lr=lr, temperature=temperature, weight_decay_lambda=weight_decay_lambda)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            temp = group['temperature']
            lam = group['weight_decay_lambda']          # strength of 1/fan_in prior
            noise_std = math.sqrt(2 * lr * temp)        # standard SGLD noise scaling

            for p in group['params']:
                if p.grad is None:
                    continue

                # -------------------------------------------------
                # 1. Standard gradient (negative log-likelihood part)
                # -------------------------------------------------
                d_p = p.grad.clone()

                # -------------------------------------------------
                # 2. Add 1/fan_in regularization for Linear layers
                # -------------------------------------------------
                # p is a weight tensor of shape (out_features, in_features) for nn.Linear
                if p.ndim == 2 and isinstance(p, torch.nn.Parameter):
                    # Find which module this parameter belongs to
                    # Works because nn.Linear stores weight as (out, in)
                    fan_in = p.shape[1]  # in_features
                    if fan_in > 0:
                        # Prior p(w) ∝ exp(-λ ||w||² / (2 * fan_in))
                        # → gradient contribution = -λ * w / fan_in
                        d_p.add_(p, alpha=-lam * fan_in * temp * lr)

                # -------------------------------------------------
                # 3. SGD-style update (preconditioner is identity here)
                # -------------------------------------------------
                p.add_(d_p, alpha=-lr)

                # -------------------------------------------------
                # 4. Inject Langevin noise
                # -------------------------------------------------
                noise = torch.randn_like(p) * noise_std
                p.add_(noise)

        return loss


# --- 2. Hermite Polynomial and Model Definition (unchanged) ---
def hermite_3(x):
    return x**3 - 3*x


class ThreeLayerFCN_ERF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.erf(x)
        x = self.fc2(x)
        x = torch.erf(x)
        x = self.fc3(x)
        return x


class CustomSyntheticDataset(Dataset):
    def __init__(self, num_samples, input_dim, noise_std=0.1):
        self.input_dim = input_dim
        self.X = torch.randn(num_samples, input_dim)
        x0 = self.X[:, 0]
        pure_y = x0 + 0.03  * hermite_3(x0)
        noise = torch.randn(num_samples) * noise_std
        self.Y = (pure_y).view(-1, 1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --- 3. Configuration ---
D = 5
NUM_SAMPLES = 2 * 5**3
HIDDEN_SIZE = 512
OUTPUT_SIZE = 1
BATCH_SIZE = 100
LEARNING_RATE = 1e-5
TEMPERATURE = 0.2 
WEIGHT_DECAY_LAMBDA = 1.0        # <-- strength of the 1/fan_in prior (tune this!)
NUM_EPOCHS = 30000

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = CustomSyntheticDataset(NUM_SAMPLES, D)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ThreeLayerFCN_ERF(input_size=D, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
criterion = nn.MSELoss(reduction='sum')

# Use the new SGLD with 1/fan_in regularization
optimizer = SGLDOptimizer(
    model.parameters(),
    lr=LEARNING_RATE,
    temperature=TEMPERATURE,
    weight_decay_lambda=WEIGHT_DECAY_LAMBDA
)

print(f"Starting SGLD + 1/fan_in regularization training")
print(f"LR={LEARNING_RATE}, Temp={TEMPERATURE}, λ={WEIGHT_DECAY_LAMBDA}")


# --- 4. Training loop ---
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch_X, batch_Y in dataloader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)

    avg_loss = total_loss / NUM_SAMPLES
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

print("\nSGLD + 1/fan_in Training complete!")


# --- 5. Plotting function (with device support and saving) ---
def plot_prediction_vs_target(model, dataset, device, save_path='arcsin_diagnostic_plot.png'):
    model.eval()
    X_full = dataset.X.to(device)
    Y_true = dataset.Y
    x0_data = dataset.X[:, 0].numpy().squeeze()

    with torch.no_grad():
        Y_pred = model(X_full).cpu().numpy().squeeze()

    x0_min, x0_max = x0_data.min(), x0_data.max()
    x_test_range = np.linspace(x0_min, x0_max, 200)
    y_true_curve = x_test_range + 0.03 * (x_test_range**3 - 3*x_test_range)

    plt.figure(figsize=(10, 6))
    plt.scatter(x0_data, Y_true.numpy().squeeze(),
                label='Noisy Training Data', s=10, alpha=0.4, color='gray')
    plt.plot(x_test_range, y_true_curve,
             label='True Function $x_0 + H_3(x_0)$', linewidth=3, color='blue', linestyle='--')
    plt.scatter(x0_data, Y_pred, label='SGLD Prediction',
                s=25, alpha=0.8, color='red', marker='x')
    plt.title('SGLD with 1/fan_in Regularization – Prediction vs True Function')
    plt.xlabel('$x_0$')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


from pathlib import Path
output_dir = Path(__file__).parent
plot_prediction_vs_target(model, dataset, device, save_path=str(output_dir / 'arcsin_diagnostic_plot.png'))

# Save the trained model
model_path = output_dir / 'arcsin_diagnostic_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")