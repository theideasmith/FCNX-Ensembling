import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from opt_einsum import contract
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# 2. Generate Data
input_size = 30
hidden_size = 400
output_size = 1
num_samples = 50
epochs = 2_000_000  # You might increase this with a decaying LR
chi = hidden_size
k = 1.0
t0 = 2 * k
t = t0 / chi  # Temperature for Langevin (currently unused in pure GD, but defined)

# --- Learning Rate Schedule Parameters ---
import math as mt
T = epochs * 0.8
lrA = 1e-8 / num_samples
lrB = (1.0 / 3) * lrA / num_samples 
beta = mt.log(lrA / lrB) / T
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Set seeds as constants
DATA_SEED = 613
MODEL_SEED = 26
LANGEVIN_SEED = 480

# Set the default dtype to float64
torch.set_default_dtype(torch.float64)

class FCN3NetworkEnsembleLinear(nn.Module):
    def __init__(self, d, n1, n2, ens=1, weight_initialization_variance=(1.0, 1.0, 1.0)):
        super().__init__()
        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.W0 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n1, d), weight_initialization_variance[0]**0.5)).to(device),
                               requires_grad=True)
        self.W1 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n2, n1), weight_initialization_variance[1]**0.5)).to(device),
                               requires_grad=True)
        self.A = nn.Parameter(torch.normal(mean=0.0,
                                          std=torch.full((ens, 1, n2), weight_initialization_variance[2]**0.5)).to(device),
                              requires_grad=True)

    def h1_activation(self, X):
        return contract(
            'ijk,ikl,unl->uij',
            self.W1, self.W0, X,
            backend='torch'
        )

    def h0_activation(self, X):
        return contract(
            'ikl,unl->uik',
            self.W0, X,
            backend='torch'
        )

    def forward(self, X):
        """
        Efficiently computes the outputs of a three layer network using opt_einsum
        f : P*d -> P*e*1
        C1_ui = W1_ijk*x_uk
        C2_uij = W2_ijk*C1_uik
        C3_ui = A_ij*C2_uij
        """
        A = self.A
        W1 = self.W1
        W0 = self.W0
        return contract(
            'eij,ejk,ekl,ul->uie',
            A, W1, W0, X,
            backend='torch'
        )

if __name__ == '__main__':
    # Create custom folder with P, D, N, epochs, lrA, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modeldesc = f"P_{num_samples}_D_{input_size}_N_{hidden_size}_epochs_{epochs}_lrA_{lrA:.2e}_time_{timestamp}"
    save_dir = os.path.join("/home/akiva/gpnettrain", modeldesc)
    os.makedirs(save_dir, exist_ok=True)
    runs_dir = os.path.join(save_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=runs_dir)

    # Seed for data
    torch.manual_seed(DATA_SEED)
    X = torch.randn((num_samples, input_size), dtype=torch.float64).to(device)
    Y = X[:, 0].unsqueeze(-1).to(device)

    # Seed for model
    torch.manual_seed(MODEL_SEED)
    ens = 100
    model = FCN3NetworkEnsembleLinear(input_size, hidden_size, hidden_size,
                                     ens=ens,
                                     weight_initialization_variance=(1/input_size, 1.0/hidden_size, 1.0/(hidden_size * chi))).to(device)

    losses = []
    weight_decay = np.array([input_size, hidden_size, hidden_size*chi], dtype=np.float64) * t

    criterion = nn.MSELoss(reduction='sum')

    epoch = 0
    with SummaryWriter(runs_dir) as writer:
        # 3. Optimization Loop with Learning Rate Schedule
        with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
            while epoch < epochs:
                if epoch < 2_000_000 * 0.8:
                    current_base_learning_rate = lrA
                else:
                    current_base_learning_rate = lrA / 3
                effective_learning_rate_for_update = current_base_learning_rate 
                noise_scale = (2 * effective_learning_rate_for_update * t)**0.5

                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, Y.unsqueeze(-1))
                losses.append(loss.item())

                # Backward pass
                model.zero_grad()
                loss.backward()

                # Manually update weights (Pure Gradient Descent)
                param_index = 0

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.data.add_(-effective_learning_rate_for_update * param.grad.data)
                            # Keep these commented out for pure GD
                            noise = torch.randn_like(param.data) * noise_scale 
                            param.data.add_(noise)
                            param.data.add_(-param.data * weight_decay[param_index] * effective_learning_rate_for_update)
                        param_index += 1 

                if (epoch + 1) % 500 == 0 or epoch == 0:
                    avg_loss = loss.item() / (ens * num_samples)
                    # Log to TensorBoard
                    writer.add_scalar('Loss/MSE', avg_loss, epoch)
                    writer.add_scalar('Learning_Rate/LR', current_base_learning_rate, epoch)
                    W0 = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1))
                    W1 = model.W1.permute(*torch.arange(model.W1.ndim - 1, -1, -1))

                    covW0W1 = contract('kje,ije,nme,kme->in', W1,W0,W0,W1, backend='torch') / (hidden_size * ens)

                    lH = covW0W1.diagonal().squeeze()
                    scalarsH = {}

                    for i in range(lH.shape[0]):
                        scalarsH[str(i)] = lH[i]

                    writer.add_scalars(f'Eigenvalues/lambda_H(W1)', scalarsH, epoch)
                    model_filename = f"model.pth"
                    torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
                    with open(os.path.join(save_dir, f"losses.txt"), "a") as f: 
                        f.write(f"{epoch},{loss.item()}\n")

                # Update tqdm progress bar
                pbar.update(1)
                pbar.set_postfix({"MSE Loss": f"{loss.item() / (ens * num_samples):.6f}", "Lr": f"{current_base_learning_rate:.2e}"})
                epoch += 1

    # Close TensorBoard writer
    writer.close()