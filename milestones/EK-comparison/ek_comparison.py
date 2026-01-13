"""
EK-Comparison: Train networks and compare with EK predictions.

Configuration:
- chi = 1.0 (standard scaling)
- N = 512 (hidden width)
- d = 2, 3, 4, ..., 10 (input dimensions)
- P = round(1.5 * d) (number of samples)
- 10K epochs per network
- 50 datasets per d value
- 5 networks per dataset (ensemble=5)
- Seeds: Ramanujan partition function values

Output:
- Trained networks stored in data/networks/
- EK predictions and comparisons in data/results/
- Plots saved to plots/
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add library path
sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')

import standard_hyperparams as hp
from activations import linear_activation
from ramanujan_seeds import get_ramanujan_partition_seeds
from GPKit import gpr_dot_product_explicit


@dataclass
class ExperimentConfig:
    """Configuration for EK comparison experiment."""
    chi: float = 1.0
    N: int = 256  # hidden width
    P_factor: float = 1.5
    epochs: int = 50_000
    num_datasets: int = 100
    ensemble_size: int = 10
    kappa_ek: float = 1.0
    learning_rate: float = 1e-5
    temperature: float = 2.0  # Temperature for Langevin dynamics
    d_range: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.d_range is None:
            self.d_range = list(range(2, 12,3))
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'chi': self.chi,
            'N': self.N,
            'P_factor': self.P_factor,
            'epochs': self.epochs,
            'num_datasets': self.num_datasets,
            'ensemble_size': self.ensemble_size,
            'kappa_ek': self.kappa_ek,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            'd_range': self.d_range,
        }
        return d


class FCN3NetworkLinear(nn.Module):
    """Simple 3-layer FCN with linear activations."""
    
    def __init__(self, d: int, n1: int, n2: int, device='cpu'):
        super().__init__()
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.device = device
        
        # Layer 1: d -> n1
        self.fc1 = nn.Linear(d, n1, device=device, bias = False)
        # Layer 2: n1 -> n2
        self.fc2 = nn.Linear(n1, n2, device=device, bias = False)
        # Output layer: n2 -> 1
        self.fc3 = nn.Linear(n2, 1, device=device, bias = False)
        
        # Initialize weights
        with torch.no_grad():
            self.fc1.weight.normal_(0, np.sqrt(1.0 / d))
            self.fc2.weight.normal_(0, np.sqrt(1.0 / n1))
            self.fc3.weight.normal_(0, np.sqrt(1.0 / n2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with linear activations."""
        x = self.fc1(x)  # Linear activation (identity)
        x = self.fc2(x)  # Linear activation (identity)
        x = self.fc3(x)  # Output layer
        return x
    
    def to_dict(self) -> Dict:
        """Serialize model to dictionary."""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.detach().cpu().numpy()
        return state
    
    @classmethod
    def from_dict(cls, state: Dict, d: int, n1: int, n2: int, device='cpu'):
        """Load model from dictionary."""
        model = cls(d, n1, n2, device)
        for name, value in state.items():
            param = getattr(model, name.split('.')[0])
            for attr in name.split('.')[1:]:
                if attr.isdigit():
                    param = param[int(attr)]
                else:
                    param = getattr(param, attr)
            param.data = torch.tensor(value, dtype=torch.float32, device=device)
        return model


def generate_synthetic_data(d: int, P: int, seed: Optional[int] = None, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data with train/test split (80/20)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    X = torch.randn(P, d, device=device)
    # True function: first component as target
    y = X[:, 0].unsqueeze(-1)
    
    # 80/20 train/test split
    split_idx = int(0.8 * P)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def train_network(
    model: FCN3NetworkLinear,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    learning_rate: float,
    device='cpu',
    temperature: float = 1.0,
    pbar: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """Train a single network using Langevin dynamics with sum reduction loss and weight regularization.
    
    Langevin dynamics with L2 weight decay:
    θ_{t+1} = θ_t - η∇L(θ_t) - ηλθ_t + √(2ηT)ξ_t
    
    where:
    - η is learning rate
    - L is the sum of squared errors loss
    - λ is weight decay coefficient
    - T is temperature
    - ξ_t is standard normal noise
    """
    model.to(device)
    
    # Langevin noise standard deviation: sigma = sqrt(2 * eta * temperature)
    noise_std = torch.sqrt(torch.tensor(2.0 * learning_rate * temperature, device=device))
    
    history = {'loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_train)
        
        # Sum reduction loss (not MSE mean)
        loss = torch.sum((y_pred - y_train) ** 2)
        
        total_loss = loss 
        
        # Zero gradients
        if epoch > 0:
            for param in model.parameters():
                param.grad = None
        
        # Backward pass on total loss (MSE + L2 reg)
        total_loss.backward()
        
        # Langevin dynamics update for each parameter with weight decay
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Gradient term: -eta * grad(L) - eta * lambda * theta
                    grad_term = -learning_rate * param.grad
                    if len(param.shape) == 1:
                        weight_decay_term = -learning_rate * temperature * param
                    else:
                        weight_decay_term = -learning_rate * temperature * param * param.shape[1]
                    
                    # Langevin noise term: sqrt(2*eta*T) * N(0,1)
                    noise = noise_std * torch.randn_like(param)
                    
                    # Update: theta = theta + grad_term + weight_decay_term + noise
                    param.add_(grad_term + weight_decay_term + noise)
        
        if pbar is not None:
            pbar.update(1)
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            history['loss'].append(loss.item())
            history['epoch'].append(epoch + 1)
    
    return history


def compute_empirical_kernel(
    model: FCN3NetworkLinear,
    X: torch.Tensor,
    device='cpu'
) -> torch.Tensor:
    """Compute empirical NTK: K_ij = ∑_p (∂f/∂θ_p)|_i * (∂f/∂θ_p)|_j"""
    # This is a simplified version
    # For full NTK, we'd need to compute jacobians
    model.eval()
    
    # Get Jacobian (parameters x samples)
    jacobians = []
    for i in range(X.shape[0]):
        x_i = X[i:i+1]
        x_i.requires_grad_(True)
        y = model(x_i)
        
        for param in model.parameters():
            if param.requires_grad:
                if jacobians and len(jacobians[-1]) != len(list(model.parameters())):
                    jacobians.append([])
                
                grad_outputs = torch.ones_like(y)
                grads = torch.autograd.grad(y, param, grad_outputs=grad_outputs, 
                                           create_graph=True, retain_graph=True)
                if grads[0] is not None:
                    if not jacobians:
                        jacobians = [[] for _ in range(len(list(model.parameters())))]
                    jacobians[len(jacobians)-1].append(grads[0].detach().flatten())
    
    # Simplified: return identity for now (placeholder)
    return torch.eye(X.shape[0], device=device)


class NetworkTrainer:
    """Trainer for managing network ensembles."""
    
    def __init__(self, config: ExperimentConfig, device='cpu'):
        self.config = config
        self.device = device
        self.seeds = get_ramanujan_partition_seeds(10)
        
        # TensorBoard setup
        self.tb_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/runs')
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.tb_dir / f'ek_comparison_{int(time.time())}'))
        
        # Tracking metrics per d
        self.loss_tracker = {}  # d -> list of losses
        self.eigs_leading = {}  # d -> list of leading eigenvalues
        self.eigs_perp = {}     # d -> list of perpendicular eigenvalues
        self.slope_tracker = {} # d -> list of slopes (model vs GPR)
    
    def compute_eigenvalues_and_slope(
        self, 
        model: FCN3NetworkLinear,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        d: int
    ) -> Tuple[float, float, float]:
        """
        Compute:
        1. Leading eigenvalue (H_eig max)
        2. Perpendicular eigenvalues (mean of rest)
        3. Slope between model output and GPR prediction
        """
        model.eval()
        with torch.no_grad():
            # Generate large X_inf for eigenvalue computation
            X_inf = torch.randn(1000, d, device=self.device)
            y_inf = X_inf[:, 0].unsqueeze(-1)
            
            # Compute eigenvalues using Gram matrix
            h_train = model.fc2(model.fc1(X_inf))
            gram = (h_train @ h_train.T) / X_inf.shape[0] / self.config.N
            eigvals = torch.linalg.eigvalsh(gram).cpu().numpy()
            
            eigs_sorted = np.sort(eigvals)[::-1]
            leading_eig = float(eigs_sorted[0]) if len(eigs_sorted) > 0 else 0.0
            perp_eig = float(np.mean(eigs_sorted[1:])) if len(eigs_sorted) > 1 else 0.0
            
            # Compute slope: correlation between model output and GPR on test set
            try:
                gpr_pred = gpr_dot_product_explicit(X_train, y_train, X_test, 1e-5)
                model_pred = model(X_test)
                
                # Linear regression slope: y = slope * x
                slope = torch.cov(torch.stack([model_pred.squeeze(), gpr_pred.squeeze()]))[0, 1] / (torch.var(gpr_pred) + 1e-8)
                slope = float(slope.item())
            except:
                slope = 0.0
        
        return leading_eig, perp_eig, slope
    
    def log_metrics_for_d(self, d: int, dataset_idx: int, ek_loss: float):
        """Log running averages of metrics for a given d."""
        if d not in self.loss_tracker:
            self.loss_tracker[d] = []
            self.eigs_leading[d] = []
            self.eigs_perp[d] = []
            self.slope_tracker[d] = []
        
        # Compute running averages
        avg_loss = np.mean(self.loss_tracker[d]) if self.loss_tracker[d] else 0.0
        avg_eig_leading = np.mean(self.eigs_leading[d]) if self.eigs_leading[d] else 0.0
        avg_eig_perp = np.mean(self.eigs_perp[d]) if self.eigs_perp[d] else 0.0
        avg_slope = np.mean(self.slope_tracker[d]) if self.slope_tracker[d] else 0.0
        
        # Log to TensorBoard
        global_step = d * self.config.num_datasets + dataset_idx
        
        self.writer.add_scalars(f'd_{d}/loss', {
            'network_empirical': avg_loss,
            'ek_predicted': ek_loss,
        }, global_step)
        
        self.writer.add_scalar(f'd_{d}/eigenvalue_leading', avg_eig_leading, global_step)
        self.writer.add_scalar(f'd_{d}/eigenvalue_perp', avg_eig_perp, global_step)
        self.writer.add_scalar(f'd_{d}/model_gpr_slope', avg_slope, global_step)
        
        # Log histograms
        if self.eigs_leading[d]:
            self.writer.add_histogram(f'd_{d}/eigs_leading_hist', np.array(self.eigs_leading[d]), global_step)
        if self.eigs_perp[d]:
            self.writer.add_histogram(f'd_{d}/eigs_perp_hist', np.array(self.eigs_perp[d]), global_step)
    
    def train_ensemble_on_dataset(
        self,
        d: int,
        dataset_idx: int,
        seed_offset: int,
        pbar: Optional[Any] = None,
    ) -> Dict:
        """Train a full ensemble (5 networks) on a single dataset."""
        P = int(np.round(self.config.P_factor * d))
        
        # Generate dataset with train/test split
        dataset_seed = self.seeds[seed_offset % len(self.seeds)] + dataset_idx * 1000
        X_train, X_test, y_train, y_test = generate_synthetic_data(d, P, seed=dataset_seed, device=self.device)
        
        ensemble_models = []
        ensemble_histories = []
        ensemble_predictions = []
        dataset_losses = []
        dataset_eigs_leading = []
        dataset_eigs_perp = []
        dataset_slopes = []
        
        for net_idx in range(self.config.ensemble_size):
            # Create model
            model = FCN3NetworkLinear(d, self.config.N, self.config.N, device=self.device)
            
            # Train model with Langevin dynamics and weight regularization
            history = train_network(
                model, X_train, y_train,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                device=self.device,
                temperature=self.config.temperature,
                pbar=pbar
            )
            
            # Get predictions on train and test sets
            model.eval()
            with torch.no_grad():
                pred_train = model(X_train).cpu().numpy()
                pred_test = model(X_test).cpu().numpy()
            
            # Compute loss on test set
            test_loss = float(np.sum((pred_test - y_test.cpu().numpy()) ** 2))
            dataset_losses.append(test_loss)
            
            # Compute eigenvalues and slope
            leading_eig, perp_eig, slope = self.compute_eigenvalues_and_slope(
                model, X_train, X_test, y_train, y_test, d
            )
            dataset_eigs_leading.append(leading_eig)
            dataset_eigs_perp.append(perp_eig)
            dataset_slopes.append(slope)
            
            ensemble_models.append(model)
            ensemble_histories.append(history)
            ensemble_predictions.append({
                'pred_train': pred_train.tolist(),
                'pred_test': pred_test.tolist(),
                'y_train': y_train.cpu().numpy().tolist(),
                'y_test': y_test.cpu().numpy().tolist(),
            })
        
        # Track metrics for logging
        self.loss_tracker[d] = self.loss_tracker.get(d, []) + dataset_losses
        self.eigs_leading[d] = self.eigs_leading.get(d, []) + dataset_eigs_leading
        self.eigs_perp[d] = self.eigs_perp.get(d, []) + dataset_eigs_perp
        self.slope_tracker[d] = self.slope_tracker.get(d, []) + dataset_slopes
        
        return {
            'models': ensemble_models,
            'histories': ensemble_histories,
            'predictions': ensemble_predictions,
            'X_train': X_train.cpu().numpy(),
            'X_test': X_test.cpu().numpy(),
            'y_train': y_train.cpu().numpy(),
            'y_test': y_test.cpu().numpy(),
            'P': P,
            'd': d
        }
    
    def train_all_ensembles(self) -> Dict:
        """Train all ensembles across all d values and datasets."""
        all_results = {}
        
        # Calculate total epochs for progress bar
        total_epochs = sum(
            self.config.ensemble_size * self.config.epochs
            for d in self.config.d_range
        )
        
        with tqdm(total=total_epochs, desc="Overall Training Progress") as pbar_overall:
            for d in self.config.d_range:
                results_d = []
                print(f"\n{'='*60}")
                print(f"Training for d={d}")
                print(f"{'='*60}")
                
                # Compute EK loss for this d
                P = int(np.round(self.config.P_factor * d))
                ek_loss_dict = compute_ek_loss(d, P, kappa=self.config.kappa_ek, chi=self.config.chi)
                ek_loss = ek_loss_dict['loss']
                
                for dataset_idx in tqdm(range(self.config.num_datasets), desc=f"Datasets (d={d})"):
                    ensemble_data = self.train_ensemble_on_dataset(
                        d=d,
                        dataset_idx=dataset_idx,
                        seed_offset=dataset_idx % len(self.seeds),
                        pbar=pbar_overall
                    )
                    results_d.append(ensemble_data)
                    
                    # Log metrics after each dataset
                    self.log_metrics_for_d(d, dataset_idx, ek_loss)
                
                all_results[d] = results_d
        
        self.writer.close()
        return all_results


def compute_ek_prediction(d: int, kappa: float = 0.5, chi: float = 1.0) -> float:
    """
    Compute EK prediction: (1/d) / (1/d + kappa) * x[0]
    
    For loss comparison:
    bias = (κ / (lH + κ))^2
    variance = (κ) / (chi * P) * lH / (lH + κ)
    loss = (bias + variance) * chi / κ
    
    With lH = 1/d (NNGP), chi = 1.0
    """
    # For normalized predictions
    prediction_factor = (1.0 / d) / (1.0 / d + kappa)
    return prediction_factor


def compute_ek_loss(d: int, P: int, kappa: float = 0.5, chi: float = 1.0) -> Dict[str, float]:
    """
    Compute EK predicted loss = bias + variance.
    
    bias = (κ / (lH + κ))^2
    variance = (κ) / (chi * P) * lH / (lH + κ)
    loss = (bias + variance) * chi / κ
    
    With lH = 1/d (NNGP)
    """
    lH = 1.0 / d
    
    bias = (kappa / (lH + kappa)) ** 2
    variance = (kappa / (chi * P)) * (lH / (lH + kappa))
    loss = (bias + variance) * chi / kappa
    
    return {
        'bias': bias,
        'variance': variance,
        'loss': loss,
        'lH': lH
    }


def main():
    """Main execution function."""
    print("="*70)
    print("EK-Comparison: Network vs Theory")
    print("="*70)
    
    # Setup
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = ExperimentConfig()
    print(f"\nConfiguration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Create output directories
    data_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/data')
    networks_dir = data_dir / 'networks'
    results_dir = data_dir / 'results'
    networks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = results_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Train ensembles
    trainer = NetworkTrainer(config, device=device)
    all_results = trainer.train_all_ensembles()
    
    # Save results
    results_path = results_dir / 'ensemble_results.json'
    print(f"\nSaving results to: {results_path}")
    
    # Compute EK predictions and empirical losses
    print(f"\nComputing EK predictions and empirical losses...")
    comparison_data = {}
    
    for d in config.d_range:
        P = int(np.round(config.P_factor * d))
        ek_pred = compute_ek_prediction(d, kappa=config.kappa_ek, chi=config.chi)
        ek_loss = compute_ek_loss(d, P, kappa=config.kappa_ek, chi=config.chi)
        
        # Compute average empirical loss over all datasets/ensembles
        empirical_losses = []
        all_predictions = []
        
        for dataset_idx, ensemble_data in enumerate(all_results[d]):
            for net_idx, history in enumerate(ensemble_data['histories']):
                if history['loss']:
                    empirical_losses.append(history['loss'][-1])
            
            # Collect predictions for this dataset
            for pred_dict in ensemble_data['predictions']:
                all_predictions.append(pred_dict)
        
        avg_empirical_loss = np.mean(empirical_losses) if empirical_losses else 0.0
        std_empirical_loss = np.std(empirical_losses) if empirical_losses else 0.0
        
        comparison_data[d] = {
            'P': P,
            'ek_prediction_factor': ek_pred,
            'ek_loss': ek_loss,
            'empirical_loss_mean': avg_empirical_loss,
            'empirical_loss_std': std_empirical_loss,
            'num_samples': len(empirical_losses),
            'predictions': all_predictions
        }
        
        print(f"\nd={d}, P={P}:")
        print(f"  EK Prediction Factor: {ek_pred:.6f}")
        print(f"  EK Loss (bias + variance): {ek_loss['loss']:.6e}")
        print(f"    - Bias: {ek_loss['bias']:.6e}")
        print(f"    - Variance: {ek_loss['variance']:.6e}")
        print(f"  Empirical Loss (mean ± std): {avg_empirical_loss:.6e} ± {std_empirical_loss:.6e}")
    
    # Save comparison data
    comparison_path = results_dir / 'ek_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison data saved to: {comparison_path}")
    
    print(f"\n{'='*70}")
    print("Training and analysis complete!")
    print(f"{'='*70}")
    
    return all_results, comparison_data, config


if __name__ == '__main__':
    results, comparison, config = main()
