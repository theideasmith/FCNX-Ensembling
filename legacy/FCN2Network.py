import torch
import torch.nn as nn
from opt_einsum import contract, contract_path

import standard_hyperparams as hp
from typing import Tuple, Optional, Callable, Dict

# Add these after imports
INIT_SEED = 222
try:
    DEVICE = hp.DEVICE
except AttributeError:
    DEVICE = 'cpu'

class FCN2Network(nn.Module):
    """
    A base class for a two-layer fully connected neural network.
    Allows for specification of activation functions for the layer.
    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer. 
        activation (Callable): The activation function for the layer.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_activation_width: int,
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 weight_sigma: Tuple[float,float]
                 ) -> None:
                 
        """
        Initializes the Network.
        Args:
            input_dim (int): The dimensionality of the input (d).
            activation (Callable[[torch.Tensor], torch.Tensor]): The activation function for the layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_activation_width, bias=False)
        self.fc2 = nn.Linear(hidden_activation_width, 1, bias=False)
        self.activation = activation
        self.weight_sigma : Tuple[float,float] = weight_sigma
        self._initialize_weights()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return super()._save_to_state_dict(destination, prefix, keep_vars)
    
    @staticmethod
    def model_from_hyperparameters(hyperparameters) -> 'FCN2Network':
        sigma_1 = hyperparameters.get('weight_sigma1', 1.0)
        sigma_2 = hyperparameters.get('weight_sigma2', 1.0)

        input_dim :int  = hyperparameters.get('input_dimension', 1)
        hidden_width:int= hyperparameters.get('hidden_width', 10)
        model = FCN2Network(input_dim,hidden_width,hyperparameters['activation'], (sigma_1,sigma_2))

        return model
    
    
    def _reset_with_weight_sigma(self, weight_sigma : tuple = (1.0,1.0)) -> None:
        """
        Initializes the weights of the network from centered Gaussian distributions
        with the specified standard deviations.

        Args:
            sigma1 (float): Standard deviation for the first layer's weights.
            sigma2 (float): Standard deviation for the second layer's weights.
        """
        with torch.no_grad():
            self.fc1.weight.data.normal_(0, (self.weight_sigma[0])**0.5)
            self.fc2.weight.data.normal_(0, (self.weight_sigma[1])**0.5)
        return self
    
    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the network from a centered Gaussian distribution
        with the specified standard deviation.
        Args:
            sigma (float): Standard deviation for the layer's weights.
        """
        weight_sigma = self.weight_sigma
        print(f"INITIALIZING FCN2 WITH WEIGHTS: {weight_sigma}")
        with torch.no_grad():
            self.fc1.weight.data.normal_(mean=0, std=weight_sigma[0]**0.5)
            self.fc2.weight.data.normal_(mean=0, std=weight_sigma[1]**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).
        """
        x = x.to(hp.DEVICE)
        # Preactivation of the layer
        h1: torch.Tensor = self.fc1(x)
        h2: torch.Tensor = self.fc2(h1)
        # Activation of the layer
        a1: torch.Tensor = self.activation(h2)
        return a1

class FCN_2_Ensemble(nn.Module):
    def __init__(self, d, n1, s2W, s2A, ensembles=1, init_seed=None):
        super().__init__()
        if init_seed is None:
            torch.manual_seed(INIT_SEED)

        self.arch = [d, n1]
        self.d = d
        self.n1 = n1
        self.W0 = nn.Parameter(torch.normal(mean=0.0,
            std=torch.full((ensembles, n1, d), s2W ** 0.5)).to(DEVICE),
            requires_grad=True)
        self.A = nn.Parameter(torch.normal(
            mean=0.0,
            std=torch.full((ensembles, n1), s2A ** 0.5)).to(DEVICE),
            requires_grad=True)

    def forward(self, X):
        Xp = X.squeeze()
        return contract(
            'ik,ikl,ul->ui',
            self.A, self.W0, Xp,
            backend='torch'
        )

    def h_activation(self, X):
        return contract(
            'ikl,ul->uik',
            self.W0, X,
            backend='torch'
        )

class FCN2NetworkEnsembleErf(nn.Module):
    def __init__(self, d, n1, P, ens=1, weight_initialization_variance=(1.0, 1.0), device=DEVICE):
        super().__init__()
        self.arch = [d, n1]
        self.d = d
        self.n1 = n1
        self.ens = ens
        self.num_samples = P
        self.device = device

        self.W0 = nn.Parameter(
            torch.normal(mean=0.0, std=torch.full((ens, n1, d), weight_initialization_variance[0] ** 0.5)).to(device),
            requires_grad=True
        )
        self.A = nn.Parameter(
            torch.normal(mean=0.0, std=torch.full((ens, n1), weight_initialization_variance[1] ** 0.5)).to(device),
            requires_grad=True
        )

        self.forward_path_h0 = None
        self.forward_path_f = None
        if self.num_samples is not None:
            self._precompute_einsum_paths_h0(self.num_samples)
            self._precompute_einsum_paths_f(self.num_samples)

    def _precompute_einsum_paths_h0(self, num_samples):
        eq = 'qkl,ul->uqk'
        shapes = [
            (self.ens, self.n1, self.d),
            (num_samples, self.d)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path_h0 = path

    def _precompute_einsum_paths_f(self, num_samples):
        eq = 'qk,uqk->uq'
        shapes = [
            (self.ens, self.n1),
            (num_samples, self.ens, self.n1)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path_f = path

    def h_activation(self, X):
        h0 = contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
            backend='torch'
        )
        return torch.erf(h0)

    def forward(self, X):
        h0 = contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
            backend='torch'
        )
        a0 = torch.erf(h0)
        f = contract(
            'qk,uqk->uq',
            self.A, a0,
            optimize=self.forward_path_f if self.forward_path_f is not None else None,
            backend='torch'
        ).unsqueeze(1)
        return f

