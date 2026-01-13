import torch
import torch.nn as nn
from opt_einsum import contract, contract_path

class FCN3NetworkEnsembleLinear(nn.Module):
    def __init__(self, d, n1, n2, ens=1, weight_initialization_variance=(1.0, 1.0, 1.0), device=None, num_samples=None):
        super().__init__()
        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.ens = ens
        if device is None:
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.W0 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n1, d), weight_initialization_variance[0]**0.5, device=device)),
                               requires_grad=True)
        self.W1 = nn.Parameter(torch.normal(mean=0.0,
                                           std=torch.full((ens, n2, n1), weight_initialization_variance[1]**0.5, device=device)),
                               requires_grad=True)
        self.A = nn.Parameter(torch.normal(mean=0.0,
                                          std=torch.full((ens, 1, n2), weight_initialization_variance[2]**0.5, device=device)),
                              requires_grad=True)
        # Pre-compute einsum paths for better performance
        self._precompute_einsum_paths(num_samples)
        # Pre-allocate single noise buffer for all parameters
        self.noise_buffer = torch.empty(1, device=device, dtype=torch.float32)

    def _precompute_einsum_paths(self, num_samples):
        """Pre-compute einsum paths for repeated operations"""
        if num_samples is None:
            num_samples = 400  # default fallback
        eq = 'eij,ejk,ekl,ul->uie'
        shapes = [
            (self.ens, 1, self.n2),
            (self.ens, self.n2, self.n1),
            (self.ens, self.n1, self.d),
            (num_samples, self.d)
        ]
        dummy_tensors = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes]
        path, _ = contract_path(eq, *dummy_tensors)
        self.forward_path = path

    def h1_activation(self, X):
        X = X.to(dtype=torch.float32)
        if X.dim() == 3: 
            contraction = 'ijk,ikl,unl->uij'
        elif X.dim() == 2:
            contraction = 'ijk,ikl,ul->uij'
        else:
            raise Exception(f"Dimensions of data {X.shape} are incompatible with dimensions of operator W0 with dimensions {self.W0.shape}")
        return contract(
            contraction,
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
        """
        return contract(
            'eij,ejk,ekl,ul->uie',
            self.A, self.W1, self.W0, X,
            backend='torch',
            optimize=self.forward_path
        ) 