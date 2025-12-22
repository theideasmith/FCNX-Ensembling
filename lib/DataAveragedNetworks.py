"""
Dataset-averaged ensemble networks for FCN2 and FCN3.

These networks support training on multiple datasets in parallel, with an additional
dataset index beyond the ensemble index. This allows for dataset-averaging at the end.
"""

import torch
import torch.nn as nn
from opt_einsum import contract, contract_path
from typing import Tuple, Optional
import standard_hyperparams as hp


class FCN2NetworkDataAveragedEnsemble(nn.Module):
    """
    Two-layer FCN with both ensemble and dataset indices.
    Architecture: Input(d) -> Hidden(n1) with activation -> Output(1) linear readout
    
    Weights have shape (num_datasets, num_ensembles, ...)
    Allows parallel training on multiple datasets and dataset averaging.
    """
    
    def __init__(self, d, n1, P, num_datasets=1, num_ensembles=1, 
                 activation="erf",
                 weight_initialization_variance=(1.0, 1.0), 
                 device=hp.DEVICE):
        """
        Args:
            d: Input dimension
            n1: Hidden layer width
            P: Number of training samples (for einsum path optimization)
            num_datasets: Number of datasets
            num_ensembles: Number of ensemble members per dataset
            activation: "erf" or "linear"
            weight_initialization_variance: (sigma_W0^2, sigma_A^2)
            device: torch device
        """
        super().__init__()
        self.d = d
        self.n1 = n1
        self.num_datasets = num_datasets
        self.num_ensembles = num_ensembles
        self.num_samples = P
        self.device = device
        self.activation_name = activation
        
        if activation not in ["erf", "linear"]:
            raise ValueError(f"Activation must be 'erf' or 'linear', got '{activation}'")
        
        # Initialize weights with dataset and ensemble indices
        v0, v1 = weight_initialization_variance
        std0 = v0 ** 0.5
        std1 = v1 ** 0.5
        
        self.W0 = nn.Parameter(
            torch.empty(num_datasets, num_ensembles, n1, d, device=device).normal_(0.0, std0)
        )
        self.A = nn.Parameter(
            torch.empty(num_datasets, num_ensembles, n1, device=device).normal_(0.0, std1)
        )
        
        self.forward_path_h0 = None
        self.forward_path_f = None
        if P is not None:
            self._precompute_einsum_paths(P)
    
    def _precompute_einsum_paths(self, num_samples):
        """Precompute einsum contraction paths for efficiency."""
        # Path for h0: W0 @ X (per dataset/ensemble)
        eq_h0 = 'dqkl,ul->udqk'
        shapes_h0 = [(self.num_datasets, self.num_ensembles, self.n1, self.d), (num_samples, self.d)]
        dummy_h0 = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes_h0]
        try:
            path_h0, _ = contract_path(eq_h0, *dummy_h0)
            self.forward_path_h0 = path_h0
        except:
            pass
        
        # Path for output: A @ a0
        eq_f = 'dqk,udqk->udq'
        shapes_f = [(self.num_datasets, self.num_ensembles, self.n1), 
                    (num_samples, self.num_datasets, self.num_ensembles, self.n1)]
        dummy_f = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes_f]
        try:
            path_f, _ = contract_path(eq_f, *dummy_f)
            self.forward_path_f = path_f
        except:
            pass
    
    def h0_preactivation(self, X):
        """Compute pre-activation: (P, num_datasets, num_ensembles, n1)"""
        h0 = contract(
            'dqkl,ul->udqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
            backend='torch'
        )
        return h0
    
    def h0_activation(self, X):
        """Compute activation: (P, num_datasets, num_ensembles, n1)"""
        h0 = self.h0_preactivation(X)
        
        if self.activation_name == "erf":
            return torch.erf(h0)
        elif self.activation_name == "linear":
            return h0
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, X):
        """Forward pass. Returns (P, num_datasets, num_ensembles)"""
        a0 = self.h0_activation(X)
        output = contract(
            'dqk,udqk->udq',
            self.A, a0,
            optimize=self.forward_path_f if self.forward_path_f is not None else None,
            backend='torch'
        )
        return output

    def H_eig_data_averaged(self, X, Y, std=False):
        """Compute eigenvalues of the data-averaged pre-activation kernel H.

        The kernel over samples u,v is averaged over dataset and ensemble axes:
            K[u,v] = (1/(n1 * P * D * Q)) * sum_{d,q,k} h0[u,d,q,k] * h0[v,d,q,k]

        where h0[u,d,q,k] are hidden pre-activations for sample u, dataset d,
        ensemble q, neuron k. We then compute Rayleigh quotients against one or
        more eigenfunction vectors provided in Y.

        Args:
            X: Input data of shape (P, d)
            Y: Eigenfunctions: shape (P,) or (P, M). If 1D, treated as diagonal
               projection producing P eigenvalues. If 2D, columns are eigenfunctions.
            std: If True, also return the standard deviation of eigenvalues across
                 the (dataset, ensemble) axes based on per-(d,q) projections.

        Returns:
            If std=False: eigenvalues tensor of shape (M,) or (P,) if Y is 1D
            If std=True: (eigenvalues_mean, eigenvalues_std) where std is across
                        all (d,q) combinations.
        """
        with torch.no_grad():
            # Hidden pre-activations: (P, D, Q, n1)
            h0 = self.h0_preactivation(X)
            P_actual = h0.shape[0]
            D = self.num_datasets
            Q = self.num_ensembles

            # Determine eigenfunction matrix
            is_1d = Y.dim() == 1
            if is_1d:
                M = Y.shape[0]
                Y_matrix = torch.diag(Y)  # (P, P)
            else:
                Y_matrix = Y
                M = Y.shape[1]

            # Full data-averaged kernel via contraction over d,q,k
            # K_avg[u,v] = (1/(n1 * P * D * Q)) sum_{d,q,k} h0[u,d,q,k] h0[v,d,q,k]
            K_avg = torch.einsum('udqk,vdqk->uv', h0, h0) / (self.n1 * P_actual * D * Q)

            # Compute eigenvalues via Rayleigh quotients against K_avg
            numerator_avg = torch.zeros(M, device=self.device)
            for m in range(M):
                y_m = Y_matrix[:, m]
                Ky = torch.mv(K_avg, y_m)  # (P,)
                numerator_avg[m] = torch.dot(y_m, Ky) / P_actual

            y_norms_sq = torch.sum(Y_matrix * Y_matrix, dim=0) / P_actual
            eigenvalues_mean = numerator_avg / y_norms_sq

            if std:
                # Build per-(d,q) kernels and eigenvalues to estimate spread
                # K_{d,q}[u,v] = (1/(n1 * P)) sum_k h0[u,d,q,k] h0[v,d,q,k]
                K_per_dq = torch.einsum('udqk,vdqk->dquv', h0, h0) / (self.n1 * P_actual)
                eigenvalues_per_dq = torch.zeros(D, Q, M, device=self.device)
                for m in range(M):
                    y_m = Y_matrix[:, m]
                    # For each (d,q): y^T K_{d,q} y / ||y||^2
                    Ky_dq = torch.einsum('dquv,v->dqu', K_per_dq, y_m)  # (D, Q, P)
                    num_dq = torch.einsum('u,dqu->dq', y_m, Ky_dq) / P_actual  # (D, Q)
                    eigenvalues_per_dq[:, :, m] = num_dq / y_norms_sq[m]

                # Std over all (d,q)
                eigenvalues_std = torch.std(eigenvalues_per_dq.reshape(D * Q, M), dim=0)
                return eigenvalues_mean, eigenvalues_std
            else:
                return eigenvalues_mean


class FCN3NetworkDataAveragedEnsemble(nn.Module):
    """
    Three-layer FCN with both ensemble and dataset indices.
    Architecture: Input(d) -> Hidden1(n1) with activation -> Hidden2(n2) with activation -> Output(1)
    
    Weights have shape (num_datasets, num_ensembles, ...)
    """
    
    def __init__(self, d, n1, n2, P, num_datasets=1, num_ensembles=1, 
                 activation="erf",
                 weight_initialization_variance=(1.0, 1.0, 1.0), 
                 device=hp.DEVICE):
        """
        Args:
            d: Input dimension
            n1: First hidden layer width
            n2: Second hidden layer width
            P: Number of training samples
            num_datasets: Number of datasets
            num_ensembles: Number of ensemble members per dataset
            activation: "erf" or "linear"
            weight_initialization_variance: (sigma_W0^2, sigma_W1^2, sigma_A^2)
            device: torch device
        """
        super().__init__()
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.num_datasets = num_datasets
        self.num_ensembles = num_ensembles
        self.num_samples = P
        self.device = device
        self.activation_name = activation
        
        if activation not in ["erf", "linear"]:
            raise ValueError(f"Activation must be 'erf' or 'linear', got '{activation}'")
        
        v0, v1, v2 = weight_initialization_variance
        std0 = v0 ** 0.5
        std1 = v1 ** 0.5
        std2 = v2 ** 0.5
        
        self.W0 = nn.Parameter(
            torch.empty(num_datasets, num_ensembles, n1, d, device=device).normal_(0.0, std0)
        )
        self.W1 = nn.Parameter(
            torch.empty(num_datasets, num_ensembles, n2, n1, device=device).normal_(0.0, std1)
        )
        self.A = nn.Parameter(
            torch.empty(num_datasets, num_ensembles, n2, device=device).normal_(0.0, std2)
        )
    
    def h0_preactivation(self, X):
        """Compute h0 preactivation: (P, num_datasets, num_ensembles, n1)"""
        return contract('dqkl,ul->udqk', self.W0, X, backend='torch')
    
    def h0_activation(self, X):
        """Compute h0 activation: (P, num_datasets, num_ensembles, n1)"""
        h0 = self.h0_preactivation(X)
        if self.activation_name == "erf":
            return torch.erf(h0)
        else:
            return h0
    
    def h1_preactivation(self, X):
        """Compute h1 preactivation: (P, num_datasets, num_ensembles, n2)"""
        h0 = self.h0_activation(X)
        return contract('dqkj,udqj->udqk', self.W1, h0, backend='torch')
    
    def h1_activation(self, X):
        """Compute h1 activation: (P, num_datasets, num_ensembles, n2)"""
        h1_pre = self.h1_preactivation(X)
        if self.activation_name == "erf":
            return torch.erf(h1_pre)
        else:
            return h1_pre
    
    def forward(self, X):
        """Forward pass. Returns (P, num_datasets, num_ensembles)"""
        h1 = self.h1_activation(X)
        output = contract('dqk,udqk->udq', self.A, h1, backend='torch')
        return output

    def H_eig_random_svd_data_averaged(self, X, k=100, p=25, chunk_size=2048):
        """
        Approximate leading eigenvalues of the data-averaged H kernel via random SVD.
        Averages over dataset and ensemble axes to produce a single kernel over samples.
        """
        with torch.no_grad():
            l = k + p
            h1 = self.h1_preactivation(X)  # (P, num_datasets, num_ensembles, n2)
            scale = float(self.num_datasets * self.num_ensembles * self.n1)

            N = X.shape[0]
            Omega = torch.randn((N, l), device=self.device, dtype=torch.float32)
            res = torch.zeros((N, l), device=self.device, dtype=torch.float32)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch_h1 = h1[start:end]
                res[start:end] = torch.einsum(
                    'bdek,Ndek,Nl->bl',
                    batch_h1, h1, Omega
                ) / scale

            Q, _ = torch.linalg.qr(res)
            Z = torch.zeros((N, l), device=self.device, dtype=torch.float32)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch_h1 = h1[start:end]
                K_uv = torch.einsum('bdek,Ndek->bN', batch_h1, h1) / scale
                Z[start:end] = torch.matmul(K_uv, Q)

            B = torch.matmul(Q.t(), Z)
            evals = torch.linalg.eigvalsh(B)
            ret, _ = (evals / N).sort(descending=True)
            return ret
