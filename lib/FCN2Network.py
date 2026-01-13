"""
Two-layer fully connected neural network with ensemble support and kernel eigenvalue computation.
"""

import torch
import torch.nn as nn
from opt_einsum import contract, contract_path
from typing import Tuple, Optional
import standard_hyperparams as hp


class FCN2NetworkActivationGeneric(nn.Module):
    """
    Two-layer fully connected network with ensemble support.
    Architecture: Input(d) -> Hidden(n1) with activation -> Output(1) linear readout
    
    Supports erf and linear activations for the hidden layer.
    Computes eigenvalues of the pre-activation kernel H for the hidden layer.
    """
    
    def __init__(self, d, n1, P, ens=1, 
                 activation="erf",
                 weight_initialization_variance=(1.0, 1.0), 
                 device=hp.DEVICE):
        """
        Args:
            d: Input dimension
            n1: Hidden layer width
            P: Number of training samples (for einsum path optimization)
            ens: Number of ensemble members
            activation: "erf" or "linear"
            weight_initialization_variance: (sigma_W0^2, sigma_A^2)
            device: torch device
        """
        super().__init__()
        self.d = d
        self.n1 = n1
        self.ens = ens
        self.ensembles = self.ens
        self.num_samples = P
        self.device = device
        self.activation_name = activation
        
        # Validate activation
        if activation not in ["erf", "linear"]:
            raise ValueError(f"Activation must be 'erf' or 'linear', got '{activation}'")
        
        # Initialize weights
        v0, v1 = weight_initialization_variance
        std0 = v0 ** 0.5
        std1 = v1 ** 0.5
        
        self.W0 = nn.Parameter(
            torch.empty(ens, n1, d, device=device).normal_(0.0, std0)
        )
        self.A = nn.Parameter(
            torch.empty(ens, n1, device=device).normal_(0.0, std1)
        )
        
        # Precompute einsum paths
        self.forward_path_h0 = None
        self.forward_path_f = None
        if P is not None:
            self._precompute_einsum_paths(P)
    
    def _precompute_einsum_paths(self, num_samples):
        """Precompute einsum contraction paths for efficiency."""
        # Path for h0: W0 @ X
        eq_h0 = 'qkl,ul->uqk'
        shapes_h0 = [(self.ens, self.n1, self.d), (num_samples, self.d)]
        dummy_h0 = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes_h0]
        path_h0, _ = contract_path(eq_h0, *dummy_h0)
        self.forward_path_h0 = path_h0
        
        # Path for output: A @ a0
        eq_f = 'qk,uqk->uq'
        shapes_f = [(self.ens, self.n1), (num_samples, self.ens, self.n1)]
        dummy_f = [torch.empty(s, device=self.device, dtype=torch.float32) for s in shapes_f]
        path_f, _ = contract_path(eq_f, *dummy_f)
        self.forward_path_f = path_f
    
    def h0_preactivation(self, X):
        """Compute pre-activation of hidden layer: W0 @ X
        
        Returns:
            h0: shape (P, ens, n1) - pre-activations before activation function
        """
        h0 = contract(
            'qkl,ul->uqk',
            self.W0, X,
            optimize=self.forward_path_h0 if self.forward_path_h0 is not None else None,
            backend='torch'
        )
        return h0
    
    def h0_activation(self, X):
        """Compute activation of hidden layer.
        
        Returns:
            a0: shape (P, ens, n1) - activations after activation function
        """
        h0 = self.h0_preactivation(X)
        
        if self.activation_name == "erf":
            return torch.erf(h0)
        elif self.activation_name == "linear":
            return h0
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, X):
        """Forward pass through the network.
        
        Args:
            X: Input data of shape (P, d)
            
        Returns:
            output: shape (P, ens) - network outputs for each ensemble member
        """
        a0 = self.h0_activation(X)
        output = contract(
            'qk,uqk->uq',
            self.A, a0,
            optimize=self.forward_path_f if self.forward_path_f is not None else None,
            backend='torch'
        )
        return output


    def H_eig(self, X, Y, std=False):
        """Compute trace-normalized eigenvalues of the pre-activation kernel H for hidden layer.
        
        The kernel is defined as:
            K[u,v] = (1/(n1*P)) * sum_{q,k} h0[u,q,k] * h0[v,q,k]
        
        where h0[u,q,k] = (W0[q] @ X[u])[k] is the pre-activation of sample u,
        ensemble q, neuron k.
        
        We compute per-ensemble kernels K_q[u,v] = (1/(n1*P)) sum_k h0[u,q,k] * h0[v,q,k],
        then average over ensembles to get K = mean_q(K_q).
        
        Eigenvalues are computed via Rayleigh quotient: lambda_i = Y[:,i]^T K Y[:,i] / ||Y[:,i]||^2
        
        Args:
            X: Input data of shape (P, d)
            Y: Eigenfunctions to project onto, shape (P,) or (P, M) for M eigenfunctions.
               If Y is 1D with shape (P,), each element is treated as a separate eigenfunction
               (equivalent to using an identity matrix), returning P eigenvalues.
            std: If True and ens > 1, return (mean_eigenvalues, std_eigenvalues)
            
        Returns:
            If std=False: eigenvalues tensor of shape (M,) or (P,) if Y is 1D
            If std=True: (eigenvalues, std_eigenvalues) where std is over ensemble members
        """
        with torch.no_grad():
            # Compute pre-activations: (P, ens, n1)
            h0 = self.h0_activation(X)
            P_actual = h0.shape[0]
            
            # Build per-ensemble kernels: K_q[u,v] = (1/(n1*P)) sum_k h0[u,q,k] * h0[v,q,k]
            # Using einsum: 'uqk,vqk->quv' gives (ens, P, P)
            K_per_ens = contract(
                'uqk,vqk->quv',
                h0, h0,
                backend='torch'
            ) / (self.n1 * P_actual)
            
            # Handle Y shape
            is_1d = Y.dim() == 1
            if is_1d:
                # Treat 1D Y as an identity matrix: each element is a separate eigenfunction
                # Convert to diagonal form: Y[i] becomes the i-th basis vector
                M = Y.shape[0]  # P eigenfunctions
                Y_matrix = torch.diag(Y)  # (P, P) diagonal matrix
            else:
                Y_matrix = Y
                M = Y.shape[1]  # number of eigenfunction columns
            
            # Compute Rayleigh quotients per ensemble: lambda_q[m] = Y[:,m]^T K_q Y[:,m] / ||Y[:,m]||^2
            # Numerator: Y^T K_q Y for each q,m
            numerator = torch.zeros(self.ens, M, device=self.device)
            for m in range(M):
                y_m = Y_matrix[:, m]  # (P,)
                # K_q @ y_m: (ens, P, P) @ (P,) -> (ens, P)
                Ky = torch.einsum('quv,v->qu', K_per_ens, y_m)
                # y_m^T @ (K_q @ y_m): (P,) @ (ens, P) -> (ens,)
                numerator[:, m] = torch.einsum('u,qu->q', y_m, Ky) / P_actual
            
            # Denominator: ||Y[:,m]||^2 for each m
            y_norms_sq = torch.sum(Y_matrix * Y_matrix, dim=0) / P_actual  # (M,)
            
            # Eigenvalues per ensemble: (ens, M)
            eigenvalues_per_ens = numerator / y_norms_sq.unsqueeze(0)

            
            # Average over ensembles
            eigenvalues_mean = torch.mean(eigenvalues_per_ens, dim=0)  # (M,)
            
            if std and self.ens > 1:
                eigenvalues_std = torch.std(eigenvalues_per_ens, dim=0)  / (self.ens) ** 0.5  # (M,)
                return eigenvalues_mean, eigenvalues_std
            else:
                return eigenvalues_mean
    
    def H_eig_random_svd(self, X, k=100, p=25):
        """Compute leading eigenvalues using random SVD (Halko et al. 2011).
        
        Efficient for large P without storing full P×P kernel matrix.
        
        Args:
            X: Input data of shape (P, d)
            k: Number of eigenvalues to return
            p: Oversampling parameter (total rank = k + p)
            
        Returns:
            eigenvalues: Top k eigenvalues of the H kernel
        """
        with torch.no_grad():
            h0 = self.h0_activation(X)  # (P, ens, n1)
            P_actual = h0.shape[0]
            l = k + p
            
            # Random projection matrix
            Omega = torch.randn(P_actual, l, device=self.device, dtype=h0.dtype)
            
            # Build Y = K @ Omega where K is the kernel (averaged over ensembles)
            # K[u,v] = (1/(n1*P)) sum_{q,k} h0[u,q,k] * h0[v,q,k]
            Y = torch.zeros(P_actual, l, device=self.device, dtype=h0.dtype)
            chunk_size = 2048  # Process in chunks to save memory
            
            for u in range(0, P_actual, chunk_size):
                u_end = min(u + chunk_size, P_actual)
                h0_u = h0[u:u_end]  # (chunk, ens, n1)
                
                for v in range(P_actual):
                    h0_v = h0[v:v+1]  # (1, ens, n1)
                    # K[u,v] = (1/(n1*P)) * sum_q,k h0[u,q,k] * h0[v,q,k]
                    K_uv = torch.einsum('cqn,eqn->', h0_u, h0_v) / (self.n1 * P_actual)
                    Y[u:u_end, :] += K_uv * Omega[v:v+1, :]
            
            # QR decomposition
            Q, _ = torch.linalg.qr(Y)
            Q = Q[:, :l]
            
            # Compute B = Q^T @ K @ Q (small l×l matrix)
            B = torch.zeros(l, l, device=self.device, dtype=h0.dtype)
            for i in range(l):
                for j in range(l):
                    for u in range(P_actual):
                        for v in range(P_actual):
                            h0_u = h0[u:u+1]
                            h0_v = h0[v:v+1]
                            K_uv = torch.einsum('eqn,fqn->', h0_u, h0_v) / (self.n1 * P_actual)
                            B[i, j] += Q[u, i] * K_uv * Q[v, j]
            
            # Eigenvalue decomposition of B
            evals_B = torch.linalg.eigvalsh(B)
            
            # Take largest k eigenvalues (sorted in ascending order)
            eigenvalues = evals_B[-k:].flip(0)
            
            return eigenvalues
    def H_Kernel(self, X, avg_ens=True):
        """Compute the pre-activation kernel H for the hidden layer.
        
        The kernel is defined as:
            K[u,v] = (1/(n1*P)) * sum_{q,k} h0[u,q,k] * h0[v,q,k]
        """
        with torch.no_grad():
            h0 = self.h0_activation(X)  # (P, ens, n1)
            P_actual = h0.shape[0]
            
            # Build per-ensemble kernels: K_q[u,v] = (1/(n1*P)) sum_k h0[u,q,k] * h0[v,q,k]
            K_per_ens = contract(
                'uqk,vqk->quv',
                h0, h0,
                backend='torch'
            ) / (self.n1)
            
            if avg_ens is True:
            # Average over ensembles
                K = torch.mean(K_per_ens, dim=0)  # (P, P)
            else: 
                K = K_per_ens  # (ens, P, P)
            return K

class FCN2NetworkEnsembleErf(FCN2NetworkActivationGeneric):
    """Convenience class for erf activation (backward compatibility)."""
    
    def __init__(self, d, n1, P, ens=1, 
                 weight_initialization_variance=(1.0, 1.0), 
                 device=hp.DEVICE):
        super().__init__(d, n1, P, ens, 
                        activation="erf",
                        weight_initialization_variance=weight_initialization_variance,
                        device=device)


class FCN2NetworkEnsembleLinear(FCN2NetworkActivationGeneric):
    """Convenience class for linear activation (backward compatibility)."""
    
    def __init__(self, d, n1, P, ens=1,
                 weight_initialization_variance=(1.0, 1.0),
                 device=hp.DEVICE):
        super().__init__(d, n1, P, ens,
                        activation="linear",
                        weight_initialization_variance=weight_initialization_variance,
                        device=device)
