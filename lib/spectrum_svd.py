def qb_projection_eigenvalues(model, d, device, N=10000, k=900, p=25, chunk_size=4096):
    """
    Compute eigenvalues via Hermite/left projections using QB streaming, as in compute_empirical_j_spectrum_streaming.
    Returns a dict with summary (lJ1T, lJ1P, lJ3T, lJ3P) and all singular values.
    """
    with torch.no_grad():
        l = k + p
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = torch.randn((N, d), device=device, dtype=torch.float32)
        # QB decomposition
        Omega = torch.randn((N, l), device=device, dtype=torch.float32)
        res = torch.zeros((N, l), device=device, dtype=torch.float32)
        h0_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            torch.manual_seed(seed + start)
            np.random.seed(seed + start)
            X_chunk = torch.randn(end - start, d, device=device, dtype=torch.float32)
            batch_h0 = model.h0_activation(X_chunk)
            h0_chunks.append(batch_h0)
        h0 = torch.cat(h0_chunks, dim=0)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)
        Q, _ = torch.linalg.qr(res)
        Z = torch.zeros((N, l), device=device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)
        # SVD on Z.T for spectrum
        Z_T = Z.T
        Z_T = torch.nan_to_num(Z_T, nan=0.0, posinf=0.0, neginf=0.0)
        Ut, S, V = torch.linalg.svd(Z_T)
        m, n = Z.shape[1], Z.shape[0]
        k_eff = min(m, n)
        Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
        Sigma[:k_eff, :k_eff] = torch.diag(S[:k_eff])
        U = torch.matmul(Q, Ut)
        # Use the same seed for X as in QB
        torch.manual_seed(seed)
        np.random.seed(seed)
        Y1 = X[:, :]
        Y3 = (X[:, :] ** 3 - 3.0 * X[:, :])
        Y1_norm = Y1 / torch.norm(Y1, dim=0)
        Y3_norm = Y3 / torch.norm(Y3, dim=0)
        left_eigenvalues_Y1 = (torch.matmul(Y1_norm.t(), U) @ torch.diag(S[:k_eff]) @ torch.matmul(U.T, Y1_norm)).diagonal() / torch.norm(Y1_norm, dim=0) / X.shape[0]
        left_eigenvaluesY3 = (torch.matmul(Y3_norm.t(), U) @ torch.diag(S[:k_eff]) @ torch.matmul(U.T, Y3_norm)).diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]
        lJ1T = float(left_eigenvalues_Y1[0].cpu().numpy())
        lJ1P = float(left_eigenvalues_Y1[1].cpu().numpy())
        lJ3T = float(left_eigenvaluesY3[0].cpu().numpy())
        lJ3P = float(left_eigenvaluesY3[1:].mean().cpu().numpy())
        all_eigvals = np.sort(S.detach().cpu().numpy())[::-1] / X.shape[0]
        return {
            "summary": np.array([lJ1T, lJ1P, lJ3T, lJ3P]),
            "all_eigenvalues": all_eigvals,
        }
import torch
import numpy as np

def randomized_svd_spectrum(model, d, device, N=10000, k=50, p=10, chunk_size=4096):
    """
    Compute top-k singular values using QB streaming (as in j_random_QB_activation_generic_streaming).
    Returns the singular values (descending order).
    """
    with torch.no_grad():
        l = k + p
        dtype = torch.float32
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        Omega = torch.randn((N, l), device=device, dtype=dtype)
        res = torch.zeros((N, l), device=device, dtype=dtype)
        h0_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            torch.manual_seed(seed + start)
            np.random.seed(seed + start)
            X_chunk = torch.randn(end - start, d, device=device, dtype=dtype)
            batch_h0 = model.h0_activation(X_chunk)
            h0_chunks.append(batch_h0)
        h0 = torch.cat(h0_chunks, dim=0)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)
        Q, _ = torch.linalg.qr(res)
        Z = torch.zeros((N, l), device=device, dtype=dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)
        Z_T = Z.T
        Z_T = torch.nan_to_num(Z_T, nan=0.0, posinf=0.0, neginf=0.0)
        _, S, _ = torch.linalg.svd(Z_T)
        return S.detach().cpu().numpy() / N
