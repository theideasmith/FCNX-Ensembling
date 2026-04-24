import numpy as np
import torch


def qb_projection_eigenvalues(model, d, device, N=10000, k=900, p=25, chunk_size=4096):
    """
    FCN3-specific QB projection eigenvalue estimation using `model.h1_preactivation`.
    Empirical eigenvalues are returned without extra scaling.
    """
    with torch.no_grad():
        l = k + p
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        Omega = torch.randn((N, l), device=device, dtype=torch.float32)
        res = torch.zeros((N, l), device=device, dtype=torch.float32)

        h1_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            torch.manual_seed(seed + start)
            np.random.seed(seed + start)
            X_chunk = torch.randn(end - start, d, device=device, dtype=torch.float32)
            batch_h1 = model.h1_preactivation(X_chunk)
            h1_chunks.append(batch_h1)
        h1 = torch.cat(h1_chunks, dim=0)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h1, h1, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)
        Z = torch.zeros((N, l), device=device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h1, h1) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        Z_T = torch.nan_to_num(Z.T, nan=0.0, posinf=0.0, neginf=0.0)
        Ut, S, _ = torch.linalg.svd(Z_T)
        m, n = Z.shape[1], Z.shape[0]
        k_eff = min(m, n)
        U = torch.matmul(Q, Ut)

        # Probe vectors (He3 normalized by 1/sqrt(6))
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = torch.randn((N, d), device=device, dtype=torch.float32)
        Y1 = X
        Y3 = (X ** 3 - 3.0 * X) / (6 ** 0.5)
        Y1_norm = Y1 / Y1.norm(axis=0)
        Y3_norm = Y3 / Y3.norm(axis=0)

        left_eigenvalues_Y1 = (Y1_norm.t() @ U @ torch.diag(S[:k_eff]) @ U.T @ Y1_norm).diagonal() / N
        left_eigenvalues_Y3 = (Y3_norm.t() @ U @ torch.diag(S[:k_eff]) @ U.T @ Y3_norm).diagonal() / N

        lJ1T = float(left_eigenvalues_Y1[0].cpu().numpy())
        lJ1P = float(left_eigenvalues_Y1[1].cpu().numpy())
        lJ3T = float(left_eigenvalues_Y3[0].cpu().numpy())
        lJ3P = float(left_eigenvalues_Y3[1:].mean().cpu().numpy())
        all_eigvals = np.sort(S.detach().cpu().numpy())[::-1] 

        return {
            "summary": np.array([lJ1T, lJ1P, lJ3T, lJ3P]),
            "all_eigenvalues": all_eigvals,
        }


def randomized_svd_spectrum(model, d, device, N=10000, k=50, p=10, chunk_size=4096):
    """
    FCN3-specific randomized SVD spectrum using `model.h1_preactivation`.
    Returns singular values (descending) normalized by N.
    """
    with torch.no_grad():
        l = k + p
        dtype = torch.float32
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        Omega = torch.randn((N, l), device=device, dtype=dtype)
        res = torch.zeros((N, l), device=device, dtype=dtype)

        h1_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            torch.manual_seed(seed + start)
            np.random.seed(seed + start)
            X_chunk = torch.randn(end - start, d, device=device, dtype=dtype)
            batch_h1 = model.h1_preactivation(X_chunk)
            h1_chunks.append(batch_h1)
        h1 = torch.cat(h1_chunks, dim=0)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h1, h1, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)
        Z = torch.zeros((N, l), device=device, dtype=dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h1 = h1[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h1, h1) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        Z_T = torch.nan_to_num(Z.T, nan=0.0, posinf=0.0, neginf=0.0)
        _, S, _ = torch.linalg.svd(Z_T)
        return S.detach().cpu().numpy() / N
