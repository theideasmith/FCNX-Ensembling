import torch
import numpy as np


def second_moment(tensor):
    k = tensor.flatten().shape[0]
    return torch.sum(tensor ** 2) / k

def compute_h3_projections_streaming(model, d, P_total=2_000_000, batch_size=100_000, device=None, perp_dim=1, silent=False):
    """
    Stream random inputs and compute cubic projections of hidden-layer activations.
    Returns a dict with mean/std/var/second_moment for target and perpendicular projections.
    If silent=True, disables the tqdm progress bar.
    """
    from tqdm import tqdm
    assert perp_dim >= 0 and perp_dim < d, f"perp_dim {perp_dim} out of range for d={d}"
    dtype = torch.float32
    ens = model.ens
    n1 = model.n1
    model.to(dtype)
    proj3_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj3_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    proj1_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    num_batches = P_total // batch_size
    remainder = P_total % batch_size
    batch_iter = range(num_batches + (1 if remainder > 0 else 0))
    if not silent:
        batch_iter = tqdm(batch_iter, desc="Batches", leave=True)
    with torch.no_grad():
        for i in batch_iter:
            bs = batch_size if i < num_batches else remainder
            if bs == 0:
                break
            X_batch = torch.randn(bs, d, dtype=dtype, device=device)
            x0 = X_batch[:, 0]
            x_perp = X_batch[:, perp_dim]
            phi3_target = (x0**3 - 3.0 * x0)
            phi3_perp = (x_perp**3 - 3.0 * x_perp)
            phi1_target = x0
            phi1_perp = x_perp
            a0 = model.h0_activation(X_batch)
            proj3_target_sum += torch.einsum('pqn,p->qn', a0, phi3_target) / P_total
            proj3_perp_sum += torch.einsum('pqn,p->qn', a0, phi3_perp) / P_total
            proj1_target_sum += torch.einsum('pqn,p->qn', a0, phi1_target) / P_total
            proj1_perp_sum += torch.einsum('pqn,p->qn', a0, phi1_perp) / P_total
            torch.cuda.empty_cache()
            del X_batch, x0, x_perp, phi3_target, phi3_perp, phi1_target, phi1_perp, a0
    proj3_target = proj3_target_sum
    proj3_perp = proj3_perp_sum
    proj1_target = proj1_target_sum
    proj1_perp = proj1_perp_sum
    eig_target1 = proj1_target.var().cpu().numpy().flatten()
    eig_perp1 = proj1_perp.var().cpu().numpy().flatten()
    eig_target3 = proj3_target.var().cpu().numpy().flatten()
    eig_perp3 = proj3_perp.var().cpu().numpy().flatten()
    d_minus_1 = d - 1
    eigenvalues = np.concatenate([
        eig_target1, np.tile(eig_perp1, d_minus_1),
        eig_target3, np.tile(eig_perp3, d**3 - 1)
    ]).flatten()
    stats = {
        "ens": int(ens),
        "n1": int(n1),
        "d": int(d),
        "P_total": int(P_total),
        "batch_size": int(batch_size),
        "perp_dim": int(perp_dim),
        "h3": {
            "target": {
                "mean": float(torch.mean(proj3_target).item()),
                "std": float(torch.std(proj3_target).item()),
                "var": float(proj3_target.var().item()),
                "second_moment": float(second_moment(proj3_target).item()),
            },
            "perp": {
                "mean": float(torch.mean(proj3_perp).item()),
                "std": float(torch.std(proj3_perp).item()),
                "var": float(proj3_perp.var().item()),
                "second_moment": float(second_moment(proj3_perp).item()),
            },
        },
        "h1": {
            "target": {
                "mean": float(torch.mean(proj1_target).item()),
                "std": float(torch.std(proj1_target).item()),
                "var": float(proj1_target.var().item()),
                "second_moment": float(second_moment(proj1_target).item()),
            },
            "perp": {
                "mean": float(torch.mean(proj1_perp).item()),
                "std": float(torch.std(proj1_perp).item()),
                "var": float(proj1_perp.var().item()),
                "second_moment": float(second_moment(proj1_perp).item()),
            },
        },
        "h3_eigenvalues": eigenvalues.tolist(),
        "h3_target_eigenvalues": eig_target3.tolist(),
        "h3_perp_eigenvalues": eig_perp3.tolist(),
    }
    return stats
