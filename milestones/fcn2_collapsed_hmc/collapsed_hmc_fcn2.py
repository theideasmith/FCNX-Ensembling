"""
Collapsed (exact Born-Oppenheimer) HMC for the FCN2 erf posterior.

The Langevin trainer (train_fcn2_erf_sigma_a.py) samples
    p(W0, A) ∝ exp(-Σ_μ (f_μ - y_μ)² / T - |W0|²/(2σ_W²) - |A|²/(2σ_A²)),
    f = Φ A,  Φ_{μi} = erf(w_i · x_μ),  σ_W² = s0/d,  σ_A² = sa0/(N χ).
A is the fast, stiff variable (curvature T/σ_A² + 2 λ_max(ΦᵀΦ)), W0 the slow one
(curvature T/σ_W²), so plain Langevin has condition number ~1e6 and needs ~1e7+ steps.

Given W0 the readout is exactly Gaussian, so it can be integrated out:
    U(W0) = |W0|²/(2σ_W²) + ½ yᵀ K⁻¹ y + ½ log det K,   K = σ_A² Φ Φᵀ + (T/2) I_P
(Woodbury N×N form when P > N). HMC runs on Z = W0/σ_W with exact Metropolis
correction (no step-size bias, so no lr schedule), plus exact per-neuron sign flips
w_i → -w_i (U is invariant since erf is odd). A | W0 is drawn exactly for outputs.
"""

import math
import time

import numpy as np
import torch


def _cholesky(A):
    """Cholesky without the host-side error check, so U and its gradient can be captured in a CUDA graph.
    A non-positive-definite matrix gives NaNs, which HMC rejects (dH = inf)."""
    return torch.linalg.cholesky_ex(A, check_errors=False)[0]


def _he3(vx):
    return (vx ** 3 - 3.0 * vx) / (6.0 ** 0.5)


DATA_DEVICE = "cuda:0"  # the 4090 on zohar-gpu-2, where the Langevin runs drew their data


def make_data(d, P, eps, he1, seed, device, data_device=DATA_DEVICE):
    """Same draws as train_fcn2_erf_sigma_a.py.

    CUDA randn output depends on the GPU model, so the data are always drawn on
    data_device (the device the Langevin runs used) and then moved to device.
    """
    if not torch.cuda.is_available():
        # No GPU: reproduce the 4090's CUDA randn stream on the CPU (cuda_randn_emulator; matches the stored
        # data_fingerprint to ~1e-5 relative, i.e. last-bit float32 rounding)
        from cuda_randn_emulator import cuda_randn
        X = torch.from_numpy(cuda_randn(P * d, seed).reshape(P, d))
        Y = he1 * X[:, 0] + eps * _he3(X[:, 0])
        Xinf = torch.from_numpy(cuda_randn(3000 * d, seed + 1).reshape(3000, d))
        Yinf = he1 * Xinf[:, 0] + eps * _he3(Xinf[:, 0])
        return tuple(t.to(device) for t in (X, Y, Xinf, Yinf))
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=data_device)
    Y = he1 * X[:, 0] + eps * _he3(X[:, 0])
    torch.manual_seed(seed + 1)
    Xinf = torch.randn(3000, d, device=data_device)
    Yinf = he1 * Xinf[:, 0] + eps * _he3(Xinf[:, 0])
    return tuple(t.to(device) for t in (X, Y, Xinf, Yinf))


def data_fingerprint(X):
    """Short checksum of the training inputs, to catch data mismatches between runs."""
    return f"{X.double().sum().item():.10e}/{(X.double() ** 2).sum().item():.10e}"


class CollapsedFCN2:
    def __init__(self, X, Y, N, T, s0, sa0, chi=1.0):
        self.X, self.Y = X, Y
        self.P, self.d = X.shape
        self.N = N
        self.T = T / chi
        self.sw = math.sqrt(s0 / self.d)
        self.sa2 = sa0 / (N * chi)
        self.use_woodbury = self.P > N

    def phi(self, Z):
        # Z: (C, N, d) -> Φ: (C, P, N)
        return torch.erf(torch.einsum("cnd,pd->cpn", Z * self.sw, self.X))

    def U(self, Z):
        """Collapsed negative log posterior per chain, shape (C,)."""
        Phi = self.phi(Z)
        y = self.Y.expand(Z.shape[0], -1)
        half_T = 0.5 * self.T
        if not self.use_woodbury:
            K = self.sa2 * Phi @ Phi.transpose(1, 2)
            K = K + half_T * torch.eye(self.P, dtype=K.dtype, device=K.device)
            L = _cholesky(K)
            v = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            quad = (y * v).sum(-1)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
        else:
            # K⁻¹ = (2/T)[I - (2/T) Φ M⁻¹ Φᵀ],  M = (2/T)ΦᵀΦ + I/σ_A²
            G = Phi.transpose(1, 2) @ Phi
            M = G / half_T + torch.eye(self.N, dtype=G.dtype, device=G.device) / self.sa2
            L = _cholesky(M)
            b = (Phi.transpose(1, 2) @ y.unsqueeze(-1)) / half_T
            u = torch.cholesky_solve(b, L)
            quad = (y * y).sum(-1) / half_T - (b * u).sum((-1, -2))
            logdet = (self.P * math.log(half_T) + self.N * math.log(self.sa2)
                      + 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1))
        return 0.5 * (Z * Z).sum((1, 2)) + 0.5 * quad + 0.5 * logdet

    def U_and_grad(self, Z):
        Z = Z.detach().requires_grad_(True)
        u = self.U(Z)
        (g,) = torch.autograd.grad(u.sum(), Z)
        return u.detach(), g.detach()

    @torch.no_grad()
    def sample_A(self, Z):
        """Exact draw of A | W0 (C, N)."""
        Phi = self.phi(Z)
        half_T = 0.5 * self.T
        G = Phi.transpose(1, 2) @ Phi
        M = G / half_T + torch.eye(self.N, dtype=G.dtype, device=G.device) / self.sa2
        L = torch.linalg.cholesky(M)
        b = (Phi.transpose(1, 2) @ self.Y.unsqueeze(-1).expand(Z.shape[0], -1, -1)) / half_T
        mean = torch.cholesky_solve(b, L).squeeze(-1)
        # Σ = M⁻¹ = L⁻ᵀ L⁻¹, so L⁻ᵀ ξ has covariance Σ
        xi = torch.randn_like(mean).unsqueeze(-1)
        noise = torch.linalg.solve_triangular(L.transpose(1, 2), xi, upper=True).squeeze(-1)
        return mean + noise


def hmc(model, Z0, n_iter, burn, n_leap, eps0, target_acc, flip_signs, callback=None, start_iter=0,
        step_init=None, pre_step=None):
    """HMC on the collapsed potential. callback(it, Z, u, info) is called after every iteration.

    pre_step(it, Z) may return (Z_new, stats) to replace the state before the HMC move
    (e.g. a Gibbs sweep); stats are passed to the callback as info["pre"].
    """
    C = Z0.shape[0]
    Z = Z0.clone()
    u, g = model.U_and_grad(Z)
    step = step_init.clone() if step_init is not None else torch.full((C,), eps0, dtype=Z.dtype, device=Z.device)
    log_step = torch.log(step)
    t0 = time.time()
    for it in range(start_iter, n_iter):
        pre = pre_step(it, Z) if pre_step is not None else None
        if pre is not None:
            Z = pre[0]
            u, g = model.U_and_grad(Z)
        if flip_signs:
            s = torch.randint(0, 2, (C, Z.shape[1], 1), device=Z.device, dtype=Z.dtype) * 2 - 1
            Z = Z * s
            g = g * s
        # Jittered trajectory length avoids resonances
        L = int(np.random.randint(max(1, n_leap // 2), n_leap + 1))
        e = step.view(C, 1, 1)
        p = torch.randn_like(Z)
        H0 = u + 0.5 * (p * p).sum((1, 2))
        Zn, gn = Z, g
        pn = p - 0.5 * e * gn
        for li in range(L):
            Zn = Zn + e * pn
            un, gn = model.U_and_grad(Zn)
            if li < L - 1:
                pn = pn - e * gn
        pn = pn - 0.5 * e * gn
        H1 = un + 0.5 * (pn * pn).sum((1, 2))
        dH = H1 - H0
        dH = torch.where(torch.isfinite(dH), dH, torch.full_like(dH, float("inf")))
        acc_prob = torch.exp(torch.clamp(-dH, max=0.0))
        accept = torch.rand_like(acc_prob) < acc_prob
        a = accept.view(C, 1, 1)
        Z = torch.where(a, Zn, Z)
        g = torch.where(a, gn, g)
        u = torch.where(accept, un, u)
        if it < burn:
            # Robbins-Monro step adaptation per chain
            log_step = log_step + (acc_prob - target_acc) * (2.0 / math.sqrt(it + 10))
            step = torch.exp(log_step)
        if callback is not None:
            callback(it, Z, u, {"acc": acc_prob, "step": step, "L": L, "wall": time.time() - t0,
                                "pre": pre[1] if pre is not None else None})
    return Z, step


@torch.no_grad()
def observables(model, Z, A, Xinf, Yinf):
    """Per-chain observables, each of shape (C,)."""
    W = Z * model.sw
    w0 = W[..., 0]
    wp2 = (W[..., 1:] ** 2).sum(-1) / max(model.d - 1, 1)
    f_tr = torch.einsum("cpn,cn->cp", model.phi(Z), A)
    f_inf = torch.einsum("cpn,cn->cp", torch.erf(torch.einsum("cnd,pd->cpn", W, Xinf)), A)
    he1 = Xinf[:, 0]
    he3 = _he3(Xinf[:, 0])
    f_res = f_inf - (f_inf * he1).mean(-1, keepdim=True) * he1
    return {
        "w0_sq": (w0 ** 2).mean(-1),
        "w0_abs": w0.abs().mean(-1),
        "w0_kurt": (w0 ** 4).mean(-1) / (w0 ** 2).mean(-1) ** 2,
        "frac_aligned": (w0.abs() > 2.5 * model.sw).to(w0.dtype).mean(-1),
        "wperp_sq": wp2.mean(-1),
        "A_sq_over_sa2": (A ** 2).mean(-1) / model.sa2,
        "train_loss": ((f_tr - model.Y) ** 2).sum(-1),
        "L3": (f_res * he3).mean(-1) / (Yinf * he3).mean(),
        "L1_he1": (f_inf * he1).mean(-1),
        "test_mse": ((f_inf - Yinf) ** 2).mean(-1),
    }
