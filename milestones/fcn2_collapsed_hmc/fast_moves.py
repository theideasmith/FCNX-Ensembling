"""
Faster moves for collapsed HMC on the FCN2 erf posterior.

GraphedUGrad
    U(Z) and ∇U(Z) captured once in a CUDA graph (static input/output buffers) and replayed for every
    leapfrog step. Same interface as CollapsedFCN2.U_and_grad; removes per-op launch overhead.

BlockGibbs
    Vectorized w0 moves for a random block S of neurons with an exact joint Metropolis-Hastings test:
      1. for every i ∈ S, the single-neuron conditional U_i(w0) given the current state of all other
         neurons (the Schur-complement form of neuron_gibbs.NeuronGibbs) is evaluated in one batch:
         the drive term r(w0) = (2/T) φ(w0)·e_i exactly on a fine grid, the self-reaction s(w0) on a
         coarse grid and interpolated (it is smooth);
      2. each w0_i' is drawn from its gridded conditional (cell ~ softmax, then uniform in the cell),
         giving an explicit forward density q(W→W');
      3. the same gridded conditionals at W' give the reverse density q(W'→W);
      4. accept the whole block with min(1, exp(U(W) - U(W')) q(W'→W)/q(W→W')), U the exact collapsed
         potential in float64.
    The proposal neglects interactions inside the block and is only approximately the conditional,
    but it is a fixed function of the state, so the MH test makes the move exact; the approximations
    only lower the acceptance rate. The block size adapts toward a target acceptance.

Status (2026-10-27). GraphedUGrad matches eager exactly but gives no speedup at d=150-300 (the gradient
is compute-bound: Cholesky and matmuls), so the runner does not use it. BlockGibbs is exact: on a
2-neuron problem one block move leaves exact posterior samples invariant, and on a correlated Gaussian
it recovers mean, variance and correlation. It can, however, stay in one joint mode when neurons are
strongly coupled, and at the d=150 ray point (N=1500) its chains relax more slowly than the sequential
sweep: block 128, 400 iterations ends at w0² 0.0370/0.0376 vs the converged 0.0379. Treat it as
experimental (--gibbs-block); the sequential NeuronGibbs sweep remains the validated default.

GraphedNeuronGibbs (--gibbs-graph) is the exact sequential sweep with each neuron update in a CUDA graph.
Most of its gain comes from keeping M⁻¹ in float32 (the 4090 runs float64 at 1/64 rate): 0.43 s per sweep
at d=150, N=1500 (original 2.05 s), M⁻¹ relative error 1e-6 after a sweep. Validated at the d=150 ray
point: prior and theory-sampled starts both reach w0² 0.0379, frac_aligned 0.040 (the converged values).
"""

import math

import torch


class GraphedUGrad:
    def __init__(self, model, Z_example, warmup=3):
        self.model = model
        self.Z = Z_example.detach().clone()
        self.Z.requires_grad_(True)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                u = model.U(self.Z)
                torch.autograd.grad(u.sum(), self.Z)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.u = model.U(self.Z)
            (self.g,) = torch.autograd.grad(self.u.sum(), self.Z)

    def __call__(self, Z):
        with torch.no_grad():
            self.Z.copy_(Z)
        self.graph.replay()
        return self.u.detach().clone(), self.g.detach().clone()


class BlockGibbs:
    def __init__(self, X, Y, N, T, s0, sa0, chi=1.0, wmax=2.4, n_fine=768, n_coarse=33, block=128,
                 target_acc=0.4, dtype=torch.float32):
        self.X, self.Y = X.to(dtype), Y.to(dtype)
        self.P, self.d = X.shape
        self.N = N
        self.T = T / chi
        self.g = 2.0 / self.T
        self.sw2 = s0 / self.d
        self.sa2 = sa0 / (N * chi)
        self.dtype = dtype
        dev = X.device
        self.edges = torch.linspace(-wmax, wmax, n_fine + 1, device=dev, dtype=dtype)
        self.cells = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.dw = (self.edges[1] - self.edges[0]).item()
        self.coarse = torch.linspace(-wmax, wmax, n_coarse, device=dev, dtype=dtype)
        self.block = block
        self.target_acc = target_acc
        self.adapt = True               # set False after burn-in: a fixed block size keeps the chain Markov
        self.max_elems = 2.5e8
        self.stats = {"proposed": 0, "accepted": 0, "changed": 0.0, "crossed": 0.0}

    # ------------------------------------------------------------------ exact potential (float64)
    def _factor(self, W):
        """Φ, M⁻¹ (float32 copy), z, residual e, and the exact collapsed U(W) (float64)."""
        Wd, Xd, Yd = W.double(), self.X.double(), self.Y.double()
        Phi = torch.erf(Xd @ Wd.T)
        M = self.g * (Phi.T @ Phi) + torch.eye(self.N, device=W.device, dtype=torch.float64) / self.sa2
        L = torch.linalg.cholesky(M)
        b = self.g * (Phi.T @ Yd)
        z = torch.cholesky_solve(b[:, None], L)[:, 0]
        quad = self.g * (Yd @ Yd) - b @ z
        logdet = (self.P * math.log(1 / self.g) + self.N * math.log(self.sa2)
                  + 2 * torch.log(torch.diagonal(L)).sum())
        U = 0.5 * (Wd * Wd).sum() / self.sw2 + 0.5 * quad + 0.5 * logdet
        Minv = torch.cholesky_inverse(L)
        e = Yd - Phi @ z
        dt = self.dtype
        return dict(Phi=Phi.to(dt), Minv=Minv.to(dt), z=z.to(dt), e=e.to(dt), U=U.item())

    # ------------------------------------------------------------------ batched conditionals
    def _log_cond(self, W, S, fac):
        """log of the gridded conditional (unnormalized -U_i) on the fine cells, shape (|S|, n_fine)."""
        Phi, Minv, z, e = fac["Phi"], fac["Minv"], fac["z"], fac["e"]
        X0 = self.X[:, 0]
        base = self.X[:, 1:] @ W[S, 1:].T                          # (P, s) perp preactivation
        dS = torch.diagonal(Minv)[S]                               # (s,)
        colS = Minv[:, S]                                          # (N, s)
        # cavity residuals e_i = e + Φ M⁻¹[:, i] z_i / M⁻¹_ii
        E = e[:, None] + (Phi @ colS) * (z[S] / dS)[None, :]       # (P, s)
        s_n = len(S)
        # drive on the fine grid: r = g φ(w0)·e_i, chunked over neurons to bound memory (P·chunk·F floats)
        chunk = max(1, int(self.max_elems // (self.P * len(self.cells))))
        r = torch.cat([self.g * torch.einsum("psf,ps->sf",
                                             torch.erf(X0[:, None, None] * self.cells[None, None, :]
                                                       + base[:, j:j + chunk, None]), E[:, j:j + chunk])
                       for j in range(0, s_n, chunk)])
        # self-reaction on the coarse grid: s = m - [cᵀ M⁻¹ c - (colᵀ c)²/M⁻¹_ii], c = g Φ_{-i}ᵀ φ
        Fc = torch.erf(X0[:, None, None] * self.coarse[None, None, :] + base[:, :, None])  # (P, s, C)
        nC = Fc.shape[2]
        C = self.g * (Phi.T @ Fc.reshape(self.P, -1))                                      # (N, s*C)
        rows = torch.as_tensor(S, device=W.device).repeat_interleave(nC)
        C[rows, torch.arange(s_n * nC, device=W.device)] = 0.0
        MC = Minv @ C
        quad = (C * MC).sum(0) - (colS.repeat_interleave(nC, 1) * C).sum(0) ** 2 / dS.repeat_interleave(nC)
        m = self.g * (Fc * Fc).sum(0).reshape(-1) + 1.0 / self.sa2
        sc = (m - quad).reshape(s_n, nC).clamp_min(1e-6 / self.sa2)
        # linear interpolation of log s onto the fine cells
        pos = (self.cells - self.coarse[0]) / (self.coarse[1] - self.coarse[0])
        lo = pos.floor().long().clamp(0, nC - 2)
        t = (pos - lo).clamp(0, 1)
        logs = torch.log(sc)
        logs_f = logs[:, lo] * (1 - t) + logs[:, lo + 1] * t
        s_f = torch.exp(logs_f)
        prior = 0.5 * self.cells ** 2 / self.sw2
        return -(prior[None, :] + 0.5 * logs_f - 0.5 * r * r / s_f)

    def _log_q(self, logc, w0):
        """log density of w0 (|S|,) under the piecewise-constant proposal built from logc; -inf outside."""
        logp = torch.log_softmax(logc.double(), 1)
        k = torch.bucketize(w0.to(self.dtype), self.edges) - 1
        inside = (k >= 0) & (k < len(self.cells))
        lq = logp.gather(1, k.clamp(0, len(self.cells) - 1)[:, None])[:, 0] - math.log(self.dw)
        return torch.where(inside, lq, torch.full_like(lq, -math.inf)), logp

    # ------------------------------------------------------------------ one sweep
    @torch.no_grad()
    def sweep(self, W):
        """Block MH updates covering every neuron once (random blocks). W: (N, d); returns new W."""
        W = W.clone()
        order = torch.randperm(self.N, device=W.device)
        fac = self._factor(W)
        thr = 2.5 * math.sqrt(self.sw2)
        n_changed = n_crossed = 0
        i0 = 0
        while i0 < self.N:
            S = order[i0:i0 + self.block]
            i0 += len(S)
            logc = self._log_cond(W, S, fac)
            logp = torch.log_softmax(logc.double(), 1)
            k = torch.multinomial(torch.exp(logp), 1)[:, 0]
            w_new = self.edges[k] + torch.rand(len(S), device=W.device, dtype=self.dtype) * self.dw
            lq_f = (logp.gather(1, k[:, None])[:, 0] - math.log(self.dw)).sum()
            W2 = W.clone()
            W2[S, 0] = w_new
            fac2 = self._factor(W2)
            lq_r, _ = self._log_q(self._log_cond(W2, S, fac2), W[S, 0])
            log_a = (fac["U"] - fac2["U"]) + lq_r.sum().item() - lq_f.item()
            self.stats["proposed"] += 1
            accept = math.log(torch.rand(()).item() + 1e-300) < log_a
            if accept:
                self.stats["accepted"] += 1
                n_changed += len(S)
                n_crossed += ((W[S, 0].abs() > thr) != (w_new.abs() > thr)).sum().item()
                W, fac = W2, fac2
            if self.adapt:   # block-size adaptation toward the target acceptance (burn-in only)
                a = min(1.0, math.exp(min(log_a, 0.0)))
                self.block = int(min(self.N, max(8, round(self.block * math.exp(0.5 * (a - self.target_acc))))))
        self.stats["changed"] = n_changed / self.N
        self.stats["crossed"] = n_crossed / self.N
        return W


class GraphedNeuronGibbs:
    """
    The exact sequential w0 sweep of neuron_gibbs.NeuronGibbs (iterated SIR over B candidates plus the
    current value, Schur-complement conditional, rank-2 update of M⁻¹) with each neuron update captured
    once in a CUDA graph and replayed N times per sweep. Differences from NeuronGibbs, both exact:
      * selection by Gumbel-max (argmax of log-weight + Gumbel noise), i.e. the same categorical draw;
      * the update is always applied: picking the current value re-inserts the same column, which leaves
        M⁻¹, b and z unchanged algebraically.
    The neuron index comes from a device-side permutation and pointer, so a sweep has no host syncs.
    Same sweep(W, B, mode) interface; mode must be 'w0'.
    """

    def __init__(self, X, Y, N, T, s0, sa0, chi=1.0, omega=0.6, s_perp=None, B=64, inv_dtype=torch.float64):
        self.inv_dtype = inv_dtype
        self.X, self.Y = X, Y
        self.P, self.d = X.shape
        self.N, self.B = N, B
        self.T = T / chi
        self.g = 2.0 / self.T
        self.sw2 = s0 / self.d
        self.sa2 = sa0 / (N * chi)
        self.omega = omega
        self.graph = None

    def _rebuild(self, W):
        Phi = torch.erf(self.X @ W.T)
        G = (Phi.T @ Phi).double()
        M = self.g * G + torch.eye(self.N, dtype=torch.float64, device=W.device) / self.sa2
        Minv = torch.cholesky_inverse(torch.linalg.cholesky(M))
        b = self.g * (Phi.T @ self.Y).double()
        dt = self.inv_dtype
        return Phi, Minv.to(dt), b.to(dt), (Minv @ b).to(dt)

    def _alloc(self, W):
        dev = W.device
        self.W = W.clone()
        self.Phi, self.Minv, self.b, self.zf = self._rebuild(W)
        self.order = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ptr = torch.zeros(1, dtype=torch.long, device=dev)
        self.n_changed = torch.zeros((), dtype=torch.float32, device=dev)
        self.n_crossed = torch.zeros((), dtype=torch.float32, device=dev)
        self.X0, self.Xp = self.X[:, 0], self.X[:, 1:]
        self.Yd = self.Y.double()
        self.thr = 2.5 * math.sqrt(self.sw2)

    def _step(self):
        """One neuron update on the static buffers (graph-capturable: no host syncs, static shapes)."""
        g, B = self.g, self.B
        idx = self.order.index_select(0, self.ptr)                       # (1,)
        self.ptr.add_(1)
        wi = self.W.index_select(0, idx)[0]                              # (d,)
        w0c = torch.randn(B, device=wi.device, dtype=wi.dtype) * self.omega
        w0all = torch.cat([wi[:1], w0c])                                 # (B+1,)
        base = self.Xp @ wi[1:]
        Fc = torch.erf(self.X0[:, None] * w0all[None, :] + base[:, None])    # (P, B+1)
        dt = self.inv_dtype
        prior = 0.5 * w0all.to(dt) ** 2 / self.sw2
        logq = -0.5 * w0all.to(dt) ** 2 / self.omega ** 2
        Minv_ii = self.Minv.index_select(0, idx).index_select(1, idx)[0, 0]
        col = self.Minv.index_select(1, idx)[:, 0].index_fill(0, idx, 0.0)
        z_i = self.zf.index_select(0, idx)[0]
        zp = (self.zf - col * (z_i / Minv_ii)).index_fill(0, idx, 0.0)
        C = (g * (self.Phi.T @ Fc)).to(dt).index_fill(0, idx, 0.0)       # (N, B+1)
        MC = self.Minv @ C
        colC = col @ C
        s = g * (Fc * Fc).sum(0).to(dt) + 1.0 / self.sa2 - ((C * MC).sum(0) - colC ** 2 / Minv_ii)
        beta = g * (Fc.T @ self.Y).to(dt)
        r = beta - zp @ C
        logw = -(prior + 0.5 * torch.log(s) - 0.5 * r * r / s) - logq
        gumbel = -torch.log(-torch.log(torch.rand_like(logw).clamp_min(1e-300)))
        j = torch.argmax(logw + gumbel).reshape(1)                       # (1,) on device
        # rank-2 update (re-inserting the current value when j == 0 leaves everything unchanged)
        sj = s.index_select(0, j)[0]
        u = (MC.index_select(1, j)[:, 0] - col * (colC.index_select(0, j)[0] / Minv_ii)).index_fill(0, idx, 0.0)
        Mn = self.Minv - torch.outer(col, col) / Minv_ii + torch.outer(u, u) / sj
        Mn.index_copy_(1, idx, (-u / sj)[:, None])
        Mn.index_copy_(0, idx, (-u / sj)[None, :])
        Mn.index_put_((idx, idx), (1.0 / sj).reshape(1))
        self.Minv.copy_(Mn)
        w_new = w0all.index_select(0, j)
        self.n_changed.add_((j != 0).float().sum())
        self.n_crossed.add_(((wi[0].abs() > self.thr) != (w_new[0].abs() > self.thr)).float())
        self.W.index_put_((idx, torch.zeros_like(idx)), w_new)
        self.Phi.index_copy_(1, idx, Fc.index_select(1, j))
        self.b.index_copy_(0, idx, beta.index_select(0, j))
        self.zf.copy_(self.Minv @ self.b)

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = [t.clone() for t in (self.W, self.Phi, self.Minv, self.b, self.zf)]
        with torch.cuda.stream(s):
            for _ in range(2):
                self.ptr.zero_()
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        for t, v in zip((self.W, self.Phi, self.Minv, self.b, self.zf), saved):
            t.copy_(v)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    @torch.no_grad()
    def sweep(self, W, B=None, mode="w0", check=False):
        if mode != "w0":
            raise ValueError("GraphedNeuronGibbs supports mode='w0' only")
        if self.graph is None:
            self._alloc(W)
            self._capture()
        self.W.copy_(W)
        Phi, Minv, b, zf = self._rebuild(self.W)
        self.Phi.copy_(Phi); self.Minv.copy_(Minv); self.b.copy_(b); self.zf.copy_(zf)
        self.order.copy_(torch.randperm(self.N, device=W.device))
        self.ptr.zero_(); self.n_changed.zero_(); self.n_crossed.zero_()
        for _ in range(self.N):
            self.graph.replay()
        stats = {"changed": self.n_changed.item() / self.N, "crossed": self.n_crossed.item() / self.N}
        if check:
            _, Minv_ref, _, zf_ref = self._rebuild(self.W)
            stats["minv_rel_err"] = ((self.Minv - Minv_ref).norm() / Minv_ref.norm()).item()
            stats["z_rel_err"] = ((self.zf - zf_ref).norm() / zf_ref.norm()).item()
        return self.W.clone(), stats
