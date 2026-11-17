"""
Fast GPU path for the exact FCN3 posterior sampler (fcn3_sampler.FCN3). Same target and the same moves as the
reference; the speed comes from three changes, none of which touches exactness:

1. HMC gradient in float32 (FastFCN3.hmc_step).
   A leapfrog step built from ANY deterministic force field F(q) (kick p += eps F(q), drift q += eps p) is a
   composition of shears: volume preserving and, with the momentum flip, time reversible. HMC is therefore exact
   for an approximate gradient provided the Metropolis test uses the exact energy. The leapfrog uses a float32
   gradient (float32 Cholesky and inverse; the reference does them in float64, which a GeForce card runs at 1/64
   rate and which was ~90% of the HMC time) and the energy at both ends of the trajectory is the reference
   float64 collapsed potential. A float32 failure anywhere on the trajectory rejects it (the rule is symmetric
   under reversal). The energy at intermediate leapfrog points is not computed at all.
   The accepted state's float64 factorisation is cached and reused by draw_a.

2. Jump sweeps with no host synchronisation, captured in CUDA graphs (FastSweep).
   One graph per layer holds one neuron update; it is replayed N times per sweep. The neuron index comes from a
   device-side counter, all random numbers of a sweep are drawn up front from the sampler's generator, the
   categorical draw over candidates is a Gumbel argmax (same distribution as torch.multinomial on the softmax),
   and "accept candidate i" is applied unconditionally with torch.where (candidate 0 is the current state).

3. Fused Triton kernels for the read-in move (the O(P N2) part):
   - pass 1: h0 = H - F1[:, j] B[:, j]^T, writes Dm = a * erf'(h0) and the row sums sum_k a_k erf(h0),
     sum_k a_k erf(H) in one read of H;
   - pass 2: sum_k a_k erf(h0 + phi_c B_kc) for all new candidates c without materialising the
     (candidates x P x N2) tensor (the reference writes, erf's and re-reads ~0.3 GB per neuron at N=1500).
   Candidate 0 (the current state) needs no pass 2: its prediction is the cached f.

The read-in proposal mean m(w) (CG) and the energies use the same precision as the reference (float32 data
terms, float64 accumulation of the energies). Sweeps support the "direct" energy only (series is not needed:
the fused kernel is cheaper than the series at every candidate count).
"""

import math

import torch
import triton
import triton.language as tl

from fcn3_sampler import FCN3, SQPI, F64, readin_coeffs


# ----------------------------------------------------------------------------------------------------------
# Triton kernels for the read-in move
# ----------------------------------------------------------------------------------------------------------
@triton.jit
def _readin_pass1(H, f1j, bj, a, Dm, d0, e0a, P, N2, sH, sD, KCH, BP: tl.constexpr, BK: tl.constexpr):
    """Rows block x k-split: Dm = a erf'(h0), partial row sums d0 = sum a (erf(H) - erf(h0)), e0a = sum a erf(h0)
    over k in [ks*KCH, (ks+1)*KCH), written to d0[ks, p], e0a[ks, p]."""
    pid = tl.program_id(0)
    ksp = tl.program_id(1)
    rows = pid * BP + tl.arange(0, BP)
    rm = rows < P
    f1 = tl.load(f1j + rows, mask=rm, other=0.0)
    acc_d = tl.zeros((BP,), tl.float32)
    acc_e = tl.zeros((BP,), tl.float32)
    kend = tl.minimum(ksp * KCH + KCH, N2)
    for k0 in range(ksp * KCH, kend, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < kend
        m2 = rm[:, None] & km[None, :]
        h = tl.load(H + rows[:, None] * sH + ks[None, :], mask=m2, other=0.0)
        b = tl.load(bj + ks, mask=km, other=0.0)
        av = tl.load(a + ks, mask=km, other=0.0)
        h0 = h - f1[:, None] * b[None, :]
        e0 = tl.math.erf(h0)
        acc_e += tl.sum(e0 * av[None, :], axis=1)
        acc_d += tl.sum((tl.math.erf(h) - e0) * av[None, :], axis=1)
        g = 1.1283791670955126 * tl.exp(-h0 * h0)
        tl.store(Dm + rows[:, None] * sD + ks[None, :], g * av[None, :], mask=m2)
    tl.store(d0 + ksp * P + rows, acc_d, mask=rm)
    tl.store(e0a + ksp * P + rows, acc_e, mask=rm)


@triton.jit
def _readin_pass2(H, f1j, bj, a, phi, BcT, out, P, N2, C, sH, sphp, sphc, sBc, KCH,
                  BP: tl.constexpr, CB: tl.constexpr, BK: tl.constexpr):
    """out[ks, c, p] = sum_{k in split ks} a_k erf(H[p,k] - f1j[p] bj[k] + phi[p, c] BcT[c, k]).
    phi[p, c] at phi + p*sphp + c*sphc. Grid: (row blocks, candidate blocks of CB, k-splits of KCH)."""
    pid = tl.program_id(0)
    cid = tl.program_id(1)
    ksp = tl.program_id(2)
    rows = pid * BP + tl.arange(0, BP)
    rm = rows < P
    cs = cid * CB + tl.arange(0, CB)
    cm = cs < C
    f1 = tl.load(f1j + rows, mask=rm, other=0.0)
    ph = tl.load(phi + rows[:, None] * sphp + cs[None, :] * sphc, mask=rm[:, None] & cm[None, :], other=0.0)
    acc = tl.zeros((BP, CB), tl.float32)
    kend = tl.minimum(ksp * KCH + KCH, N2)
    for k0 in range(ksp * KCH, kend, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < kend
        h = tl.load(H + rows[:, None] * sH + ks[None, :], mask=rm[:, None] & km[None, :], other=0.0)
        b = tl.load(bj + ks, mask=km, other=0.0)
        av = tl.load(a + ks, mask=km, other=0.0)
        h0 = h - f1[:, None] * b[None, :]                                                            # (BP, BK)
        bc = tl.load(BcT + cs[:, None] * sBc + ks[None, :], mask=cm[:, None] & km[None, :], other=0.0)  # (CB, BK)
        x = h0[:, None, :] + ph[:, :, None] * bc[None, :, :]                                        # (BP, CB, BK)
        acc += tl.sum(tl.math.erf(x) * av[None, None, :], axis=2)
    tl.store(out + ksp * C * P + cs[None, :] * P + rows[:, None], acc, mask=rm[:, None] & cm[None, :])


@triton.jit
def _rank2_update(H, phT, Bc, cptr, f1j, bj, P, N2, sH, sph, sbc, BP: tl.constexpr, BK: tl.constexpr):
    """H += phT[c] Bc[c]^T - f1j bj^T for the selected candidate c (no-op, and no memory traffic, when c == 0)."""
    c = tl.load(cptr)
    if c != 0:
        rows = tl.program_id(0) * BP + tl.arange(0, BP)
        cols = tl.program_id(1) * BK + tl.arange(0, BK)
        rm = rows < P
        cm = cols < N2
        m2 = rm[:, None] & cm[None, :]
        ph = tl.load(phT + c * sph + rows, mask=rm, other=0.0)
        f1 = tl.load(f1j + rows, mask=rm, other=0.0)
        b = tl.load(Bc + c * sbc + cols, mask=cm, other=0.0)
        b0 = tl.load(bj + cols, mask=cm, other=0.0)
        h = tl.load(H + rows[:, None] * sH + cols[None, :], mask=m2, other=0.0)
        tl.store(H + rows[:, None] * sH + cols[None, :], h + ph[:, None] * b[None, :] - f1[:, None] * b0[None, :],
                 mask=m2)


def _rank2(H, phT, Bc, c, f1j, bj):
    P, N2 = H.shape
    BP, BK = 32, 128
    _rank2_update[(triton.cdiv(P, BP), triton.cdiv(N2, BK))](H, phT, Bc, c, f1j, bj, P, N2, H.stride(0),
                                                            phT.stride(0), Bc.stride(0), BP=BP, BK=BK, num_warps=4)


def _rsum(x, L=128):
    """Row sums of a (R, n) tensor as a two-stage reduction ((R, n/L) blocks first): few long rows otherwise give
    one thread block per row."""
    R, n = x.shape
    G = -(-n // L)
    if G * L != n:
        x = torch.nn.functional.pad(x, (0, G * L - n))
    return x.view(R, G, L).sum(2).sum(1)


def _rlse(x, L=128):
    """Row logsumexp of (R, n), two-stage like _rsum."""
    R, n = x.shape
    G = -(-n // L)
    if G * L != n:
        x = torch.nn.functional.pad(x, (0, G * L - n), value=-math.inf)
    return torch.logsumexp(torch.logsumexp(x.view(R, G, L), 2), 1)


PASS1_CFG = dict(BP=8, BK=64, KS=8, num_warps=4)       # tuned on an RTX 5090 at P=3000, N2=1500
PASS2_CFG = dict(BP=8, CB=8, BK=32, KS=4, num_warps=4)


def _pass1(H, f1j, bj, a, Dm, d0p, e0ap, cfg=None):
    """d0p, e0ap: (KS, P) partial sums (sum over dim 0 afterwards)."""
    c = cfg or PASS1_CFG
    P, N2 = H.shape
    KS = d0p.shape[0]
    KCH = triton.cdiv(triton.cdiv(N2, KS), c["BK"]) * c["BK"]
    _readin_pass1[(triton.cdiv(P, c["BP"]), KS)](H, f1j, bj, a, Dm, d0p, e0ap, P, N2, H.stride(0), Dm.stride(0),
                                                KCH, BP=c["BP"], BK=c["BK"], num_warps=c["num_warps"])


def _pass2(H, f1j, bj, a, phi, BcT, outp, cfg=None):
    """phi: (P, C) (any strides); BcT: (C, N2) contiguous; outp: (KS, C, P) partial sums."""
    c = cfg or PASS2_CFG
    P, N2 = H.shape
    KS, C = outp.shape[0], outp.shape[1]
    KCH = triton.cdiv(triton.cdiv(N2, KS), c["BK"]) * c["BK"]
    grid = (triton.cdiv(P, c["BP"]), triton.cdiv(C, c["CB"]), KS)
    _readin_pass2[grid](H, f1j, bj, a, phi, BcT, outp, P, N2, C, H.stride(0), phi.stride(0), phi.stride(1),
                        BcT.stride(0), KCH, BP=c["BP"], CB=c["CB"], BK=c["BK"], num_warps=c["num_warps"])


# ----------------------------------------------------------------------------------------------------------
# graph-safe population proposal (same law as fcn3_sampler.PopProposal)
# ----------------------------------------------------------------------------------------------------------
def _prop_sample(pp, pop, k, comp, z, idx, flip):
    """PopProposal.sample_ex with pre-drawn randomness: comp (n,) component, z (n, m) normals, idx (n,) uniform in
    [0, npop-2] (shifted past the excluded row k), flip (n,) bool sign flip of the KDE draw."""
    idx = idx + (idx >= k).long()
    s = torch.where(flip[:, None] & pp.sym[None, :], -1.0, 1.0).to(F64)
    kde = s * pop.index_select(0, idx) + pp.bw * z
    c = comp[:, None]
    return torch.where(c == 0, kde, torch.where(c == 1, pp.mu0 + pp.sd0 * z, pp.mu0 + pp.sdb * z))


def _prop_consts(pp):
    """Constants of PopProposal.logpdf: 1/sd and sum(log sd) + m/2 log 2pi per component, log mixture weights."""
    c = {}
    for name, sd in (("kde", pp.bw), ("pri", pp.sd0), ("brd", pp.sdb)):
        c[name] = (1.0 / sd, float(torch.log(sd).sum()) + 0.5 * pp.m * math.log(2 * math.pi))
    c["logw"] = tuple(math.log(float(w) + 1e-300) for w in pp.w)
    c["flip"] = torch.where(pp.sym, -1.0, 1.0).to(F64)
    return c


def _prop_logpdf_ex(pp, X, pop, k, c):
    """PopProposal.logpdf_ex(X, pop, k) without host syncs (k a 1-element device tensor, c = _prop_consts(pp))."""
    lg = lambda D, isd, lz: -0.5 * ((D * isd) ** 2).sum(-1) - lz
    isd, lz = c["kde"]
    k1 = lg(X[:, None, :] - pop[None], isd, lz).index_fill_(1, k, -math.inf)
    k2 = lg(X[:, None, :] - pop[None] * c["flip"], isd, lz).index_fill_(1, k, -math.inf)
    lkde = _rlse(torch.cat([k1, k2], 1)) - math.log(2 * (pop.shape[0] - 1))
    lw = c["logw"]
    terms = torch.stack([lkde + lw[0], lg(X - pp.mu0, *c["pri"]) + lw[1], lg(X - pp.mu0, *c["brd"]) + lw[2]])
    return torch.logsumexp(terms, 0)


def _gumbel(shape, gen, dev):
    u = torch.rand(shape, dtype=F64, device=dev, generator=gen)
    return -torch.log(-torch.log(u.clamp_min(1e-300)))


# ----------------------------------------------------------------------------------------------------------
# graphed jump sweeps
# ----------------------------------------------------------------------------------------------------------
class FastSweep:
    """Exact per-neuron jump sweeps of FCN3._middle_sweep / FCN3._readin_sweep, one CUDA-graph replay per neuron.
    Candidate-major layouts ((candidates, P) rather than (P, candidates)) keep every reduction contiguous. With
    compile=True the per-neuron bodies go through torch.compile, which fuses the ~100 small ops of each into a
    handful of kernels, before graph capture."""

    def __init__(self, m: FCN3, use_graph=True, compile=True):
        assert m.dev.type == "cuda", "FastSweep needs a CUDA device"
        self.m = m
        self.use_graph = use_graph
        P, d, N1, N2, nc, S = m.P, m.d, m.N1, m.N2, m.n_cand, len(m.ks)
        dev, dt = m.dev, m.dt
        self.dev, self.dt = dev, dt
        z = lambda *s, t=dt: torch.zeros(*s, dtype=t, device=dev)
        lz = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        # network state
        self.W, self.B, self.a = z(N1, d), z(N2, N1), z(N2)
        self.F1, self.H, self.E, self.f = z(P, N1), z(P, N2), z(P, N2), z(P)
        self.ctr = lz(1)
        self.order = lz(max(N1, N2))
        # middle sweep state
        self.C, self.LP, self.Pc = z(N1, S, t=F64), z(S, N1, t=F64), z(N1, S, t=F64)
        self.Bd, self.Nperp, self.popb = z(N2, N1, t=F64), z(N2, N1, t=F64), z(N2, S + 1, t=F64)
        self.mid_rnd = dict(comp=lz(N2, nc), z=z(N2, nc, S + 1, t=F64), idx=lz(N2, nc),
                            flip=torch.zeros(N2, nc, dtype=torch.bool, device=dev), u=z(N2, t=F64),
                            xi=z(N2, nc, N1, t=F64), gum=z(N2, nc + 1, t=F64), na=z(N2, t=F64))
        self.mid_stats = z(2, t=F64)
        # read-in sweep state
        self.popw = z(N1, 2, t=F64)
        self.Dm = z(P, N2, t=torch.bfloat16)        # a erf'(h0): enters only the proposal mean (field, CG)
        self.f1j, self.bj = z(P), z(N2)
        self.d0p, self.e0ap = z(PASS1_CFG["KS"], P), z(PASS1_CFG["KS"], P)
        self.S2p = z(PASS2_CFG["KS"], nc, P)
        self.in_rnd = dict(comp=lz(N1, nc), z=z(N1, nc, 2, t=F64), idx=lz(N1, nc),
                           flip=torch.zeros(N1, nc, dtype=torch.bool, device=dev), u=z(N1, t=F64),
                           nn=z(N1, nc, d - 1), bn=z(N1, nc, N2), gum=z(N1, nc + 1, t=F64))
        self.in_stats = z(2, t=F64)
        # optional spring (k/2)(r_j^2 - c)^2 on one read-in neuron j (umbrella_count.py); k = 0: none
        self.spring = torch.tensor([-1.0, 0.0, 0.0], dtype=F64, device=dev)        # (j, c, k)
        # optional coupling lam of read-in neuron j (lam_count.py): its feature erf(w_j x) enters the network as
        # lam * erf(w_j x); while lam < 1 the read-in sweep leaves neuron j alone. j = -1: none
        self.lam = torch.tensor([-1.0, 1.0], dtype=F64, device=dev)                # (j, lam)
        # optional confinement of neuron j's outgoing column (lam_count.py): zeta_j = s_j B[:, j].u0 / (|B[:, j]| |u0|)
        # kept on one side of thr (side +1: zeta > thr, -1: zeta <= thr, 0: off); Sbu, Sbb track s_j B_j.u0, |B_j|^2
        # side +2: soft wall instead, a penalty k/2 (thr - zeta)^2 for zeta < thr added to the candidates' energies
        self.conf = torch.tensor([-1.0, 0.0, 0.0, 1.0, 0.0], dtype=F64, device=dev)  # (j, side, thr, s_j, k)
        # team pattern = current u = B @ sO, sO_i = sign(w0_i) for the other outer neurons (0 elsewhere)
        self.sO = z(N1, t=F64)
        self.Sbu, self.Sbb, self.Suu = z(1, t=F64), z(1, t=F64), z(1, t=F64)
        self.X0 = m.X[:, 0].contiguous()
        self.XpT = m.X[:, 1:].T.contiguous()
        self.pc_b, self.pc_w = _prop_consts(m.prop_b), _prop_consts(m.prop_w)
        self.graphs = {}
        self.bodies = {"middle": self._middle_step, "readin": self._readin_step}
        if compile:
            # each update is compiled as two graphs (select / apply) so that the candidate selection is
            # materialised once instead of being recomputed inside the fused update kernels
            cc = lambda f: torch.compile(f, fullgraph=True, dynamic=False)
            self._mid_select, self._mid_update = cc(self._mid_select), cc(self._mid_update)
            self._in_select, self._in_update = cc(self._in_select), cc(self._in_update)

    # ------------------------------------------------------------------ random numbers of one sweep (eager)
    def _draw(self, rnd, pp, n):
        g, dev = self.m.gen, self.dev
        rnd["comp"].copy_(torch.multinomial(pp.w_t, rnd["comp"].numel(), replacement=True, generator=g)
                          .view_as(rnd["comp"]))
        rnd["z"].normal_(generator=g)
        rnd["idx"].random_(0, n - 1, generator=g)
        rnd["flip"].copy_(torch.randint(0, 2, rnd["flip"].shape, device=dev, generator=g).bool())
        rnd["u"].uniform_(generator=g)
        rnd["gum"].copy_(_gumbel(rnd["gum"].shape, g, dev))
        for key in ("xi", "nn", "bn", "na"):
            if key in rnd:
                rnd[key].normal_(generator=g)

    # ------------------------------------------------------------------ one middle-neuron update (graph body)
    def _middle_step(self):
        self._mid_update(*self._mid_select())

    def _mid_select(self):
        m, S, N1, sb, v, nc = self.m, len(self.m.ks), self.m.N1, self.m.sb, self.m.v, self.m.n_cand
        i = self.ctr
        k = self.order.index_select(0, i)
        R = {key: t.index_select(0, i)[0] for key, t in self.mid_rnd.items()}
        pop = self.popb
        cand = torch.cat([pop.index_select(0, k), _prop_sample(m.prop_b, pop, k, R["comp"], R["z"], R["idx"],
                                                               R["flip"])])                # (nc+1, S+1)
        logq = _prop_logpdf_ex(m.prop_b, cand, pop, k, self.pc_b)
        zc, tc = cand[:, :S], cand[:, S]
        nperp = self.Nperp.index_select(0, k)                                               # (1, N1) unit
        redraw = R["u"] < m.redraw_frac
        ncand = torch.cat([nperp, torch.where(redraw, R["xi"], nperp)])                     # (nc+1, N1) unit, perp
        bd_k = self.Bd.index_select(0, k)
        bnew_all = (zc[1:, :, None] * self.LP[None]).sum(1) + torch.exp(0.5 * tc[1:])[:, None] * ncand[1:]
        bc = torch.cat([bd_k, bnew_all])                                                    # (nc+1, N1) f64
        hcT = (self.F1 @ bc.to(self.dt).T).T                                                # (nc+1, P)
        phT = torch.erf(hcT)
        ak_old = self.a.index_select(0, k)
        Ek = self.E.index_select(1, k)[:, 0]
        fmk = self.f - ak_old * Ek
        rho = (m.Y - fmk).double()
        ph = phT.double()
        A = 1.0 / m.sa2 + _rsum(ph * ph) / v
        bb = _rsum(ph * rho[None]) / v
        Udata = -0.5 * bb ** 2 / A + 0.5 * torch.log(A)
        Uprior = 0.5 * (zc ** 2).sum(1) + 0.5 * torch.exp(tc) / sb ** 2 - 0.5 * (N1 - S) * tc
        lw = -(Udata + Uprior) - logq
        jc = self.conf[0].clamp_min(0).long()
        bj = bc.index_select(1, jc)[:, 0]                                                   # candidates' B_kj
        ukc = bc @ self.sO                                                                  # candidates' u_k
        sbu = self.Sbu[0] + self.conf[3] * (bj * ukc - bj[0] * ukc[0])
        sbb = self.Sbb[0] + bj * bj - bj[0] * bj[0]
        suu = self.Suu[0] + ukc * ukc - ukc[0] * ukc[0]
        zeta = sbu / torch.sqrt((sbb * suu).clamp_min(1e-300))
        out = (self.conf[1] * (zeta - self.conf[2]) <= 0) & (self.conf[1] == 1) | \
              (self.conf[1] * (zeta - self.conf[2]) < 0) & (self.conf[1] < 0)
        lw = torch.where(out & (torch.arange(lw.shape[0], device=lw.device) > 0), -math.inf, lw)
        vwall = 0.5 * self.conf[4] * torch.clamp(self.conf[2] - zeta, min=0.0) ** 2
        lw = lw - torch.where(self.conf[1] == 2, vwall, 0.0 * vwall)
        lw = torch.where(torch.isnan(lw), -math.inf, lw)
        j = torch.argmax(lw + R["gum"]).view(1)
        return k, j, cand, ncand, bc, hcT, phT, A, bb, Ek, fmk, bd_k, nperp, R["na"]

    def _mid_update(self, k, j, cand, ncand, bc, hcT, phT, A, bb, Ek, fmk, bd_k, nperp, na):
        S, pop = len(self.m.ks), self.popb
        zc, tc = cand[:, :S], cand[:, S]
        Aj, bbj = A.index_select(0, j), bb.index_select(0, j)
        ak = (bbj / Aj + na / torch.sqrt(Aj)).to(self.dt)                              # (1,)
        moved = j != 0
        bnew = bc.index_select(0, j)                                                        # (1, N1) f64
        jc = self.conf[0].clamp_min(0).long()
        bo, bn = bd_k.index_select(1, jc)[0, 0], bnew.index_select(1, jc)[0, 0]
        uo, un = (bd_k @ self.sO)[0], (bnew @ self.sO)[0]
        zero = 0.0 * uo
        self.Sbu.add_(torch.where(moved, self.conf[3] * (bn * un - bo * uo), zero))
        self.Sbb.add_(torch.where(moved, bn * bn - bo * bo, zero))
        self.Suu.add_(torch.where(moved, un * un - uo * uo, zero))
        self.B.index_copy_(0, k, torch.where(moved, bnew.to(self.dt), self.B.index_select(0, k)))
        self.Bd.index_copy_(0, k, torch.where(moved, bnew, bd_k))
        hnew = torch.where(moved, hcT.index_select(0, j)[0], self.H.index_select(1, k)[:, 0])
        enew = torch.where(moved, phT.index_select(0, j)[0], Ek)
        self.H.index_copy_(1, k, hnew[:, None])
        self.E.index_copy_(1, k, enew[:, None])
        pop.index_copy_(0, k, torch.where(moved, cand.index_select(0, j), pop.index_select(0, k)))
        self.Nperp.index_copy_(0, k, torch.where(moved, ncand.index_select(0, j), nperp))
        self.a.index_copy_(0, k, ak)
        self.f.copy_(fmk + ak * enew)
        out0 = zc[0].abs().max() > 5
        outj = zc.index_select(0, j).abs().max() > 5
        self.mid_stats.add_(torch.stack([moved[0].double(), (moved[0] & (out0 != outj)).double()]))
        self.ctr.add_(1)

    # ------------------------------------------------------------------ one read-in-neuron update (graph body)
    def _readin_step(self):
        self._in_update(*self._in_select())

    def _in_select(self):
        m, d, sw, sb, v, nc = self.m, self.m.d, self.m.sw, self.m.sb, self.m.v, self.m.n_cand
        dt = self.dt
        i = self.ctr
        j = self.order.index_select(0, i)
        R = {key: t.index_select(0, i)[0] for key, t in self.in_rnd.items()}
        pop = self.popw
        cand = torch.cat([pop.index_select(0, j), _prop_sample(m.prop_w, pop, j, R["comp"], R["z"], R["idx"],
                                                               R["flip"])])                # (nc+1, 2)
        logq = _prop_logpdf_ex(m.prop_w, cand, pop, j, self.pc_w)
        w0c, tc = cand[:, 0], cand[:, 1]
        qc = torch.exp(tc)
        wj = self.W.index_select(0, j)                                                      # (1, d)
        n = wj[:, 1:] / wj[:, 1:].norm()
        redraw = R["u"] < m.redraw_frac
        nn = torch.cat([n, torch.where(redraw, R["nn"], n)])                                # (nc+1, d-1)
        upT = nn @ self.XpT                                                                 # (nc+1, P)
        F1j = self.F1.index_select(1, j)[:, 0]                                              # (P,)
        phT = torch.erf(w0c.to(dt)[:, None] * self.X0[None] + torch.sqrt(qc).to(dt)[:, None] * upT)
        phT = torch.cat([F1j[None], phT[1:]])                                               # (nc+1, P)
        bj = self.B.index_select(1, j)[:, 0]                                                # (N2,)
        self.f1j.copy_(F1j)
        self.bj.copy_(bj)
        # pass 1: Dm = a erf'(h0), d0 = sum a (erf(H) - erf(h0)), e0a = sum a erf(h0)
        _pass1(self.H, self.f1j, self.bj, self.a, self.Dm, self.d0p, self.e0ap)
        d0, e0a = self.d0p.sum(0), self.e0ap.sum(0)
        f0 = self.f - d0
        Dm = self.Dm
        bf = torch.bfloat16
        field = (((f0 - m.Y)[None] * phT).to(bf) @ Dm).float() / v                           # (nc+1, N2)
        mm = torch.zeros_like(field)
        rr = -field
        pp = rr
        rs = _rsum(rr * rr)[:, None]
        w2 = phT * phT / v
        for _ in range(m.cg_iters):
            Ap = pp / sb ** 2 + ((w2 * (Dm @ pp.T.to(bf)).T.float()).to(bf) @ Dm).float()
            al = rs / _rsum(pp * Ap)[:, None].clamp_min(1e-30)
            mm = mm + al * pp
            rr = rr - al * Ap
            rsn = _rsum(rr * rr)[:, None]
            pp = rr + (rsn / rs.clamp_min(1e-30)) * pp
            rs = rsn
        Bc = torch.cat([bj[None], mm[1:] + sb * R["bn"]])                                   # (nc+1, N2)
        logN = -0.5 * _rsum((Bc - mm).double() ** 2) / sb ** 2
        # pass 2: predictions of the nc new candidates; candidate 0 is the current state (f)
        BcN = Bc[1:].contiguous()
        _pass2(self.H, self.f1j, self.bj, self.a, phT[1:].T, BcN, self.S2p)
        fcT = torch.cat([self.f[None], f0[None] + self.S2p.sum(0) - e0a[None]])             # (nc+1, P)
        Ud = _rsum((fcT - m.Y[None]).double() ** 2) / (2 * v)
        Up = 0.5 * (w0c ** 2 + qc) / sw ** 2 - 0.5 * (d - 1) * tc + 0.5 * _rsum(Bc.double() ** 2) / sb ** 2
        lw = -(Ud + Up) - (logq + logN)
        r2c = w0c ** 2 / (1 + 2 * (w0c ** 2 + qc))
        frozen = (j[0].double() == self.lam[0]) & ((self.lam[1] < 1.0) | (self.conf[1] != 0)) | \
                 (self.sO.index_select(0, j)[0] != 0) & (self.conf[1] != 0)
        lw = torch.where(frozen & (torch.arange(lw.shape[0], device=lw.device) > 0), -math.inf, lw)
        lw = lw - (j[0].double() == self.spring[0]).double() * 0.5 * self.spring[2] * (r2c - self.spring[1]) ** 2
        lw = torch.where(torch.isnan(lw), -math.inf, lw)
        c = torch.argmax(lw + R["gum"]).view(1)
        return j, c, cand, nn, phT, Bc, fcT, wj, F1j

    def _in_update(self, j, c, cand, nn, phT, Bc, fcT, wj, F1j):
        dt, pop = self.dt, self.popw
        w0c, qc = cand[:, 0], torch.exp(cand[:, 1])
        moved = c != 0
        # updates (exact no-ops when c == 0)
        wnew = torch.cat([w0c.index_select(0, c).to(dt)[:, None],
                          nn.index_select(0, c) * torch.sqrt(qc.index_select(0, c)).to(dt)[:, None]], 1)
        self.W.index_copy_(0, j, torch.where(moved, wnew, wj))
        phi_c = phT.index_select(0, c)[0]                                                   # (P,)
        b_c = Bc.index_select(0, c)[0]                                                      # (N2,)
        _rank2(self.H, phT, Bc, c, self.f1j, self.bj)                                       # skipped if c == 0
        self.F1.index_copy_(1, j, phi_c[:, None])
        self.B.index_copy_(1, j, b_c[:, None])
        self.f.copy_(torch.where(moved, fcT.index_select(0, c)[0], self.f))
        pop.index_copy_(0, j, cand.index_select(0, c))
        thr = 0.5
        cr = (w0c[0].abs() > thr) != (w0c.index_select(0, c)[0].abs() > thr)
        self.in_stats.add_(torch.stack([moved[0].double(), (moved[0] & cr).double()]))
        self.ctr.add_(1)

    # ------------------------------------------------------------------ graph management
    def _run(self, name, n):
        with torch.cuda.device(self.dev):                  # graphs are captured on the current device
            self._run_on_device(name, n)
        done = int(self.ctr)
        if done != n:
            raise RuntimeError(f"{name} sweep processed {done} of {n} neurons (CUDA graph capture failed?)")

    def _run_on_device(self, name, n):
        body = self.bodies[name]
        if not self.use_graph:
            for _ in range(n):
                body()
            return
        if name not in self.graphs:
            # warm up (compiles, allocates cuBLAS workspaces) on a copy of the state, then capture one update
            keep = [t.clone() for t in self._mutable()]
            s = torch.cuda.Stream(self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                for _ in range(3):
                    body()
            torch.cuda.current_stream(self.dev).wait_stream(s)
            for t, k in zip(self._mutable(), keep):
                t.copy_(k)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            self.graphs[name] = g
        g = self.graphs[name]
        for _ in range(n):
            g.replay()

    def _mutable(self):
        return [self.W, self.B, self.a, self.F1, self.H, self.E, self.f, self.ctr, self.Bd, self.Nperp, self.popb,
                self.popw, self.mid_stats, self.in_stats]

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def sweep(self, W, B, a, layers=("middle", "readin")):
        m = self.m
        self.W.copy_(W)
        self.B.copy_(B)
        self.a.copy_(a)
        torch.matmul(m.X, self.W.T, out=self.F1).erf_()
        jl, lam = self.lam.tolist()
        if jl >= 0 and lam != 1.0:
            self.F1[:, int(jl)] *= lam
        jc, side = self.conf[:2].tolist()
        if side != 0:
            sj = 1.0 if float(self.W[int(jc), 0]) >= 0 else -1.0
            self.conf[3] = sj
            self.sO.copy_(torch.sign(self.W[:, 0].double()) * (self.sO != 0).double())
            bj = self.B[:, int(jc)].double()
            u = self.B.double() @ self.sO
            self.Sbu.fill_(sj * float(bj @ u))
            self.Sbb.fill_(float(bj @ bj))
            self.Suu.fill_(float(u @ u))
        torch.matmul(self.F1, self.B.T, out=self.H)
        torch.erf(self.H, out=self.E)
        torch.matmul(self.E, self.a, out=self.f)
        st = {}
        if "middle" in layers:
            st.update(self._middle_sweep())
        if "readin" in layers:
            st.update(self._readin_sweep())
        return self.W.clone(), self.B.clone(), self.a.clone(), st

    def _middle_sweep(self):
        m, S, N1, N2 = self.m, len(self.m.ks), self.m.N1, self.m.N2
        C = readin_coeffs(self.W, m.ks).double()
        CtC = C.T @ C
        Lc = torch.linalg.cholesky(m.sb ** 2 * CtC)
        Pc = torch.linalg.solve(CtC, C.T).T                                                 # C (C^T C)^-1
        self.C.copy_(C)
        self.Pc.copy_(Pc)
        self.LP.copy_((Pc @ Lc).T)                                                          # z -> h_S part of B_k
        self.Bd.copy_(self.B)
        hS = self.Bd @ C
        Z = torch.linalg.solve_triangular(Lc, hS.T, upper=False).T
        Bperp = self.Bd - hS @ Pc.T
        r2 = (Bperp ** 2).sum(1, keepdim=True)
        self.Nperp.copy_(Bperp / r2.sqrt())
        self.popb.copy_(torch.cat([Z, torch.log(r2)], 1))
        self._draw(self.mid_rnd, m.prop_b, N2)
        xi = self.mid_rnd["xi"].view(-1, N1)                                                # redrawn directions:
        xi.sub_((xi @ C) @ Pc.T)                                                            # project out span(C)
        xi.div_(xi.norm(dim=1, keepdim=True))
        self.order[:N2].copy_(torch.randperm(N2, device=self.dev, generator=m.gen))
        self.ctr.zero_()
        self.mid_stats.zero_()
        self._run("middle", N2)
        s = self.mid_stats.tolist()
        return {"mid_changed": s[0] / N2, "mid_crossed": s[1] / N2}

    def _readin_sweep(self):
        m, N1 = self.m, self.m.N1
        Wd = self.W.double()
        self.popw.copy_(torch.stack([Wd[:, 0], torch.log((Wd[:, 1:] ** 2).sum(1))], 1))
        self._draw(self.in_rnd, m.prop_w, N1)
        nn = self.in_rnd["nn"]
        nn.div_(nn.norm(dim=2, keepdim=True))
        self.order[:N1].copy_(torch.randperm(N1, device=self.dev, generator=m.gen))
        self.ctr.zero_()
        self.in_stats.zero_()
        self._run("readin", N1)
        s = self.in_stats.tolist()
        return {"in_changed": s[0] / N1, "in_crossed": s[1] / N1, "in_direct": 1.0}


# ----------------------------------------------------------------------------------------------------------
# sampler with the fast paths
# ----------------------------------------------------------------------------------------------------------
class FastFCN3(FCN3):
    """FCN3 with float32-gradient HMC and graphed sweeps. Drop-in for the runner: hmc_step / draw_a / sweep keep the
    reference signatures (hmc_step ignores the incoming u, gW, gB and returns u = exact energy, gW = gB = None)."""

    def __init__(self, *args, use_graph=True, compile=True, **kw):
        super().__init__(*args, **kw)
        self.fsweep = FastSweep(self, use_graph=use_graph, compile=compile) if self.dev.type == "cuda" else None
        self._cache = None                     # (ZW, ZB, L, z) of the last exact energy evaluation
        self.I32 = torch.eye(self.N2, dtype=torch.float32, device=self.dev)

    # ---------------------------------------------------------------- exact energy (float64 solve), cached
    @torch.no_grad()
    def U_exact(self, ZW, ZB):
        W, B = self.sw * ZW, self.sb * ZB
        G, r = self.suff_stats(W, B)
        M, L, z, b, ok = self._solve(G, r)
        if not ok:
            return float("inf"), None
        logdet = 2 * torch.log(torch.diagonal(L)).sum() + self.P * math.log(self.v) + self.N2 * math.log(self.sa2)
        u = 0.5 * (ZW.double() ** 2).sum() + 0.5 * (ZB.double() ** 2).sum() + 0.5 * (self.yy / self.v - b @ z) \
            + 0.5 * logdet
        return float(u), (L, z)

    # ---------------------------------------------------------------- float32 gradient (no host syncs)
    @torch.no_grad()
    def grad32(self, ZW, ZB):
        """Gradient of the collapsed potential with every operation in float32. Returns (gW, gB, ok) with ok a
        device bool (Cholesky succeeded)."""
        f32 = torch.float32
        ZW32, ZB32 = ZW.to(f32), ZB.to(f32)
        W, B = self.sw * ZW32, self.sb * ZB32
        X, Y = self.X.to(f32), self.Y.to(f32)
        G = torch.zeros(self.N2, self.N2, dtype=f32, device=self.dev)
        r = torch.zeros(self.N2, dtype=f32, device=self.dev)
        single = self.P <= self.chunk
        cache = None
        for sl in self._chunks():
            A1 = X[sl] @ W.T
            F1 = torch.erf(A1)
            Hc = F1 @ B.T
            F2 = torch.erf(Hc)
            G.addmm_(F2.T, F2)
            r.addmv_(F2.T, Y[sl])
            if single:
                cache = (A1, F1, Hc, F2)
        M = G / self.v + self.I32 / self.sa2
        L, info = torch.linalg.cholesky_ex(M)
        z = torch.cholesky_solve((r / self.v)[:, None], L)[:, 0]
        Minv = torch.cholesky_inverse(L)
        Gbar2 = (torch.outer(z, z) + Minv) / self.v                   # 2 dU/dG
        rbar = -z / self.v
        gW = torch.zeros_like(W)
        gB = torch.zeros_like(B)
        for sl in self._chunks():
            if single:
                A1, F1, Hc, F2 = cache
            else:
                A1 = X[sl] @ W.T
                F1 = torch.erf(A1)
                Hc = F1 @ B.T
                F2 = torch.erf(Hc)
            dF2 = torch.addr(F2 @ Gbar2, Y[sl], rbar)
            dH = dF2.mul_(torch.exp(-Hc * Hc)).mul_(SQPI)
            gB.addmm_(dH.T, F1)
            dA1 = (dH @ B).mul_(torch.exp(-A1 * A1)).mul_(SQPI)
            gW.addmm_(dA1.T, X[sl])
        ok = (info == 0) & torch.isfinite(gW).all() & torch.isfinite(gB).all()
        return (ZW32 + self.sw * gW).to(self.dt), (ZB32 + self.sb * gB).to(self.dt), ok

    # ---------------------------------------------------------------- HMC with exact end-point energies
    @torch.no_grad()
    def hmc_step(self, ZW, ZB, u=None, gW=None, gB=None, step=0.05, n_leap=24):
        L = int(self.np_rng.integers(max(1, n_leap // 2), n_leap + 1))
        u0, fac0 = self.U_exact(ZW, ZB)
        pW = torch.randn(ZW.shape, dtype=self.dt, device=self.dev, generator=self.gen)
        pB = torch.randn(ZB.shape, dtype=self.dt, device=self.dev, generator=self.gen)
        K0 = 0.5 * ((pW.double() ** 2).sum() + (pB.double() ** 2).sum())
        gw, gb, ok = self.grad32(ZW, ZB)
        okall = ok.clone()
        pW = pW - 0.5 * step * gw
        pB = pB - 0.5 * step * gb
        zW, zB = ZW, ZB
        for li in range(L):
            zW = zW + step * pW
            zB = zB + step * pB
            gw, gb, ok = self.grad32(zW, zB)
            okall &= ok
            if li < L - 1:
                pW = pW - step * gw
                pB = pB - step * gb
        pW = pW - 0.5 * step * gw
        pB = pB - 0.5 * step * gb
        K1 = 0.5 * ((pW.double() ** 2).sum() + (pB.double() ** 2).sum())
        rnd = torch.rand(1, device=self.dev, generator=self.gen)
        if not bool(okall):
            self._cache = (ZW, ZB) + (fac0 if fac0 else (None, None))
            return ZW, ZB, u0, None, None, 0.0, L
        u1, fac1 = self.U_exact(zW, zB)
        dH = (u1 + float(K1)) - (u0 + float(K0))
        acc = math.exp(min(0.0, -dH)) if math.isfinite(dH) else 0.0
        if float(rnd) < acc:
            self._cache = (zW, zB) + fac1
            return zW, zB, u1, None, None, acc, L
        self._cache = (ZW, ZB) + (fac0 if fac0 else (None, None))
        return ZW, ZB, u0, None, None, acc, L

    @torch.no_grad()
    def draw_a(self, W, B):
        c = self._cache
        if c is not None and c[2] is not None and torch.equal(self.sw * c[0], W) and torch.equal(self.sb * c[1], B):
            L, z = c[2], c[3]
            xi = torch.randn(self.N2, 1, dtype=F64, device=self.dev, generator=self.gen)
            return (z + torch.linalg.solve_triangular(L.T, xi, upper=True)[:, 0]).to(self.dt)
        return super().draw_a(W, B)

    @torch.no_grad()
    def sweep(self, W, B, a, layers=("middle", "readin")):
        self._cache = None
        if self.fsweep is None:
            return super().sweep(W, B, a, layers)
        return self.fsweep.sweep(W, B, a, layers)
