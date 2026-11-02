# FCN3 matrix equations of state: GPU solver

This folder solves eqs. E1–E8 of the notes *FCN3 in Hermite matrix elements* (Nov 2026): the three-layer erf
network on the odd x0 Hermite tower S = {1, 3, …, n_max}. Both hidden layers use exact non-Gaussian actions
(read-in in (w0, q⊥), middle in (h, q_h)) and every kernel is kept as a full matrix. The FCN2 counterpart is
`../fcn2_collapsed_hmc/readin_action_theory_q.py`.

```
python fcn3_matrix_solver.py                       # thesis point (Fig. 4.3), S = {1,3}, kappa = 0.1
python fcn3_matrix_solver.py --nmax 11 --kappa 0.147
python fcn3_matrix_solver.py --nmax 7 --sweep kappa=0.147,0.12,0.1,0.08 --out sweep.json   # warm-started
python check_fcn3_solver.py                        # independent checks (below)
python conv_study.py 7 0.1                         # double/halve each quadrature knob
```
From Python: `FCN3Solver(Params(...), Numerics(...)).solve()` returns a dict with J, H, K, G, t, f̄,
L1, L3, d⟨w0²⟩, ⟨q_h⟩/q̄_h, M, m⊥, the eigenvalues of M_S, both barriers, and the ESS. Pass `init=other.state()`
to warm-start.

## Method
- **Inner loop**: Newton with Armijo backtracking on the convex potential
  Φ(t) = (κ/2P)|t|² + (N2/χ) log Z_h(t) − y·t. Its gradient is the readout condition, and its Hessian is
  κ/P + σ_A²K + σ_A² g̃ Cov[(t·B)B]. From a warm start it takes 2–4 integral passes.
- **Outer loop**: Anderson acceleration (depth 6) on (vech J_S normalised by the prior J, log q̄_h, log M).
  - From a cold start it uses **continuation in χ**, because M_S ∝ χ. At the thesis point the prior J gives
    M_11 ≈ 450, above the threshold 2β·3π/4 ≈ 354 where −½M_11c1² beats βw0², so the origin of the read-in action is unstable and no direct iteration converges.
  - The χ step adapts: it grows ×1.5 after a success and halves when a step loses the central basin.
  - Intermediate χ steps are solved to 1e-4; only the last is solved to 1e-8.
- **Middle integral** (|S|+1 dims). The coordinates are whitened, h = L z with G_S = LLᵀ, and
  z = s u + Q⊥ζ with u ∝ Lᵀt, the drive direction at h = 0.
  - s: a uniform lattice with s ≥ 0 only. The law is exactly symmetric under h → −h, so the negative half is
    not computed. The per-s sufficient statistics give the exact s-marginal, so the central-basin cut ("first
    maximum of −log p") is a cut on the lattice.
  - q_h: Gauss–Hermite in log q_h, with weights exact against the two-moment Gamma prior (eq. 16).
  - ζ: rotated to the singular directions of ζ → erf argument. The sensitivities fall off geometrically
    (7e-2, 6e-5, 1e-6, 3e-8, 8e-10 relative to s at the thesis point), so a product Gauss–Hermite rule with
    24 nodes on the one active direction and 2–3 on each spectator is enough: 576 points at n_max = 11.
    Sobol QMC with 2048 points was 1e-4–1e-3 off; it is still available as `zeta_rule="qmc"`.
  - Corrections that are differences from the prior (H − G, ⟨q_h⟩ − q̄_h) are computed on the same nodes with
    prior weights as a control variate.
  - B_n is one fp64 erf block, (s·a + A_ζ)/√(1+2q_h), followed by a matmul against the 64 Gauss–Hermite
    nodes in x0. The small |S|×|S| Gram matrices use broadcast-and-sum: fp64 tensor-core `bmm` with tiny
    outputs was 20× slower and took 70% of a pass.
- **Read-in**: a (w0, q⊥) tensor grid, 1201 × 256, sinh-spaced in w0 and log-uniform in q⊥, with the cut at the
  first maximum of −log p(w0). M = (tr G⊥)²/tr G⊥² comes from eq. (18): pairs on a 160 × 40 coarse grid with
  Gauss–Jacobi quadrature in the angle, weight (1−u²)^((d−4)/2).

## Validation (thesis point, S = {1, 3}; `check_fcn3_solver.py`)
- B_n(h1 e1, q_h) from quadrature equals the closed form c_n(h1, q_h): 1e-12 at q_h = 2, 1e-6 at h1 = 3 with
  q_h = 0.46. The central basin has |h1| < 0.8, where the error is smaller still.
- The readout identity (σ_A²K + κ/P)t = y holds to 1e-15.
- A brute-force middle law in the original (h1, h3) coordinates (801 × 401 × 64 grid, no whitening, no
  lattice, no control variate) matches K, H − G and ⟨q_h⟩ − q̄_h to < 5e-6 relative.
- E_{x,x'}J² from Monte Carlo over explicit read-in vectors in R^d matches the pair quadrature within MC error.
- Doubling or halving each quadrature knob changes the observables by ≤ 1e-7 (`conv_study.py`; n_qh = 8
  is the worst case at 1.1e-7). The
  barrier heights, which are diagnostics only, move by ≤ 3e-4.

Against Table 1 of the notes (central basin, S = {1, 3}):

| κ | | d⟨w0²⟩ | H11 | H33 | H13 | L1 | L3 | barriers (read-in / middle) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | notes | 1.63 | 0.852e-2 | 0.55e-6 | −0.61e-4 | 0.993 | −0.196 | +30 / +35 |
| 0.1 | this solver | 1.632 | 0.849e-2 | 0.548e-6 | −0.605e-4 | 0.993 | −0.1965 | +29.4 / +34.6 |
| 0.147 | notes | 1.70 | 0.889e-2 | 0.67e-6 | −0.68e-4 | 0.990 | −0.212 | +50 / +64 |
| 0.147 | this solver | 1.693 | 0.886e-2 | 0.666e-6 | −0.680e-4 | 0.990 | −0.2123 | +48 / +60 |

The tower converges: L3 is −0.1965057, −0.1965026, −0.1965025 and −0.1965024 at n_max = 3, 5, 7 and 11, and
f̄5 = 5.35e-4, f̄7 = −3.0e-5, f̄9 = 2.2e-6, f̄11 = −2.0e-7.

## Timings (RTX 4090, fp64, cold start including the χ continuation)
| n_max | one middle pass | cold solve | warm-started neighbour (κ 0.1 → 0.09, incl. setup) |
|---|---|---|---|
| 3 | 1.7 ms | 0.7 s | 0.15 s |
| 7 | 4.8 ms | 1.0 s | 0.2 s |
| 11 | 19 ms | 2.8 s | 1.1 s |

## Limitations
- `branch="global"` (outer wells) does not solve. χ continuation from the prior follows the central branch
  and fails where that branch ends. The outer-well state needs its own starting point (notes §7, item 5).
- The theory's own approximations remain: the restriction to S, Gaussian h⊥ over x⊥, the two-moment q_h
  law, and the dropped connected term χT (notes §7).
