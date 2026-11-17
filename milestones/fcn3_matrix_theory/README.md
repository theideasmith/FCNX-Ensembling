# FCN3 matrix equations of state: GPU solvers

## Default: `fcn3_fixed.solve_fcn3(p)`

Use this one. It finds the equilibrium as the global maximum of the free energy W over the full read-in and
hidden-field distributions (`BlazingFreeEnergy`), at n_max = 3 from the prior and then at n_max = 11 from that
solution. It takes a few seconds per solve on a GPU (about 3 s each stage at d = 150 on an idle GPU).

```python
from fcn3_matrix_solver import Params
from fcn3_fixed import solve_fcn3
r = solve_fcn3(Params(...), nmax=11)        # r["L1"], r["L3"], r["d_w0sq"], r["H"], r["J"], r["M"], r["W"]
```

- Read-in layer: fixed (w0, log q) grid, distribution parametrised by its exponents, re-whitened every Newton step
  (the moments r^2k are nearly collinear at n_max = 11, so a J or moment parametrisation is infeasible or stalls).
- Hidden field: fixed importance samples from a mixture (prior, fitted near basin, each far basin and its mirror).
- Exact Hessian by autograd; two sample-set fidelities; pruned read-in grid with a guard; M from (E8) updated inside
  Newton in the final stage.
- Validated with `solve_fcn3` itself: thesis point N1 = 1600, n_max = 3 gives mu1 = 0.993744, mu3 = 0.30296, far
  read-in mass 4.848e-4 (verified lattice solution: 0.993745, 0.30310, 4.85e-4); ray d = 150, N1 = inf, n_max = 11
  gives mu3 = 0.38556, <h1^2> = 0.03875 (lattice: 0.3856, 0.03878); ray d = 150, N1 = 1500 agrees with the
  J-parametrised solver at n_max = 3 (mu3 0.77154); a different Monte Carlo seed changes mu3 by < 1e-4.
- N1 = inf has no read-in degrees of freedom (the read-in distribution is the prior) and is handled by the
  J-parametrised path, which only solves for t.
- The FCN3 sampler (`../fcn3_collapsed_hmc/run_fcn3.py`) calls it for the theory column of summary.json.

The solvers below are older (lattice integrators) and are kept for reference.

## Older solvers

Two solvers for equations (E1)–(E8) of `notes/fcn3_matrix_eos.tex`. Both integrate over the **full** read-in and
hidden-field distributions. There is no cut at a barrier, and the grids are wide enough that their edges hold no
mass (`check_closed_loop.py` checks this).

| solver | what | thesis point, n_max = 3 | n_max = 11 |
|---|---|---|---|
| `fcn3_fast.FastSolver` | one Newton system, quadrature only where there is mass | 3.3 s | 19 s |
| `fcn3_matrix_solver.FCN3Solver` | nested reference (Anderson + Brent + Newton), full grids | ~40–80 s | does not converge |

```
python fcn3_fast.py 3          # thesis point (d=150, P=3000, N1=N2=1600, chi=10, kappa=0.1, eps=0.074), n_max = 3
python fcn3_fast.py 11
python check_closed_loop.py    # independent check of every equation over the full distributions (n_max = 3)
python conv_study.py 3         # refine/widen each grid of the reference solver
```
Warm start for sweeps: `FastSolver(p2).solve(init=other.state())` (~0.3–1 s per neighbouring point).

## What makes the full equations hard, and what the solvers do about it
1. **Far minima, exponentially sensitive mass.** The read-in −log p(w0, q) develops a second minimum at large w0
   and small q (rows nearly aligned with the target). Its mass depends exponentially on the state: in (E5) the
   He_n term carries 1/G_nn, so ~1e-4 of the rows there multiplies J_33 (and J_nn for n ≥ 5 by much more), which
   removes the minimum again. Plain iteration jumps between "all mass far" and "none". The solvers make the
   log mass of each far basin an unknown ell_g. The inner distributions are reweighted to those masses, and ell is
   solved from ell_g = log P(B_g) − log P(B_0). At the root the reweighting is the identity, so the solution is that
   of the full distributions whatever the basins are.
2. **15 decades in J.** At n_max = 11, J_nn runs from 4e-3 to 5e-18. The iteration variables are log J_nn and the
   atanh of the correlations. Whitening uses G = D R D with the Cholesky factor of the correlation matrix R
   (a plain Cholesky of G loses all precision in the smallest directions).
3. **G^-1 (G − H) G^-1 without cancellation.** By Gaussian integration by parts (Stein's lemma) twice,
   G^-1 (G − H) G^-1 = −E[∇∇T + ∇T ∇T^T] with T = (g~/2)(t·e)^2. The right side has no G^-1 and no
   difference of large numbers. The left side, computed as I − E[zz^T] in whitened coordinates, carries rounding
   noise ~1e-16 multiplied by |L^-1 c|^2 ~ 1e15 at n_max = 11, i.e. O(1) noise in the read-in action. Both forms
   agree to 7e-16 at n_max = 3.

## FastSolver
- Unknowns: x = (log J_nn, atanh correlations, log qbar_h, log M), tau = (kappa/P) t, far-basin logits ell.
  Residual: (E6)–(E8) map minus x, (E4), and ell_g − log mass ratio.
- Per continuation step: a pass over the full grids (one node on each spectator zeta direction) finds the basins of
  both distributions and the cells/points within e^40 of each basin's peak. Newton then works on those only, with the
  quadrature frozen, so the residual is smooth. A finite-difference Jacobian is computed and then updated by
  Broyden's method. After the step another full pass checks that the cells left out hold < e^-30 of the mass and
  that no new far basin has appeared; otherwise the quadrature is rebuilt.
- Continuation in log chi from chi/64: secant predictor, step from how much the solution moved, at most ×√2.
- Reductions with tiny outputs and long sums (e.g. J = Σ c c^T p over 6e4 points, per-cell 6×6 sums over 1e4 zeta
  nodes) are written as elementwise products and sums. As cuBLAS fp64 matmuls they were ~100× slower.
- Transverse zeta rule: default (counts [16, 4, 2, 2, 1] at n_max = 11) is within 1.5e-6 of mu_3 of a ~1e4-times
  larger rule. `fcn3_matrix_solver.ZETA_ACCURATE` gives ~4e-7 at ~100× the cost.

## Results at the thesis point (full distributions)
| n_max | mu_1 | mu_3 | d<w0^2> | read-in mass in the far minimum |
|---|---|---|---|---|
| 3 | 0.993745 | 0.30310 | 1.959 | 4.85e-4 |
| 11 | 0.993435 | 0.14879 | 1.862 | 4.17e-4 |

The solver that cut the distributions at the first barrier (removed) gave mu_3 = −0.1965 at n_max = 3. That state
is not a solution of (E1)–(E8).

## Closed-loop check against the notes (`check_closed_loop.py`)
Takes only a solver's converged state (J, q̄_h, M, t) and re-evaluates (E1)–(E8) over the full distributions with
independent numerics: Gauss–Hermite e_n and c_n (not the closed form), a plain whitened grid for the hidden field
(±48 standard deviations, q_h from e^-14 to e^3 × q̄_h), a plain (w0, log q) grid with both signs of w0, and Monte
Carlo for E k². It also reports the mass on every grid edge.

FastSolver at the thesis point, n_max = 3:
- No grid edge holds mass (0 on the hidden-field grids, 2.6e-144 on the read-in q edge).
- t agrees to 3e-12, J to 1e-9, q̄_h to 3e-13, and H and ⟨q_h⟩ in every digit shown.
- M is within the Monte Carlo error (9e-5).

The read-in mass at |w0| > 1 on the independent grid (4.848e-4) equals the solver's far-basin mass.

## Convergence (`conv_study.py`, reference solver, n_max = 3)
I halved ds, doubled the q_h points, x0 nodes and u nodes, widened s, q_h, w0 and q, and refined the pair blocks.
mu_1, mu_3, H and the far mass move by ≤ 2e-11. M moves by ≤ 2e-4, from the read-in q resolution and the atom
blocks for E k².

## Limitations
- The checker's hidden-field grid is 2-D, so the independent check covers n_max = 3 only. At n_max = 11 the
  evidence is the zeta-rule study above and the excluded-mass check inside FastSolver.
- The theory's own approximations remain (notes, section "Approximations").

## N1 → ∞: comparison with the FCN2 theory (`compare_fcn2_limit.py`)
N1 enters only through rB = σ_B² N2/N1, which scales both back-couplings. At N1 = ∞ the read-in law is its prior
and the middle layer is the FCN2 read-in problem on a fixed field: w0 variance → G11, d − 1 → M, perp mean → q̄,
N → N2 (the readout and tilt map exactly). With G_S projected onto He1 the solver reproduces the FCN2 q-theory
(`../fcn2_collapsed_hmc/readin_action_theory_q.py`, with that prior) to 7–9 digits at d = 75–1200, global
branch. The full G_S also carries the first layer's He3(x0) content (√(G33/G11) ≈ 1/d, ρ13 ≈ −0.78), which moves
L3 at every d (thesis point, N1 = ∞, global, S = {1, …, 11}; output in `compare_fcn2_limit.json`):

| d | 75 | 150 | 300 | 600 | 1200 |
|---|---|---|---|---|---|
| L3, FCN3 | 0.0695 | −0.1290 | −0.0956 | −0.0602 | −0.0384 |
| L3, FCN2 (mapped) = FCN3 with He1-only field | 0.1224 | −0.0869 | −0.0613 | −0.0428 | −0.0297 |

The default tower is now S = {1, …, 11} (`Params.nmax = 11`). With S = {1, 3} the d = 75 values were 0.2009 / 0.2229
(outer-well neurons there carry He5+); at d ≥ 150 the truncation moves L3 by < 0.01.

At d = 150 the full field also deepens the middle outer well (2e-5 of the mass, but it carries much of K33), so
the central and global branches differ there (L3 −0.152 vs −0.121 at S = {1, 3}); the He1-only field has outer mass 5e-14.
At d ≥ 300 the middle law is unimodal.
