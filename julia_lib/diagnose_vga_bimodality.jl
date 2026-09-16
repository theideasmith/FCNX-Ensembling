#!/usr/bin/env julia
# Is the VGA muW=0 solution a free-energy minimum or a symmetric saddle?
#
# nlsolve_solver_fcn2 drives the *gradient* of the variational free energy to
# zero. F depends on the mode location m only through m^2, so m=0 is a
# stationary point for every parameter choice. This script checks the curvature
# at m=0 and scans F(m) to see whether a bimodal minimum exists.

include("FCS2_VGA.jl")
using .FCS2_VGA
using ForwardDiff
using Printf

function landscape(; d, n1, P, chi, kappa, s0=1.0, delta=1.0, advanced=false)
    init = [1.0 / d, 1.0 / d^3, sqrt(0.8 / d), 0.1]
    params = FCS2_VGA.ProblemParams2(
        d=Float32(d), κ=Float32(kappa), ϵ=Float32(0.5),
        P=Float32(P), n1=Float32(n1), χ=Float32(chi), δ=Float32(delta), s0=Float32(s0),
    )
    sol = FCS2_VGA.solve_FCN2_Erf(params, init; anneal_steps=3000, use_anneal=true, advanced=advanced)

    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    # Self-consistent free energy: A_pot recomputed at each (s, m).
    function F_sc(s, m)
        lJ1_gen = FCS2_VGA.compute_lambda1(m, s, T_floor)
        lT1 = -(chi^2 / (kappa / P + lJ1_gen)^2 * delta) + chi * lJ1_gen^(-1) +
              (chi^2 * kappa / (P * chi)) * (lJ1_gen / (lJ1_gen + kappa / P))
        A_pot = lT1 / (n1 * chi)
        prior = (d / s0) * 0.5 * (m^2 + s^2)
        energy = A_pot * FCS2_VGA.compute_lambda1(m, s, T_floor)
        neg_S = advanced ? -FCS2_VGA.exact_symmetric_gmm_entropy_1d(m, s) :
                (-0.5 * log(s^2 + 1e-25) + log(1.0 + exp(-m^2 / s^2)))
        return prior + energy + neg_S
    end

    # Fixed-field free energy: A_pot frozen at the returned solution, which is
    # exactly the objective whose gradient the solver zeroes.
    lJ1_sol = FCS2_VGA.compute_lambda1(sol.muW, sol.sigS, T_floor)
    lT1_sol = -(chi^2 / (kappa / P + lJ1_sol)^2 * delta) + chi * lJ1_sol^(-1) +
              (chi^2 * kappa / (P * chi)) * (lJ1_sol / (lJ1_sol + kappa / P))
    A_pot_sol = lT1_sol / (n1 * chi)
    function F_fixed(s, m)
        prior = (d / s0) * 0.5 * (m^2 + s^2)
        energy = A_pot_sol * FCS2_VGA.compute_lambda1(m, s, T_floor)
        neg_S = advanced ? -FCS2_VGA.exact_symmetric_gmm_entropy_1d(m, s) :
                (-0.5 * log(s^2 + 1e-25) + log(1.0 + exp(-m^2 / s^2)))
        return prior + energy + neg_S
    end

    @printf("d=%g n1=%g P=%g chi=%g kappa=%.4g  advanced=%s\n", d, n1, P, chi, kappa, advanced)
    @printf("  solver:  sigS=%.6g  muW=%.6g  lWT=%.6g  lJ1=%.6g  lJ3=%.6g\n",
        sol.sigS, sol.muW, sol.lWT, sol.lJ1, sol.lJ3)
    @printf("  A_pot at solution = %.6g\n", A_pot_sol)

    s_star = sol.sigS
    for (label, F) in (("fixed-field", F_fixed), ("self-consistent", F_sc))
        # Curvature in m at m -> 0 (finite difference; m=0 is singular for the
        # Hershey-Olsen log term's derivative, so probe a small offset).
        h = 1e-4
        d2 = (F(s_star, h) - 2 * F(s_star, 0.0) + F(s_star, -h)) / h^2
        best_m, best_F = 0.0, F(s_star, 0.0)
        for m in range(0.0, 1.5, length=1501)
            val = F(s_star, m)
            if val < best_F
                best_F, best_m = val, m
            end
        end
        @printf("  [%s] d2F/dm2|_0 = %+.6g   argmin_m F(s*,m) = %.4g   F(0)-F(min) = %.4g\n",
            label, d2, best_m, F(s_star, 0.0) - best_F)
    end

    # Full 2D minimum over (s, m) of the self-consistent free energy.
    best = (Inf, 0.0, 0.0)
    for s in range(0.02, 1.2, length=300), m in range(0.0, 1.2, length=300)
        val = F_sc(s, m)
        if val < best[1]
            best = (val, s, m)
        end
    end
    @printf("  [self-consistent] 2D grid min: s=%.4g  m=%.4g  (lWT=%.4g)  F=%.6g\n",
        best[2], best[3], best[2]^2 + best[3]^2, best[1])
    println()
    return sol
end

println("="^80)
println("Journal Langevin cubic runs (d=20, N=chi=400, kappa=kappa_eff)")
println("="^80)
for (P, kap) in ((50, 0.7028), (2000, 0.5374), (16000, 0.5223))
    landscape(d=20.0, n1=400.0, P=P, chi=400.0, kappa=kap)
end

println("="^80)
println("Milestone-style runs for comparison")
println("="^80)
landscape(d=150.0, n1=1600.0, P=600.0, chi=1600.0, kappa=1.0)
landscape(d=150.0, n1=700.0, P=600.0, chi=700.0, kappa=1.0)
landscape(d=100.0, n1=800.0, P=1200.0, chi=80.0, kappa=2.0)
