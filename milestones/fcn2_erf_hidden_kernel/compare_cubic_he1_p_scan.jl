#!/usr/bin/env julia
# Compare cubic-mode sample complexity: y = (0, ε) vs y = (1, ε).
#
# Fixed (κ, N, χ=N, d, sa0, ε); scan P. Matrix Laplace saddle on the He1–He3
# block (single ±μ pair): V'(μ)=0, σ = 1/√V''. Learnability L3 = f₃/ε with
# f = Q (Q + ρ' I)⁻¹ y.
#
#   julia compare_cubic_he1_p_scan.jl
#   julia compare_cubic_he1_p_scan.jl --d 16 --n1 400 --kappa 0.5 --sa0 1 --eps 1 \
#       --Ps 50,100,200,500,1000,2000,5000,10000,20000,50000

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, NLsolve, Printf
const compute_lambda1 = FCS2_VGA.compute_lambda1
const compute_lambda3 = FCS2_VGA.compute_lambda3
const hermite_kernel_block = FCS2_VGA.hermite_kernel_block
const hermite_point_features = FCS2_VGA.hermite_point_features

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])
parse_list(s) = parse.(Float64, split(s, ","))

function amp_G(chi, n1, κ, P, a0, y, Q)
    chi_p = chi / a0
    ρp = κ / (a0 * P)
    Qm = Matrix(Q)
    qscale = abs(Qm[1, 1]) + abs(Qm[2, 2])
    Qreg = Qm + (1e-10 * (1 + qscale)) * I(2)
    v = (Qreg + ρp * I(2)) \ y
    G = -(chi_p^2) * (v * v') + chi_p * inv(Qreg)
    return (a0 / (n1 * chi)) * G
end

function learnab(Q, κ, P, a0, y, eps)
    ρp = κ / (a0 * P)
    f = Q * ((Q + ρp * I(2)) \ y)
    L1 = f[1]
    L3 = eps == 0 ? f[2] : f[2] / eps
    return L1, L3
end

"""Residuals [lJ1−λ1, lJ3−λ3, σ² V''−1, V'] for matrix Laplace with teacher y."""
function residuals(x, d, n1, chi, κ, P, a0, s0, y, T)
    lJ1, lJ3, sigS, muW = x
    sigS = abs(sigS)
    muW = abs(muW)
    λ1 = compute_lambda1(muW, sigS, T)
    λ3 = compute_lambda3(muW, sigS, T)
    Q = hermite_kernel_block(muW, sigS, T)
    G = amp_G(chi, n1, κ, P, a0, y, Q)
    V = w -> begin
        c1, c3 = hermite_point_features(w, T)
        (d / s0) * 0.5 * w^2 + 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
    end
    dV = ForwardDiff.derivative(V, muW)
    d2V = ForwardDiff.derivative(w -> ForwardDiff.derivative(V, w), muW)
    return [lJ1 - λ1, lJ3 - λ3, sigS^2 * d2V - 1.0, dV]
end

function solve_one(d, n1, chi, κ, P, a0, s0, eps, he1; μ0=0.4, σ0=0.25, tol=1e-10)
    T = 1.0 + 2.0 * (d - 1.0) * s0 / d
    y = [Float64(he1), Float64(eps)]
    best = nothing
    for (μ0i, σ0i) in ((μ0, σ0), (0.1, 0.3), (0.8, 0.2), (1.5, 0.25), (0.01, 0.5), (2.0, 0.3))
        λ1 = compute_lambda1(μ0i, σ0i, T)
        λ3 = compute_lambda3(μ0i, σ0i, T)
        x0 = [λ1, λ3, σ0i, μ0i]
        sol = try
            nlsolve(x -> residuals(x, d, n1, chi, κ, P, a0, s0, y, T), x0;
                xtol=tol, ftol=tol, iterations=8000)
        catch
            nothing
        end
        sol === nothing && continue
        NLsolve.converged(sol) || continue
        lJ1, lJ3, sigS, muW = abs.(sol.zero)
        Q = Matrix(hermite_kernel_block(muW, sigS, T))
        L1, L3 = learnab(Q, κ, P, a0, y, eps)
        fn = norm(residuals(sol.zero, d, n1, chi, κ, P, a0, s0, y, T))
        cand = (μ=muW, σ=sigS, L1=L1, L3=L3, Q11=Q[1, 1], Q13=Q[1, 2], Q33=Q[2, 2],
            f_norm=fn, converged=true)
        if best === nothing || cand.f_norm < best.f_norm
            best = cand
        end
    end
    return best
end

function main()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "400"))
    chi = parse(Float64, argval("--chi", string(n1)))  # χ=N default
    if chi < 0
        chi = n1
    end
    κ = parse(Float64, argval("--kappa", "0.5"))
    a0 = parse(Float64, argval("--sa0", "1"))
    eps = parse(Float64, argval("--eps", "1"))
    s0 = 1.0
    Ps = parse_list(argval("--Ps",
        "30,50,80,120,200,300,500,800,1200,2000,3000,5000,8000,12000,20000,30000,50000"))
    out_dir = argval("--outdir",
        joinpath(@__DIR__, "action_h0_activation_plots", "CubicHe1CompareChiN"))
    mkpath(out_dir)
    out = joinpath(out_dir, @sprintf("he1_compare_d%d_N%d_chi%d_k%.4g_sa0%.4g_eps%.4g.csv",
        Int(d), Int(n1), Int(chi), κ, a0, eps))

    @printf("Matrix Laplace P-scan: d=%g N=%g χ=%g κ=%g sa0=%g ε=%g\n", d, n1, chi, κ, a0, eps)
    @printf("Teachers: y=(0,ε) and y=(1,ε); cubic learnable when L3 ≳ 0.5\n")
    println("writing ", out)

    open(out, "w") do io
        println(io, "he1,P,kappa,rho_p,L1,L3,mu,sig,Q11,Q13,Q33,f_norm,converged")
        for he1 in (0.0, 1.0)
            @printf("\n=== y = (%.0f, ε=%.4g) ===\n", he1, eps)
            warm = nothing
            for P in Ps
                s = solve_one(d, n1, chi, κ, P, a0, s0, eps, he1;
                    μ0 = warm === nothing ? 0.4 : warm.μ,
                    σ0 = warm === nothing ? 0.25 : warm.σ)
                ρp = κ / (a0 * P)
                if s === nothing
                    @printf("  P=%-8.0f  no convergence\n", P)
                    @printf(io, "%.0f,%.10g,%.10g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n",
                        he1, P, κ, ρp)
                    continue
                end
                warm = s
                @printf(io, "%.0f,%.10g,%.10g,%.10g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.4g,%d\n",
                    he1, P, κ, ρp, s.L1, s.L3, s.μ, s.σ, s.Q11, s.Q13, s.Q33, s.f_norm, 1)
                mark = s.L3 >= 0.5 ? "LEARNABLE" : (s.L3 >= 0.1 ? "rising" : "small")
                @printf("  P=%-8.0f  ρ'=%.3g  L1=%+.4f  L3=%+.4f  μ=%.3f σ=%.3f  %s\n",
                    P, ρp, s.L1, s.L3, s.μ, s.σ, mark)
                flush(stdout)
            end
        end
    end
    println("\nsaved ", out)
end

main()
