#!/usr/bin/env julia
# Matrix Laplace saddle with teacher y = (he1, ε) constrained to an eigenvector of Q.
#
# Extra equilibrium condition (in addition to the usual V'=0, σ²V''=1, λ-matches):
#
#     det([y, Q y]) = 0    ⇔    y ∥ eigenvector of the He1–He3 block Q(μ,σ)
#
# At a root, Q is diagonal in the eigenbasis containing y, so the teacher loads
# a single channel (no leakage into the other eigenmode).
#
# State x = [lJ1, lJ3, σ, μ, he1]  (he1 keeps its sign; σ,μ > 0).
#
# Usage:
#   julia eigen_teacher_matrix_laplace.jl
#   julia eigen_teacher_matrix_laplace.jl --d 16 --n1 200 --chi 200 --sa0 0.03 \
#       --P 1000 --kappa 4.5 --eps 1 --branch cubic
#   julia eigen_teacher_matrix_laplace.jl --rho-ray --branch cubic
#   julia eigen_teacher_matrix_laplace.jl --compare   # fixed y=(1,ε) vs eigen-y
#
# Residual lives in FCS2_VGA.residuals_fcn2_laplace_eigen_teacher.

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using NLsolve, LinearAlgebra, Printf

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])
hasflag(name) = name in ARGS
parse_list(s) = parse.(Float64, split(s, ","))

function parse_params()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    chi = parse(Float64, argval("--chi", string(n1)))  # χ=N default
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    P = parse(Float64, argval("--P", "1000"))
    kappa = parse(Float64, argval("--kappa", "4.5"))
    eps = parse(Float64, argval("--eps", "1"))
    s0 = parse(Float64, argval("--s0", "1"))
    delta = parse(Float64, argval("--delta", "0"))
    branch = Symbol(argval("--branch", "cubic"))  # :cubic | :linear
    (; d, n1, chi, sa0, P, kappa, eps, s0, delta, branch)
end

"""NLsolve wrapper: do NOT abs(he1)."""
function solve_eigen_teacher(p; μ0=0.4, σ0=0.25, he1_0=nothing, mean_only=false, tol=1e-10)
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    Q0 = hermite_kernel_block(μ0, σ0, T)
    he1_init = he1_0
    if he1_init === nothing || !isfinite(he1_init)
        he1_init, _, _, _ = he1_from_Q_eigenvector(Q0, p.eps; branch=p.branch)
        (!isfinite(he1_init)) && (he1_init = p.branch === :linear ? 1.0 : 0.0)
    end
    l1_0 = FCS2_VGA.compute_lambda1(μ0, σ0, T)
    l3_0 = FCS2_VGA.compute_lambda3(μ0, σ0, T)
    x0 = mean_only ? [l1_0, l3_0, μ0, he1_init] : [l1_0, l3_0, σ0, μ0, he1_init]

    f!(F, x) = begin
        F .= residuals_fcn2_laplace_eigen_teacher(
            x, p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
            a0=p.sa0, mean_only=mean_only,
        )
    end
    # Finite-diff Jacobian: abs(σ),abs(μ) inside the residual are not Dual-safe,
    # and he1 must stay signed (no abs on the full state).
    sol = nlsolve(f!, x0; xtol=tol, ftol=tol, iterations=8000)
    return sol
end

function summarize(sol, p; mean_only=false)
    x = sol.zero
    if mean_only
        lJ1, lJ3, μ, he1 = x
        σ = FCS2_VGA.SIG_SADDLE_DEFAULT
    else
        lJ1, lJ3, σ, μ, he1 = x
        σ, μ = abs(σ), abs(μ)
    end
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    Q = Matrix(hermite_kernel_block(μ, σ, T))
    evals, U = eigen_sym2(Q)
    y = [he1, p.eps]
    Qy = Q * y
    # Which eigen-column is y closest to?
    yn = y / (norm(y) + eps())
    overlaps = [abs(dot(yn, U[:, k])) for k in 1:2]
    kstar = argmax(overlaps)
    ρ = p.kappa / (p.sa0 * p.P)
    ρp = p.kappa / (p.sa0 * p.P)  # with a0=sa0 in χ=N amp; same as κ/(a0 P)
    params = ProblemParams2(
        d=Float32(p.d), κ=Float32(p.kappa), ϵ=Float32(p.eps),
        P=Float32(p.P), n1=Float32(p.n1), χ=Float32(p.chi),
        δ=Float32(p.delta), s0=Float32(p.s0), a0=Float32(p.sa0),
    )
    L1, L3 = FCS2_VGA.matrix_learnabilities(μ, σ, params; he1=he1)
    r_eig = teacher_eigen_residual(Q, he1, p.eps)
    r_ang = r_eig / max(norm(y) * norm(Qy), eps())
    x_eval = mean_only ? [lJ1, lJ3, μ, he1] : [lJ1, lJ3, σ, μ, he1]
    f_norm = norm(residuals_fcn2_laplace_eigen_teacher(
        x_eval, p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
        a0=p.sa0, mean_only=mean_only,
    ))
    (;
        converged=NLsolve.converged(sol), μ, σ, he1, lJ1, lJ3, L1, L3, ρ, ρp,
        Q11=Q[1, 1], Q13=Q[1, 2], Q33=Q[2, 2],
        evals, U, kstar, overlap=overlaps[kstar],
        r_eig, r_ang, f_norm,
    )
end

function print_row(tag, s)
    @printf("%-14s conv=%s  μ=%.4f σ=%.4f  he1=%.6g  L1=%.4f L3=%.4f  ",
        tag, s.converged ? "Y" : "N", s.μ, s.σ, s.he1, s.L1, s.L3)
    @printf("Q13=%.4g  |r_ang|=%.2e  eig-ovlp=%.4f  ||F||=%.2e\n",
        s.Q13, abs(s.r_ang), s.overlap, s.f_norm)
end

"""Fixed teacher y=(1,ε) matrix Laplace (no eigen constraint), for comparison."""
function solve_fixed_teacher(p; μ0=0.4, σ0=0.25, tol=1e-10)
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    l1_0 = FCS2_VGA.compute_lambda1(μ0, σ0, T)
    l3_0 = FCS2_VGA.compute_lambda3(μ0, σ0, T)
    x0 = [l1_0, l3_0, σ0, μ0]
    f!(F, x) = begin
        xp = abs.(x)
        F .= residuals_fcn2_laplace(
            xp, p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
            a0=p.sa0, matrix=true,
        )
    end
    sol = nlsolve(f!, x0; xtol=tol, ftol=tol, iterations=8000)
    x = abs.(sol.zero)
    lJ1, lJ3, σ, μ = x
    Q = Matrix(hermite_kernel_block(μ, σ, T))
    he1 = 1.0
    params = ProblemParams2(
        d=Float32(p.d), κ=Float32(p.kappa), ϵ=Float32(p.eps),
        P=Float32(p.P), n1=Float32(p.n1), χ=Float32(p.chi),
        δ=Float32(p.delta), s0=Float32(p.s0), a0=Float32(p.sa0),
    )
    L1, L3 = FCS2_VGA.matrix_learnabilities(μ, σ, params; he1=he1)
    y = [he1, p.eps]
    Qy = Q * y
    r_ang = teacher_eigen_residual(Q, he1, p.eps) / max(norm(y) * norm(Qy), eps())
    evals, U = eigen_sym2(Q)
    (;
        converged=NLsolve.converged(sol), μ, σ, he1, lJ1, lJ3, L1, L3,
        ρ=p.kappa / (p.sa0 * p.P), ρp=p.kappa / (p.sa0 * p.P),
        Q11=Q[1, 1], Q13=Q[1, 2], Q33=Q[2, 2], evals, U, kstar=0,
        overlap=NaN, r_eig=teacher_eigen_residual(Q, he1, p.eps), r_ang,
        f_norm=norm(residuals_fcn2_laplace(
            [lJ1, lJ3, σ, μ], p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
            a0=p.sa0, matrix=true,
        )),
    )
end

function multi_start_eigen(p; mean_only=false)
    μs = parse_list(argval("--mu0", "0.0,0.2,0.35,0.5,0.7,1.0,1.5,2.0"))
    σs = parse_list(argval("--sig0", "0.15,0.25,0.35"))
    # Seed he1 from both eigen-directions of each (μ,σ) seed Q, plus a few fixed guesses.
    he1_fixed = parse_list(argval("--he1-0", "0,-0.5,-2,1"))
    best = nothing
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    for μ0 in μs, σ0 in σs
        Q0 = hermite_kernel_block(max(μ0, 1e-8), σ0, T)
        he1_seeds = Float64[]
        for br in (:cubic, :linear)
            h, _, _, _ = he1_from_Q_eigenvector(Q0, p.eps; branch=br)
            isfinite(h) && push!(he1_seeds, h)
        end
        append!(he1_seeds, he1_fixed)
        for he1_0 in unique(he1_seeds)
            sol = try
                solve_eigen_teacher(p; μ0=max(μ0, 1e-8), σ0=σ0, he1_0=he1_0, mean_only=mean_only)
            catch
                nothing
            end
            sol === nothing && continue
            s = summarize(sol, p; mean_only=mean_only)
            s.f_norm < 1e-7 || continue
            # Prefer the requested eigen-branch via |u_He3| vs |u_He1| of the matched column.
            u = s.U[:, s.kstar]
            branch_ok = p.branch === :linear ? abs(u[1]) >= abs(u[2]) - 1e-9 :
                abs(u[2]) >= abs(u[1]) - 1e-9
            # Score: prefer matching branch, then smaller |he1/ε| among cubic, then ||F||.
            score = (branch_ok ? 0.0 : 10.0) + 0.01 * abs(s.he1 / p.eps) + s.f_norm
            if best === nothing || score < best.score
                best = merge(s, (; converged=true, score, branch_ok))
            end
        end
    end
    best
end

function run_point(p; compare=false)
    @printf("\n=== d=%g N=%g χ=%g sa0=%g P=%g κ=%g ε=%g  ρ=κ/(sa0 P)=%.4g  branch=%s ===\n",
        p.d, p.n1, p.chi, p.sa0, p.P, p.kappa, p.eps, p.kappa / (p.sa0 * p.P), p.branch)
    if compare
        sf = solve_fixed_teacher(p)
        print_row("fixed y=(1,ε)", sf)
    end
    s = multi_start_eigen(p)
    if s === nothing
        println("  no converged eigen-teacher saddle")
        return nothing
    end
    print_row("eigen-y", s)
    @printf("  evals(Q) = (%.6g, %.6g)  he1/ε = %.6g  U[:,%d] ovlp=%.4f\n",
        s.evals[1], s.evals[2], s.he1 / p.eps, s.kstar, s.overlap)
    return s
end

function run_rho_ray(p0)
    rhos = parse_list(argval("--rhos", "0.15,0.08,0.03,0.015"))
    out = argval("--out",
        joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN/eigen_teacher/summary.csv"))
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "rho,kappa,d,N,chi,sa0,P,eps,branch,converged,mu,sig,he1,he1_over_eps,L1,L3,Q11,Q13,Q33,r_ang,overlap,f_norm")
        for ρ in rhos
            κ = ρ * p0.sa0 * p0.P
            p = merge(p0, (; kappa=κ))
            s = run_point(p; compare=hasflag("--compare"))
            s === nothing && continue
            println(io, join((
                ρ, κ, p.d, p.n1, p.chi, p.sa0, p.P, p.eps, p.branch,
                Int(s.converged), s.μ, s.σ, s.he1, s.he1 / p.eps, s.L1, s.L3,
                s.Q11, s.Q13, s.Q33, s.r_ang, s.overlap, s.f_norm), ","))
            flush(io)
        end
    end
    println("saved ", out)
end

function main()
    p = parse_params()
    if hasflag("--rho-ray")
        run_rho_ray(p)
    else
        run_point(p; compare=hasflag("--compare"))
    end
end

main()
