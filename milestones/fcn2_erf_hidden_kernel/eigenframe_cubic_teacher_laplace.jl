#!/usr/bin/env julia
# Matrix Laplace solved in the eigenframe with ỹ = (0, ε).
#
# At each candidate (μ,σ):
#   1. Build full Hermite block Q (keeps λ13; no artificial Q13=0)
#   2. Diagonalize Q = U Λ Uᵀ
#   3. Set eigenframe teacher ỹ = (0, ε); build G̃ = G̃(Λ, ỹ)
#   4. V(w) = prior + ½ φᵀ G̃ φ with φ = Uᵀ c(w); impose V'=0, σ²V''=1, λ-matches
#
# Equations live in the eigenframe — silent eigenchannel has zero teacher load.
#
# Distinct from:
#   - --decouple (zeros Hermite Q13)
#   - eigen_teacher_matrix_laplace.jl (fixes lab y3=ε, free he1)
#
# Usage:
#   julia eigenframe_cubic_teacher_laplace.jl --compare
#   julia eigenframe_cubic_teacher_laplace.jl --rho-ray --branch cubic
#   julia eigenframe_cubic_teacher_laplace.jl --d 16 --n1 200 --chi 200 --sa0 0.03 \
#       --P 1000 --kappa 4.5 --eps 1

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using NLsolve, LinearAlgebra, Printf, ForwardDiff

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])
hasflag(name) = name in ARGS
parse_list(s) = parse.(Float64, split(s, ","))

function parse_params()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    chi = parse(Float64, argval("--chi", string(n1)))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    P = parse(Float64, argval("--P", "1000"))
    kappa = parse(Float64, argval("--kappa", "4.5"))
    eps = parse(Float64, argval("--eps", "1"))
    s0 = parse(Float64, argval("--s0", "1"))
    delta = parse(Float64, argval("--delta", "0"))
    branch = Symbol(argval("--branch", "cubic"))
    (; d, n1, chi, sa0, P, kappa, eps, s0, delta, branch)
end

function solve_eigenframe(p; μ0=0.4, σ0=0.25, tol=1e-10)
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    x0 = [FCS2_VGA.compute_lambda1(μ0, σ0, T), FCS2_VGA.compute_lambda3(μ0, σ0, T), σ0, μ0]
    f!(F, x) = begin
        xp = [abs(x[1]), abs(x[2]), abs(x[3]), abs(x[4])]
        F .= residuals_fcn2_laplace_eigenframe_cubic(
            xp, p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
            a0=p.sa0, branch=p.branch,
        )
    end
    nlsolve(f!, x0; xtol=tol, ftol=tol, iterations=8000)
end

function solve_fixed_lab(p; μ0=0.4, σ0=0.25, he1=0.0, tol=1e-10)
    """Classical matrix Laplace with fixed lab y=(he1, ε)."""
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    x0 = [FCS2_VGA.compute_lambda1(μ0, σ0, T), FCS2_VGA.compute_lambda3(μ0, σ0, T), σ0, μ0]
    f!(F, x) = begin
        xp = abs.(x)
        Q = hermite_kernel_block(xp[4], xp[3], T)
        # use matrix laplace amp with explicit y
        # residuals_fcn2_laplace hardcodes y=(1,ε); rebuild residual inline for he1=0
        lJ1, lJ3, sigS, muW = xp
        lJ1_gen = FCS2_VGA.compute_lambda1(muW, sigS, T)
        lJ3_gen = FCS2_VGA.compute_lambda3(muW, sigS, T)
        y = [he1, p.eps]
        G = laplace_amplitude_matrix(p.chi, p.n1, p.kappa, p.P, p.delta, p.eps, Q; a0=p.sa0, y=y)
        V = w -> begin
            c1, c3 = hermite_point_features(w, T)
            (p.d / p.s0) * 0.5 * w^2 + 0.5 * (G[1,1]*c1^2 + 2G[1,2]*c1*c3 + G[2,2]*c3^2)
        end
        dV = ForwardDiff.derivative(V, muW)
        d2V = ForwardDiff.derivative(w -> ForwardDiff.derivative(V, w), muW)
        F[1] = lJ1 - lJ1_gen
        F[2] = lJ3 - lJ3_gen
        F[3] = sigS^2 * d2V - 1
        F[4] = dV
    end
    nlsolve(f!, x0; xtol=tol, ftol=tol, iterations=8000)
end

function summarize(sol, p; tag="eigenframe")
    x = abs.(sol.zero)
    lJ1, lJ3, σ, μ = x
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    Q = Matrix(hermite_kernel_block(μ, σ, T))
    y, evals, U, k = lab_teacher_eigenframe_cubic(Q, p.eps; branch=p.branch)
    ỹ = U' * y
    ρp = p.kappa / (p.sa0 * p.P)
    f = Q * ((Q + ρp * I(2)) \ y)
    # eigenframe learnabilities: only channel k is driven
    L_eig = [evals[i] / (evals[i] + ρp) * ỹ[i] for i in 1:2]
    Fn = residuals_fcn2_laplace_eigenframe_cubic(
        x, p.P, p.chi, p.d, p.kappa, p.delta, p.n1, p.s0, p.eps;
        a0=p.sa0, branch=p.branch,
    )
    (;
        tag, converged=NLsolve.converged(sol), μ, σ, lJ1, lJ3,
        y_lab=y, y_rot=ỹ, k, evals, U,
        f_lab=f, L_eig, f_norm=norm(Fn),
        he1_lab=y[1], he3_lab=y[2],
        Q11=Q[1,1], Q13=Q[1,2], Q33=Q[2,2],
    )
end

function summarize_fixed(sol, p; he1=0.0)
    x = abs.(sol.zero)
    lJ1, lJ3, σ, μ = x
    T = 1.0 + 2.0 * (p.d - 1.0) * p.s0 / p.d
    Q = Matrix(hermite_kernel_block(μ, σ, T))
    y = [he1, p.eps]
    evals, U = eigen_sym2(Q)
    ỹ = U' * y
    ρp = p.kappa / (p.sa0 * p.P)
    f = Q * ((Q + ρp * I(2)) \ y)
    (;
        tag="lab y=($he1,ε)", converged=NLsolve.converged(sol), μ, σ, lJ1, lJ3,
        y_lab=y, y_rot=ỹ, k=0, evals, U, f_lab=f,
        L_eig=[NaN, NaN], f_norm=NaN,
        he1_lab=y[1], he3_lab=y[2],
        Q11=Q[1,1], Q13=Q[1,2], Q33=Q[2,2],
    )
end

function print_row(s)
    @printf("%-18s conv=%s  μ=%.4f σ=%.4f  y_lab=(%.4g, %.4g)  ỹ=(%.3e, %.3e)  ",
        s.tag, s.converged ? "Y" : "N", s.μ, s.σ, s.he1_lab, s.he3_lab, s.y_rot[1], s.y_rot[2])
    @printf("f_lab=(%+.3e, %+.3e)  Q13=%.3g  ||F||=%.2e\n",
        s.f_lab[1], s.f_lab[2], s.Q13, s.f_norm)
end

function multi_start(p)
    best = nothing
    for μ0 in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0)
        for σ0 in (0.15, 0.25, 0.35)
            sol = try
                solve_eigenframe(p; μ0=max(μ0, 1e-6), σ0=σ0)
            catch
                nothing
            end
            sol === nothing && continue
            s = summarize(sol, p)
            s.f_norm < 1e-7 || continue
            if best === nothing || s.f_norm < best.f_norm
                best = s
            end
        end
    end
    best
end

function run_point(p; compare=false)
    @printf("\n=== d=%g N=%g χ=%g sa0=%g P=%g κ=%g ε=%g  ρ=%.4g  branch=%s ===\n",
        p.d, p.n1, p.chi, p.sa0, p.P, p.kappa, p.eps, p.kappa/(p.sa0*p.P), p.branch)
    @printf("  posing: y_lab = ε u_cubic(Q)  ⇒  ỹ = (0, ε) exactly (He1-like eigenchannel silent)\n")
    if compare
        for he1 in (0.0, 1.0)
            sol = solve_fixed_lab(p; he1=he1)
            print_row(summarize_fixed(sol, p; he1=he1))
        end
    end
    s = multi_start(p)
    if s === nothing
        println("  no converged eigenframe-cubic saddle")
        return nothing
    end
    print_row(s)
    @printf("  evals(Q)=(%.6g, %.6g)  active col k=%d  ỹ=(%.3e, %.4g)  |ỹ₁|/ε=%.2e\n",
        s.evals[1], s.evals[2], s.k, s.y_rot[1], s.y_rot[2], abs(s.y_rot[1]) / abs(p.eps))
    return s
end

function run_rho_ray(p0)
    rhos = parse_list(argval("--rhos", "0.15,0.08,0.03,0.015"))
    out = argval("--out",
        joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN/eigenframe_cubic/summary.csv"))
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "rho,kappa,branch,converged,mu,sig,he1_lab,he3_lab,yrot1,yrot2,f1,f3,Q11,Q13,Q33,k,f_norm")
        for ρ in rhos
            p = merge(p0, (; kappa=ρ * p0.sa0 * p0.P))
            s = run_point(p; compare=hasflag("--compare"))
            s === nothing && continue
            println(io, join((
                ρ, p.kappa, p.branch, Int(s.converged), s.μ, s.σ,
                s.he1_lab, s.he3_lab, s.y_rot[1], s.y_rot[2],
                s.f_lab[1], s.f_lab[2], s.Q11, s.Q13, s.Q33, s.k, s.f_norm), ","))
        end
    end
    println("saved ", out)
end

function main()
    p = parse_params()
    hasflag("--rho-ray") ? run_rho_ray(p) : run_point(p; compare=hasflag("--compare"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
