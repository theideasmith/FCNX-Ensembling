#!/usr/bin/env julia
# Standard Laplace saddle, only change = eigenframe basis:
#
#   Same as residuals_fcn2_laplace / matrix Laplace:
#     Q = hermite_kernel_block(μ, σ)   # ±μ Gaussian, width σ
#     V'(μ) = 0,   σ = 1/√V''(μ)
#
#   Modification only:
#     Q = U Λ Uᵀ
#     ỹ = (0, ε) in that frame
#     G̃ from diagonal Λ only (no lab A = U G̃ Uᵀ)
#     V(w) = (d/(2 s0)) w² + ½ Σ_k G̃_kk φ_k(w)²,  φ = Uᵀ c(w)
#
#   julia eigenframe_diag_q_action.jl
#   julia eigenframe_diag_q_action.jl --d 10 --n1 400 --chi 400 --sa0 1 \
#       --kappa 0.5 --eps 1 --Ps 50,200,1000,5000

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, NLsolve, Printf

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])
parse_list(s) = parse.(Float64, split(s, ","))

const hermite_kernel_block = FCS2_VGA.hermite_kernel_block
const hermite_point_features = FCS2_VGA.hermite_point_features
const eigen_sym2 = FCS2_VGA.eigen_sym2
const compute_lambda1 = FCS2_VGA.compute_lambda1
const compute_lambda3 = FCS2_VGA.compute_lambda3

function hermite_eigenbasis(Q)
    evals0, U0 = eigen_sym2(Matrix(Q))
    he3 = abs.(U0[2, :])
    order = he3[1] <= he3[2] ? (1, 2) : (2, 1)
    evals = [evals0[order[1]], evals0[order[2]]]
    U = similar(U0)
    for (j, src) in enumerate(order)
        u1, u2 = U0[1, src], U0[2, src]
        dom = j == 1 ? u1 : u2
        (dom < 0) && ((u1, u2) = (-u1, -u2))
        U[1, j] = u1
        U[2, j] = u2
    end
    return evals, U
end

"""G̃ from diagonal Λ and ỹ=(0,ε). Null |λ|<tol·max|λ| → G̃=0. No lab G."""
function amp_diag(chi, n1, κ, P, a0, eps_task, Q; delta=1.0, rel_eig_tol=1e-2)
    chi_p = chi / a0
    ρp = κ / (a0 * P)
    evals, U = hermite_eigenbasis(Q)
    ytilde = [0.0, Float64(eps_task)]
    scale = a0 / (n1 * chi)
    d_scale = maximum(abs, evals)
    thr = rel_eig_tol * max(d_scale, Base.eps(typeof(d_scale)))
    Gdiag = zeros(2)
    for k in 1:2
        λ = evals[k]
        # Drop only silent near-null channels. Never drop a channel the teacher loads.
        if abs(ytilde[k]) == 0 && abs(λ) < thr
            continue
        end
        λsafe = abs(λ) < thr ? thr : max(λ, thr)
        v = ytilde[k] / (λsafe + ρp)
        Gdiag[k] = scale * (-(chi_p^2 * delta) * v^2 + chi_p / λsafe)
    end
    return Gdiag, evals, U, ytilde, ρp, thr
end

function make_V(d, s0, T, Gdiag, U)
    w -> begin
        c1, c3 = hermite_point_features(w, T)
        φ1 = U[1, 1] * c1 + U[2, 1] * c3
        φ2 = U[1, 2] * c1 + U[2, 2] * c3
        (d / s0) * 0.5 * w^2 + 0.5 * (Gdiag[1] * φ1^2 + Gdiag[2] * φ2^2)
    end
end

function learn_L3(evals, ρp, thr)
    # Cubic eigenchannel (index 2); use same floor as amp_diag for consistency.
    λ2 = evals[2]
    λsafe = abs(λ2) < thr ? thr : max(λ2, thr)
    return λsafe / (λsafe + ρp)
end

"""
Standard Laplace residuals in eigenframe:
  x = [lJ1, lJ3, σ, μ]
  r = [lJ1−λ11, lJ3−λ33, σ² V''−1, V']
with V from diagonal Λ and ỹ=(0,ε).
"""
function residuals_laplace_eigenframe(x, d, n1, chi, κ, P, a0, s0, eps_task, T)
    lJ1, lJ3, sigS, muW = x
    sigS = abs(sigS)
    muW = abs(muW)
    λ1 = compute_lambda1(muW, sigS, T)
    λ3 = compute_lambda3(muW, sigS, T)
    Q = hermite_kernel_block(muW, sigS, T)
    Gdiag, evals, U, _, _, _ = amp_diag(chi, n1, κ, P, a0, eps_task, Q)
    V = make_V(d, s0, T, Gdiag, U)
    dV = ForwardDiff.derivative(V, muW)
    d2V = ForwardDiff.derivative(w -> ForwardDiff.derivative(V, w), muW)
    return [lJ1 - λ1, lJ3 - λ3, sigS^2 * d2V - 1.0, dV], V, Gdiag, evals, U, Q
end

function solve_laplace(d, n1, chi, κ, P, a0, s0, eps_task; μ0=0.4, σ0=0.25, tol=1e-10)
    T = 1.0 + 2.0 * (d - 1.0) * s0 / d
    best = nothing
    for (μ0i, σ0i) in ((μ0, σ0), (0.1, 0.3), (0.8, 0.2), (1.5, 0.25),
                        (2.0, 0.3), (2.5, 0.25), (0.01, 0.5), (3.5, 0.3))
        λ1 = compute_lambda1(μ0i, σ0i, T)
        λ3 = compute_lambda3(μ0i, σ0i, T)
        x0 = [λ1, λ3, σ0i, μ0i]
        sol = try
            nlsolve(x -> residuals_laplace_eigenframe(
                    x, d, n1, chi, κ, P, a0, s0, eps_task, T)[1], x0;
                xtol=tol, ftol=tol, iterations=8000)
        catch
            nothing
        end
        sol === nothing && continue
        NLsolve.converged(sol) || continue
        lJ1, lJ3, sigS, muW = abs.(sol.zero)
        r, V, Gdiag, evals, U, Q = residuals_laplace_eigenframe(
            [lJ1, lJ3, sigS, muW], d, n1, chi, κ, P, a0, s0, eps_task, T)
        fn = norm(r)
        ρp = κ / (a0 * P)
        d_scale = maximum(abs, evals)
        thr = 1e-2 * max(d_scale, Base.eps(typeof(d_scale)))
        L3 = learn_L3(evals, ρp, thr)
        cand = (μ=muW, σ=sigS, L3=L3, V=V, Gdiag=Gdiag, evals=evals, U=U,
            Q=Matrix(Q), ρp=ρp, f_norm=fn, thr=thr)
        # μ=0 is always a critical point (V even); prefer a converged nonzero well
        # with smallest residual, then deepest V(μ).
        better = false
        if best === nothing
            better = true
        elseif (cand.μ > 1e-3) != (best.μ > 1e-3)
            better = cand.μ > best.μ
        elseif cand.f_norm < best.f_norm - 1e-8
            better = true
        elseif abs(cand.f_norm - best.f_norm) < 1e-8 && cand.V(cand.μ) < best.V(best.μ)
            better = true
        end
        better && (best = cand)
    end
    return best
end

function main()
    d = parse(Float64, argval("--d", "10"))
    n1 = parse(Float64, argval("--n1", "400"))
    chi = parse(Float64, argval("--chi", string(n1)))
    chi < 0 && (chi = n1)
    a0 = parse(Float64, argval("--sa0", "1"))
    κ = parse(Float64, argval("--kappa", "0.5"))
    eps_task = parse(Float64, argval("--eps", "1"))
    s0 = 1.0
    Ps = parse_list(argval("--Ps", "50,200,1000,5000"))
    wmax = parse(Float64, argval("--wmax", "8"))
    ngrid = parse(Int, argval("--ngrid", "1601"))
    out_dir = argval("--outdir",
        joinpath(@__DIR__, "action_h0_activation_plots", "EigenframeDiagQ"))
    mkpath(out_dir)

    @printf("Laplace saddle + eigenframe only: ỹ=(0,ε), G̃(Λ) diagonal, V'(μ)=0, σ=1/√V''\n")
    @printf("d=%g N=%g χ=%g sa0=%g κ=%g ε=%g\n", d, n1, chi, a0, κ, eps_task)

    sum_path = joinpath(out_dir, @sprintf("summary_laplace_d%d_N%d_chi%d.csv", Int(d), Int(n1), Int(chi)))
    open(sum_path, "w") do io
        println(io, "P,kappa,rho_p,L3,mu,sig,Vmu,lam_lin,lam_cub,G_lin,G_cub,Q11,Q13,Q33,f_norm")
        for P in Ps
            s = solve_laplace(d, n1, chi, κ, P, a0, s0, eps_task)
            if s === nothing
                @printf("P=%-8.0f  no Laplace root\n", P)
                continue
            end
            grid = range(0.0, wmax; length=ngrid)
            act_path = joinpath(out_dir, @sprintf("action_laplace_P%g_d%d.csv", P, Int(d)))
            open(act_path, "w") do aio
                println(aio, "w,V,prior,data")
                for w in grid
                    v = s.V(w)
                    prior = (d / s0) * 0.5 * w^2
                    @printf(aio, "%.10g,%.10g,%.10g,%.10g\n", w, v, prior, v - prior)
                end
            end
            @printf(io, "%.10g,%.10g,%.10g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.4g\n",
                P, κ, s.ρp, s.L3, s.μ, s.σ, s.V(s.μ), s.evals[1], s.evals[2],
                s.Gdiag[1], s.Gdiag[2], s.Q[1, 1], s.Q[1, 2], s.Q[2, 2], s.f_norm)
            @printf("P=%-8.0f  L3=%.4f  μ=%.4f σ=%.4f  V=%.4g  Λ=(%.4g, %.4g)  ||r||=%.2e\n",
                P, s.L3, s.μ, s.σ, s.V(s.μ), s.evals[1], s.evals[2], s.f_norm)
            flush(stdout)
        end
    end
    println("saved ", sum_path)
end

main()
