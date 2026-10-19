#!/usr/bin/env julia
# Laplace multiwell with mixture saddle weights, eigenframe basis only:
#
#   Q = U Λ Uᵀ
#   ỹ = (0, ε)
#   G̃ from diagonal Λ (no lab A = U G̃ Uᵀ)
#   V(w) = (d/(2 s0)) w² + ½ Σ_k G̃_kk φ_k(w)²,  φ = Uᵀ c(w)
#
#   Wells of V → Laplace mixture weights
#     π_k ∝ m_k σ_k exp(−V_k)   (m_k = 2 for ±μ pair, 1 for μ=0)
#   Q = Σ_k π_k Q(μ_k, σ_k); iterate.
#
#   Action dumped on w ∈ [−wmax, wmax] so both ± wells are visible.
#
#   julia eigenframe_diag_q_action.jl --d 10 --n1 400 --chi 400 --sa0 1 \
#       --kappa 0.5 --eps 1 --Ps 50,200,1000,5000

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, Printf

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])
parse_list(s) = parse.(Float64, split(s, ","))

const hermite_kernel_block = FCS2_VGA.hermite_kernel_block
const hermite_point_features = FCS2_VGA.hermite_point_features
const eigen_sym2 = FCS2_VGA.eigen_sym2

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
    λ2 = evals[2]
    λsafe = abs(λ2) < thr ? thr : max(λ2, thr)
    return λsafe / (λsafe + ρp)
end

seed_Q(μ, σ, T) = Matrix(hermite_kernel_block(μ, σ, T))

"""Local minima of V on w ≥ 0 (each μ>0 stands for the ± pair)."""
function find_wells(V; wmax=12.0, n=4001)
    grid = collect(range(0.0, wmax; length=n))
    Vs = [V(w) for w in grid]
    wells = NamedTuple[]
    for i in 2:(n - 1)
        (Vs[i] < Vs[i - 1] && Vs[i] <= Vs[i + 1]) || continue
        w0 = grid[i]
        h = max(grid[2] - grid[1], 1e-4)
        vm, v0, vp = V(w0 - h), V(w0), V(w0 + h)
        a = (vm - 2v0 + vp) / (2h^2)
        b = (vp - vm) / (2h)
        μ = (a > 1e-12) ? clamp(w0 - b / (2a), 0.0, wmax) : w0
        σ = (a > 1e-12) ? 1 / sqrt(2a) : NaN
        (!isfinite(σ) || σ <= 0) && continue
        σ = clamp(σ, 1e-4, 5.0)
        any(abs(w.μ - μ) < 0.5 * min(σ, w.σ) for w in wells) && continue
        m = μ < 1e-3 ? 1.0 : 2.0   # ± pair
        push!(wells, (μ=μ, σ=σ, V=V(μ), m=m))
    end
    return wells
end

"""Laplace mixture saddle weights → Q."""
function mixture_Q(wells, T)
    isempty(wells) && return nothing
    logZ = [log(w.m * w.σ) - w.V for w in wells]
    lmax = maximum(logZ)
    wt = exp.(logZ .- lmax)
    wt ./= sum(wt)
    Q = sum(wt[k] * seed_Q(wells[k].μ, wells[k].σ, T) for k in eachindex(wells))
    return Q, wt
end

function solve_point(d, n1, chi, κ, P, a0, s0, eps_task; iters=80, damp=0.35, wmax=12.0)
    T = 1.0 + 2.0 * (d - 1.0) * s0 / d
    Q = 0.7 * seed_Q(0.3, 0.25, T) + 0.3 * seed_Q(2.5, 0.25, T)
    last = nothing
    for it in 1:iters
        Gdiag, evals, U, ytilde, ρp, thr = amp_diag(chi, n1, κ, P, a0, eps_task, Q)
        V = make_V(d, s0, T, Gdiag, U)
        wells = find_wells(V; wmax=wmax)
        if isempty(wells)
            # fallback: grid min on |w|
            g = collect(range(0.0, wmax; length=2001))
            Vs = [V(w) for w in g]
            i = argmin(Vs)
            μ = float(g[i])
            wells = [(μ=μ, σ=0.2, V=Vs[i], m=μ < 1e-3 ? 1.0 : 2.0)]
        end
        got = mixture_Q(wells, T)
        got === nothing && break
        Qn, wt = got
        δ = norm(Qn - Q) / max(norm(Q), 1e-30)
        Q = (1 - damp) * Q + damp * Qn
        L3 = learn_L3(evals, ρp, thr)
        po = sum(wt[k] for k in eachindex(wells) if wells[k].μ > 0.8; init=0.0)
        last = (Q=Matrix(Q), Gdiag=Gdiag, evals=evals, U=U, ρp=ρp, L3=L3, V=V,
            wells=wells, wt=wt, po=po, ytilde=ytilde, thr=thr, iters=it)
        δ < 1e-9 && break
    end
    return last
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
    ngrid = parse(Int, argval("--ngrid", "3201"))
    out_dir = argval("--outdir",
        joinpath(@__DIR__, "action_h0_activation_plots", "EigenframeDiagQ"))
    mkpath(out_dir)

    @printf("Eigenframe + Laplace mixture saddle weights: ỹ=(0,ε), G̃(Λ) only\n")
    @printf("d=%g N=%g χ=%g sa0=%g κ=%g ε=%g; action on w∈[−%g,%g]\n",
        d, n1, chi, a0, κ, eps_task, wmax, wmax)

    sum_path = joinpath(out_dir, @sprintf("summary_mix_d%d_N%d_chi%d.csv", Int(d), Int(n1), Int(chi)))
    open(sum_path, "w") do io
        println(io, "P,kappa,rho_p,L3,pi_out,nwells,mu_list,V_list,pi_list,lam_lin,lam_cub,G_lin,G_cub,Q11,Q13,Q33,iters")
        for P in Ps
            s = solve_point(d, n1, chi, κ, P, a0, s0, eps_task; wmax=wmax)
            if s === nothing
                @printf("P=%-8.0f  failed\n", P)
                continue
            end
            # Full line so both ± wells are visible
            grid = range(-wmax, wmax; length=ngrid)
            act_path = joinpath(out_dir, @sprintf("action_mix_P%g_d%d.csv", P, Int(d)))
            open(act_path, "w") do aio
                println(aio, "w,V,prior,data")
                for w in grid
                    v = s.V(w)
                    prior = (d / s0) * 0.5 * w^2
                    @printf(aio, "%.10g,%.10g,%.10g,%.10g\n", w, v, prior, v - prior)
                end
            end
            mus = join([@sprintf("%.4g", w.μ) for w in s.wells], ";")
            Vs = join([@sprintf("%.4g", w.V) for w in s.wells], ";")
            ps = join([@sprintf("%.4g", p) for p in s.wt], ";")
            @printf(io, "%.10g,%.10g,%.10g,%.8g,%.6g,%d,%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n",
                P, κ, s.ρp, s.L3, s.po, length(s.wells), mus, Vs, ps,
                s.evals[1], s.evals[2], s.Gdiag[1], s.Gdiag[2],
                s.Q[1, 1], s.Q[1, 2], s.Q[2, 2], s.iters)
            wstr = join([@sprintf("μ=±%.3f V=%.4g π=%.3g", w.μ, w.V, s.wt[k])
                for (k, w) in enumerate(s.wells)], " | ")
            @printf("P=%-8.0f  L3=%.4f  π_out≈%.3f  %s\n", P, s.L3, s.po, wstr)
            flush(stdout)
        end
    end
    println("saved ", sum_path)
end

main()
