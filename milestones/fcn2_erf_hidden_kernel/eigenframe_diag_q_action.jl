#!/usr/bin/env julia
# Eigenframe action with diagonalized Q (user spec):
#
#   1. Build lab Hermite block Q from the current well mixture.
#   2. Diagonalize Q = U Λ Uᵀ (col 1 ≈ linear, col 2 ≈ cubic).
#   3. Teacher in that frame: ỹ = (0, ε).
#   4. Amplitudes use only diagonal Λ — no lab A = U G̃ Uᵀ:
#        ρ' = κ/(a0 P), χ' = χ/a0,
#        ṽ_k = ỹ_k / (λ_k + ρ'),
#        G̃_kk = [a0/(n1 χ)] [ −(χ')² ṽ_k² + χ'/λ_k ],  G̃_12 = 0.
#      Near-null eigenvalues (|λ| < tol · max|λ|) get G̃_kk = 0
#      (same null-channel rule as offdiag VGA).
#   5. V(w) = (d/(2 s0)) w² + ½ Σ_k G̃_kk φ_k(w)²,  φ = Uᵀ c(w).
#   6. Saddle point: locate wells of V; take the global minimizer only
#      (hard assignment — no Boltzmann / Laplace mixture weights on π);
#      set Q from that well's (μ,σ); iterate to self-consistency.
#
#   julia eigenframe_diag_q_action.jl
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

"""Diagonal G̃ from Λ and ỹ=(0,ε). Null channels → G̃=0. No lab G."""
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
        abs(λ) < thr && continue
        λsafe = max(λ, thr)   # keep Onsager finite on barely-positive eigs
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

function learn_L3(evals, ρp, eps, thr)
    λ2 = evals[2]
    abs(λ2) < thr && return 0.0
    λsafe = max(λ2, thr)
    return (λsafe / (λsafe + ρp))
end

seed_Q(μ, σ, T) = Matrix(hermite_kernel_block(μ, σ, T))

"""Local minima of V on w≥0: (μ, σ, V, multiplicity=2 for ±)."""
function find_wells(V; wmax=12.0, n=8001)
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
        !isfinite(σ) && continue
        σ = clamp(σ, 1e-4, 5.0)
        # skip degenerate twin of μ≈0 counted twice
        any(abs(w.μ - μ) < 0.5 * σ for w in wells) && continue
        push!(wells, (μ=μ, σ=σ, V=V(μ), m=μ < 1e-3 ? 1.0 : 2.0))
    end
    return wells, grid, Vs
end

"""Saddle-point Q: global minimizer of V only (hard assignment)."""
function saddle_Q(wells, T)
    isempty(wells) && return nothing
    j = argmin([w.V for w in wells])
    w = wells[j]
    return seed_Q(w.μ, w.σ, T), w
end

"""Self-consistent eigenframe saddle-point solve at fixed P."""
function solve_point(d, n1, chi, κ, P, a0, s0, eps; iters=80, damp=0.35, wmax=12.0)
    T = 1.0 + 2.0 * (d - 1.0) * s0 / d
    # Seeds that can land on bulk or outer Hamiltonian minima.
    seeds = [
        seed_Q(0.4, 0.25, T),
        seed_Q(2.5, 0.25, T),
        0.5 * seed_Q(0.3, 0.25, T) + 0.5 * seed_Q(2.5, 0.25, T),
    ]
    best = nothing
    for Q0 in seeds
        Q = copy(Q0)
        last = nothing
        for it in 1:iters
            Gdiag, evals, U, ytilde, ρp, thr = amp_diag(chi, n1, κ, P, a0, eps, Q)
            V = make_V(d, s0, T, Gdiag, U)
            wells, grid, Vs = find_wells(V; wmax=wmax)
            if isempty(wells)
                i = argmin(Vs)
                μ = float(grid[i])
                wells = [(μ=μ, σ=0.2, V=Vs[i], m=μ < 1e-3 ? 1.0 : 2.0)]
            end
            got = saddle_Q(wells, T)
            got === nothing && break
            Qn, win = got
            δ = norm(Qn - Q) / max(norm(Q), 1e-30)
            Q = (1 - damp) * Q + damp * Qn
            L3 = learn_L3(evals, ρp, eps, thr)
            last = (Q=Matrix(Q), Gdiag=Gdiag, evals=evals, U=U, ρp=ρp, L3=L3, V=V,
                wells=wells, win=win, po=(win.μ > 0.8 ? 1.0 : 0.0),
                grid=grid, Vs=Vs, ytilde=ytilde, thr=thr, iters=it)
            δ < 1e-9 && break
        end
        last === nothing && continue
        # Prefer the fixed point whose occupied well has lower V.
        if best === nothing || last.win.V < best.win.V
            best = last
        end
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
    eps = parse(Float64, argval("--eps", "1"))
    s0 = 1.0
    Ps = parse_list(argval("--Ps", "50,200,1000,5000"))
    wmax = parse(Float64, argval("--wmax", "10"))
    out_dir = argval("--outdir",
        joinpath(@__DIR__, "action_h0_activation_plots", "EigenframeDiagQ"))
    mkpath(out_dir)

    @printf("Eigenframe diag-Q SADDLE action: ỹ=(0,ε), G̃(Λ) only, Q from global min of V\n")
    @printf("d=%g N=%g χ=%g sa0=%g κ=%g ε=%g\n", d, n1, chi, a0, κ, eps)

    sum_path = joinpath(out_dir, @sprintf("summary_saddle_d%d_N%d_chi%d.csv", Int(d), Int(n1), Int(chi)))
    open(sum_path, "w") do io
        println(io, "P,kappa,rho_p,L3,saddle_outer,nwells,mu_saddle,V_saddle,mu_list,V_list,lam_lin,lam_cub,G_lin,G_cub,Q11,Q13,Q33,iters")
        for P in Ps
            s = solve_point(d, n1, chi, κ, P, a0, s0, eps; wmax=wmax)
            if s === nothing
                @printf("P=%-8.0f  failed\n", P)
                continue
            end
            act_path = joinpath(out_dir, @sprintf("action_saddle_P%g_d%d.csv", P, Int(d)))
            open(act_path, "w") do aio
                println(aio, "w,V,prior,data")
                for (w, v) in zip(s.grid, s.Vs)
                    prior = (d / s0) * 0.5 * w^2
                    @printf(aio, "%.10g,%.10g,%.10g,%.10g\n", w, v, prior, v - prior)
                end
            end
            mus = join([@sprintf("%.4g", w.μ) for w in s.wells], ";")
            Vlist = join([@sprintf("%.4g", w.V) for w in s.wells], ";")
            @printf(io, "%.10g,%.10g,%.10g,%.8g,%d,%d,%.8g,%.8g,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n",
                P, κ, s.ρp, s.L3, Int(s.po), length(s.wells), s.win.μ, s.win.V, mus, Vlist,
                s.evals[1], s.evals[2], s.Gdiag[1], s.Gdiag[2],
                s.Q[1, 1], s.Q[1, 2], s.Q[2, 2], s.iters)
            wstr = join([@sprintf("μ=%.3f V=%.4g%s", w.μ, w.V, abs(w.μ - s.win.μ) < 1e-6 ? "*" : "")
                for w in s.wells], " | ")
            @printf("P=%-8.0f  L3=%.4f  saddle μ=%.3f V=%.4g  Λ=(%.4g, %.4g)  wells: %s\n",
                P, s.L3, s.win.μ, s.win.V, s.evals[1], s.evals[2], wstr)
            flush(stdout)
        end
    end
    println("saved ", sum_path)
end

main()
