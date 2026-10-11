# Pure-cubic multi-well matrix Laplace (He1–He3 basis, teacher y = (0, ε)).
#
# Same V and mixture as multiwell_laplace.jl, but the target in G, free energy,
# and learnability is y = (0, ε) instead of (1, ε). The linear channel stays in
# the 2×2 kernel (Onsager / induced He1); it just has no teacher weight.
#
# Invariant action factor (steepwell ray):
#     Finv = 2 P² sa0 / (π κ² d N)
# A cubic-weighted proxy is Finv3 = ε² Finv (only the cubic channel is driven).
#
#   julia tmp_multiwell_pure_cubic.jl --scan --d 10 --n1 400 --chi 400 --sa0 1 \
#       --eps 1 --Ps 50,100,200,500,1000,2000,5000,10000 --kappas 0.5
#
#   # Sweep Finv at steepwell-like χ=1 (P solved from Finv; κ, sa0, d, N fixed):
#   julia tmp_multiwell_pure_cubic.jl --F-scan --d 10 --n1 1000 --chi 1 --sa0 0.03 \
#       --eps 1 --kappa 0.003 --Fs 1e2,1e3,1e4,3e4,1e5,3e5,1e6,3e6,1e7
#
#   # Steepwell-exponent ray at fixed Finv (χ=1 default; --chi -1 ⇒ χ=N):
#   julia tmp_multiwell_pure_cubic.jl --ray-scan --Finv 200 --eps 1 --chi -1
#   julia tmp_multiwell_pure_cubic.jl --pi-scan --Finv 200 --beta 10 --chi -1
#   julia tmp_multiwell_pure_cubic.jl --ray-F-scan --beta 10 --chi -1 --eps 1
#
# Reports dV_raw = V(μ_out) − V(μ_in). Outer wells are Hamiltonian global mins
# when dV_raw < 0. With χ=1, outer dies before π=O(1); with χ=N, Laplace
# self-consistency reaches π_out ~ O(1) inside the Finv~100–300 window.

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, Printf

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1])
parse_list(s) = parse.(Float64, split(s, ","))

struct Cfg
    d::Float64; n1::Float64; chi::Float64; a0::Float64; eps::Float64; s0::Float64; T::Float64
end

"""Teacher in the He1–He3 block: linear off, cubic = ε."""
teacher_y(c::Cfg) = [0.0, c.eps]

"""Same formula as FCS2_VGA.laplace_amplitude_matrix, but y = (0, ε)."""
function amp(c::Cfg, Q, P, κ)
    chi_p = c.chi / c.a0
    ρp = κ / (c.a0 * P)
    Qm = Matrix(Q)
    y = teacher_y(c)
    qscale = abs(Qm[1, 1]) + abs(Qm[2, 2])
    Qreg = Qm + (1e-10 * (1 + qscale)) * I(2)
    v = (Qreg + ρp * I(2)) \ y
    G = -(chi_p^2) * (v * v') + chi_p * inv(Qreg)
    return (c.a0 / (c.n1 * c.chi)) * G
end

function potential(c::Cfg, G)
    w -> begin
        c1, c3 = FCS2_VGA.hermite_point_features(w, c.T)
        (c.d / c.s0) * 0.5 * w^2 + 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
    end
end

const WGRID = range(0.0, 40.0, length=16001)

"""Local minima of V on w ≥ 0: (μ, σ, V(μ), multiplicity)."""
function wells(V)
    d1 = w -> ForwardDiff.derivative(V, w)
    d2 = w -> ForwardDiff.derivative(d1, w)
    v = V.(WGRID)
    out = Tuple{Float64,Float64,Float64,Int}[]
    if v[1] < v[2] && d2(0.0) > 0
        push!(out, (0.0, 1 / sqrt(d2(0.0)), v[1], 1))
    end
    for j in 2:length(WGRID)-1
        (v[j] < v[j-1] && v[j] <= v[j+1]) || continue
        w = WGRID[j]
        for _ in 1:30
            h = d2(w)
            h > 0 || break
            w -= d1(w) / h
        end
        h = d2(w)
        (h > 0 && w > 0 && abs(w - WGRID[j]) < 0.05) || (w = WGRID[j]; h = d2(w))
        h > 0 || continue
        push!(out, (w, 1 / sqrt(h), V(w), 2))
    end
    out
end

function mixture(c::Cfg, ws)
    logZk = [log(m * s) - Vm for (μ, s, Vm, m) in ws]
    lmax = maximum(logZk)
    wt = exp.(logZk .- lmax)
    logZ1 = lmax + log(sum(wt)) + 0.5 * log(2pi)
    wt ./= sum(wt)
    Q = sum(wt[k] * Matrix(FCS2_VGA.hermite_kernel_block(ws[k][1], ws[k][2], c.T)) for k in eachindex(ws))
    Q, wt, logZ1
end

function free_energy(c::Cfg, Q, G, logZ1, P, κ)
    χp = c.chi / c.a0
    ρp = κ / (c.a0 * P)
    y = teacher_y(c)
    c.n1 * (-logZ1 - 0.5 * tr(G * Q)) + 0.5χp * dot(y, (Q + ρp * I) \ y) + 0.5 * logdet(Q)
end

"""Returns (f1, L3). f1 is induced He1 (teacher has none); L3 = f3/ε."""
function learnabilities(c::Cfg, Q, P, κ)
    ρp = κ / (c.a0 * P)
    y = teacher_y(c)
    f = Q * ((Q + ρp * I) \ y)
    f[1], f[2] / c.eps
end

function solve(c::Cfg, Q0, P, κ; iters=4000, η=0.2, tol=1e-9)
    Q = copy(Q0)
    for it in 1:iters
        G = amp(c, Q, P, κ)
        ws = wells(potential(c, G))
        isempty(ws) && return nothing
        Qn, _, _ = mixture(c, ws)
        δ = norm(Qn - Q) / max(norm(Q), 1e-30)
        Q = (1 - η) * Q + η * Qn
        if δ < tol
            G = amp(c, Q, P, κ)
            ws = wells(potential(c, G))
            Qf, π, logZ1 = mixture(c, ws)
            return (Q=Qf, G=G, ws=ws, π=π, F=free_energy(c, Qf, G, logZ1, P, κ), iters=it)
        end
    end
    nothing
end

seed_Q(c::Cfg, μ, σ) = Matrix(FCS2_VGA.hermite_kernel_block(μ, σ, c.T))

"""Fixed outer fraction π_out; innermost well = bulk/center, outermost = outer."""
function solve_fixed_pi(c::Cfg, po, Q0, P, κ; iters=3000, η=0.3, tol=1e-10)
    Q = copy(Q0)
    for it in 1:iters
        G = amp(c, Q, P, κ)
        ws = wells(potential(c, G))
        isempty(ws) && return nothing
        (po > 0 && length(ws) < 2) && return nothing
        win, wout = ws[1], ws[end]
        Qn = (1 - po) * seed_Q(c, win[1], win[2]) + po * seed_Q(c, wout[1], wout[2])
        δ = norm(Qn - Q) / max(norm(Q), 1e-30)
        Q = (1 - η) * Q + η * Qn
        δ < tol || continue
        G = amp(c, Q, P, κ)
        ws = wells(potential(c, G))
        (po > 0 && length(ws) < 2) && return nothing
        win, wout = ws[1], ws[end]
        Q = (1 - po) * seed_Q(c, win[1], win[2]) + po * seed_Q(c, wout[1], wout[2])
        πs = po > 0 ? [1 - po, po] : [1.0]
        sel = po > 0 ? [win, wout] : [win]
        χp = c.chi / c.a0
        ρp = κ / (c.a0 * P)
        y = teacher_y(c)
        e = sum(πs[k] * (sel[k][3] - log(sel[k][4] * sel[k][2] * sqrt(2pi))) for k in eachindex(sel))
        s = sum(p * log(p) for p in πs if p > 0)
        F = c.n1 * (e + s - 0.5 * tr(G * Q)) + 0.5χp * dot(y, (Q + ρp * I) \ y) + 0.5 * logdet(Q)
        f1, L3 = learnabilities(c, Q, P, κ)
        dV = wout[3] - win[3] - log(wout[2] / win[2])
        return (Q=Q, F=F, win=win, wout=wout, f1=f1, L3=L3, dV=dV, nwells=length(ws),
            dV_raw=wout[3] - win[3])
    end
    nothing
end

"""Walk π from 0→1 with warm Q; return densest grid of converged (po, s).

When a step fails, refine between last-good and failed π (binary) so π_max is tight.
Also try a few cold outer-heavy seeds before giving up.
"""
function walk_pi(c::Cfg, P, κ; pos=nothing)
    pos === nothing && (pos = vcat(0.0, 10 .^ range(-5, -0.7; length=28),
        range(0.25, 0.98; length=30), 0.99, 0.995, 0.999))
    grid = Tuple{Float64,Any}[]
    Qprev = seed_Q(c, 0.0, 0.5)
    failed_po = nothing
    for po in pos
        s = nothing
        seeds = Any[Qprev, seed_Q(c, 0.0, 0.5), seed_Q(c, 2.0, 0.3),
            seed_Q(c, 3.0, 0.25), seed_Q(c, 4.0, 0.2),
            (1 - po) * seed_Q(c, 0.0, 0.4) + po * seed_Q(c, 2.5, 0.25)]
        for Q0 in seeds
            s = solve_fixed_pi(c, po, Q0, P, κ)
            s === nothing || break
        end
        if s === nothing
            failed_po = po
            break
        end
        push!(grid, (po, s))
        Qprev = s.Q
    end
    # binary refine up to the failure edge
    if failed_po !== nothing && !isempty(grid)
        lo, hi, Qs = grid[end][1], failed_po, grid[end][2].Q
        for _ in 1:25
            m = 0.5 * (lo + hi)
            t = solve_fixed_pi(c, m, Qs, P, κ)
            if t === nothing
                hi = m
            else
                lo, Qs = m, t.Q
                push!(grid, (m, t))
            end
            hi - lo < 1e-5 && break
        end
        sort!(grid; by=first)
    end
    grid
end

const PO_GRID = vcat(0.0, 10 .^ range(-5, -1, length=17), range(0.12, 1.0, length=23))

"""Steepwell invariant action factor Finv = 2 P² sa0 / (π κ² d N)."""
invariant_F(P, κ, c::Cfg) = 2 * P^2 * c.a0 / (π * κ^2 * c.d * c.n1)

"""P that realizes Finv at fixed (d, N, sa0, κ)."""
P_from_Finv(Finv, κ, c::Cfg) = κ * sqrt(π * c.d * c.n1 * Finv / (2 * c.a0))

function make_cfg()
    d = parse(Float64, argval("--d", "10"))
    s0 = 1.0
    Cfg(d, parse(Float64, argval("--n1", "400")), parse(Float64, argval("--chi", "400")),
        parse(Float64, argval("--sa0", "1")), parse(Float64, argval("--eps", "1")), s0,
        1.0 + 2.0 * (d - 1.0) * s0 / d)
end

"""Lowest free-energy π_out root at fixed (P, κ). Also returns F(π=0) and F(π→1) gap."""
function equilibrium_at(c::Cfg, P, κ; Q_warm=nothing)
    Qprev = Q_warm === nothing ? seed_Q(c, 0.0, 0.5) : Q_warm
    best = nothing
    grid = Tuple{Float64,Any}[]
    F0 = NaN
    F1 = NaN
    for po in PO_GRID
        s = nothing
        Q0s = Any[Qprev, seed_Q(c, 0.0, 0.5), seed_Q(c, 2.0, 0.3),
            0.5 * seed_Q(c, 0.0, 0.5) + 0.5 * seed_Q(c, 2.0, 0.3)]
        Q_warm === nothing || push!(Q0s, Q_warm)
        for Q0 in Q0s
            s = solve_fixed_pi(c, po, Q0, P, κ)
            s === nothing || break
        end
        s === nothing && continue
        Qprev = s.Q
        push!(grid, (po, s))
        (best === nothing || s.F < best[2].F) && (best = (po, s))
        po == 0.0 && (F0 = s.F)
        po >= 0.9 && (F1 = s.F)  # last high-π point that converged
    end
    mins = Tuple{Float64,Any}[]
    if !isempty(grid) && grid[1][1] == 0.0 && (grid[1][2].nwells < 2 || grid[1][2].dV >= 0)
        push!(mins, grid[1])
    end
    for j in 1:length(grid)-1
        (pa, sa), (pb, sb) = grid[j], grid[j+1]
        (sa.dV < 0 && sb.dV >= 0) || continue
        lo, hi, Qs, sm = pa, pb, sa.Q, sa
        for _ in 1:40
            m = 0.5 * (lo + hi)
            t = solve_fixed_pi(c, m, Qs, P, κ)
            t === nothing && break
            sm, Qs = t, t.Q
            t.dV < 0 ? (lo = m) : (hi = m)
            hi - lo < 1e-6 * (1 + hi) && break
        end
        push!(mins, (0.5 * (lo + hi), sm))
    end
    isempty(mins) && best !== nothing && push!(mins, best)
    isempty(mins) && return nothing
    po, s = mins[argmin([m[2].F for m in mins])]
    ΔF_ends = (isfinite(F0) && isfinite(F1)) ? (F1 - F0) : NaN
    (po=po, s=s, F0=F0, F1=F1, dF_outer_minus_inner=ΔF_ends, grid=grid)
end

function scan_main()
    c = make_cfg()
    Ps = parse_list(argval("--Ps", "50,100,200,500,1000,2000,5000,10000"))
    κs = parse_list(argval("--kappas", "0.5"))
    length(κs) == 1 && (κs = fill(κs[1], length(Ps)))
    out = argval("--out", "tmp_pure_cubic_scan_d$(Int(c.d)).csv")
    eqout = replace(out, ".csv" => "_equilibrium.csv")
    eqio = open(eqout, "w")
    println(eqio, "P,kappa,Finv,Finv3,pi_out,Fmf,f1,L3,mu_in,sig_in,V_in,mu_out,sig_out,V_out,dV,dV_raw,dF_ends,nwells")
    @printf("pure cubic y=(0, ε=%.4g); d=%g N=%g χ=%g sa0=%g\n", c.eps, c.d, c.n1, c.chi, c.a0)
    open(out, "w") do io
        println(io, "P,kappa,Finv,pi_out,Fmf,f1,L3,mu_in,sig_in,V_in,mu_out,sig_out,V_out,dV,dV_raw,nwells,Q11,Q13,Q33")
        Qeq_prev = nothing
        for (P, κ) in zip(Ps, κs)
            Finv = invariant_F(P, κ, c)
            eq = equilibrium_at(c, P, κ; Q_warm=Qeq_prev)
            if eq === nothing
                @printf("d=%g P=%-7.0f Finv=%.4g nothing converged\n", c.d, P, Finv)
                continue
            end
            po, s = eq.po, eq.s
            Qeq_prev = s.Q
            for (poi, si) in eq.grid
                @printf(io, "%.10g,%.10g,%.10g,%.6g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g\n",
                    P, κ, Finv, poi, si.F, si.f1, si.L3, si.win[1], si.win[2], si.win[3],
                    si.wout[1], si.wout[2], si.wout[3], si.dV, si.dV_raw, si.nwells,
                    si.Q[1, 1], si.Q[1, 2], si.Q[2, 2])
            end
            @printf(eqio, "%.10g,%.10g,%.10g,%.10g,%.6g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                P, κ, Finv, Finv * c.eps^2, po, s.F, s.f1, s.L3, s.win[1], s.win[2], s.win[3],
                s.wout[1], s.wout[2], s.wout[3], s.dV, s.dV_raw, eq.dF_outer_minus_inner, s.nwells)
            flush(eqio)
            mark = s.dV_raw < 0 ? "OUTER_HAMILTONIAN_MIN" : "inner_hamiltonian_min"
            @printf("d=%g P=%-7.0f Finv=%.4g Finv3=%.4g π_out=%.4g  L3=%.4f  dV_raw=%.4g  ΔF(π1−π0)=%.4g  %s\n",
                c.d, P, Finv, Finv * c.eps^2, po, s.L3, s.dV_raw, eq.dF_outer_minus_inner, mark)
            flush(stdout)
        end
    end
    close(eqio)
    println("saved ", out, " and ", eqout)
end

"""Sweep the steepwell invariant Finv at fixed (d, N, χ, sa0, κ); P = P(Finv)."""
function f_scan_main()
    # Steepwell-like defaults: χ=1, small sa0, cold κ (override with flags).
    d = parse(Float64, argval("--d", "10"))
    n1 = parse(Float64, argval("--n1", "1000"))
    chi = parse(Float64, argval("--chi", "1"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps = parse(Float64, argval("--eps", "1"))
    s0 = 1.0
    c = Cfg(d, n1, chi, sa0, eps, s0, 1.0 + 2.0 * (d - 1.0) * s0 / d)
    κ = parse(Float64, argval("--kappa", "0.003"))
    Fs = parse_list(argval("--Fs", "100,300,1000,3000,10000,30000,46000,1e5,3e5,1e6,3e6,1e7"))
    out = argval("--out", "tmp_pure_cubic_Fscan_d$(Int(c.d)).csv")
    @printf("F-scan pure cubic y=(0, ε=%.4g); d=%g N=%g χ=%g sa0=%g κ=%g\n",
        c.eps, c.d, c.n1, c.chi, c.a0, κ)
    @printf("Finv = 2 P² sa0 / (π κ² d N); Finv3 = ε² Finv; steepwell ray sits near Finv≈4.6e4\n")
    open(out, "w") do io
        println(io, "Finv,Finv3,P,kappa,pi_out,Fmf,F0,F1,dF_ends,f1,L3,mu_in,sig_in,V_in,mu_out,sig_out,V_out,dV_raw,nwells")
        Qeq_prev = nothing
        for Finv in Fs
            P = P_from_Finv(Finv, κ, c)
            eq = equilibrium_at(c, P, κ; Q_warm=Qeq_prev)
            if eq === nothing
                @printf(io, "%.10g,%.10g,%.10g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n",
                    Finv, Finv * c.eps^2, P, κ)
                @printf("Finv=%.4g P=%.1f  nothing converged\n", Finv, P)
                continue
            end
            po, s = eq.po, eq.s
            Qeq_prev = s.Q
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.6g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                Finv, Finv * c.eps^2, P, κ, po, s.F, eq.F0, eq.F1, eq.dF_outer_minus_inner,
                s.f1, s.L3, s.win[1], s.win[2], s.win[3], s.wout[1], s.wout[2], s.wout[3],
                s.dV_raw, s.nwells)
            mark = s.dV_raw < 0 ? "OUTER_HAMILTONIAN_MIN" : "inner_hamiltonian_min"
            @printf("Finv=%.4g Finv3=%.4g P=%-8.1f π_out=%.4g L3=%.4f dV_raw=%+.4g  ΔF(π1−π0)=%+.4g  in μ=%.3f V=%.4g | out μ=%.3f V=%.4g  %s\n",
                Finv, Finv * c.eps^2, P, po, s.L3, s.dV_raw, eq.dF_outer_minus_inner,
                s.win[1], s.win[3], s.wout[1], s.wout[3], mark)
            flush(stdout)
        end
    end
    println("saved ", out)
end

function main()
    c = make_cfg()
    Ps = parse_list(argval("--Ps", "50,100,500,1000"))
    κs = parse_list(argval("--kappas", "0.5"))
    length(κs) == 1 && (κs = fill(κs[1], length(Ps)))
    out = argval("--out", "tmp_pure_cubic_d$(Int(c.d)).csv")
    @printf("pure cubic y=(0, ε=%.4g); d=%g N=%g χ=%g sa0=%g\n", c.eps, c.d, c.n1, c.chi, c.a0)
    open(out, "w") do io
        println(io, "P,kappa,sol,F,f1,L3,nwells,mu1,sig1,V1,pi1,mu2,sig2,V2,pi2,mu3,sig3,V3,pi3,Q11,Q13,Q33,equilibrium")
        for (P, κ) in zip(Ps, κs)
            seeds = [seed_Q(c, μ, σ) for (μ, σ) in (
                (0.0, 0.5), (0.0, 1.0), (0.4, 0.1), (1.0, 0.3), (2.0, 0.3), (3.0, 0.3), (4.0, 0.4))]
            for (μi, σi) in ((0.0, 0.5), (2.0, 0.3))
                push!(seeds, 0.5 * seed_Q(c, μi, σi) + 0.5 * seed_Q(c, 2.5, 0.3))
            end
            sols = []
            for Q0 in seeds
                s = solve(c, Q0, P, κ)
                s === nothing && continue
                any(t -> norm(t.Q - s.Q) < 1e-6 * max(norm(s.Q), 1e-30), sols) && continue
                push!(sols, s)
            end
            sort!(sols; by=s -> s.F)
            for (k, s) in enumerate(sols)
                f1, L3 = learnabilities(c, s.Q, P, κ)
                cols = String[]
                for j in 1:3
                    if j <= length(s.ws)
                        push!(cols, @sprintf("%.6g,%.6g,%.6g,%.6g", s.ws[j][1], s.ws[j][2], s.ws[j][3], s.π[j]))
                    else
                        push!(cols, "NaN,NaN,NaN,NaN")
                    end
                end
                @printf(io, "%.10g,%.10g,%d,%.10g,%.6g,%.6g,%d,%s,%.6g,%.6g,%.6g,%d\n",
                    P, κ, k, s.F, f1, L3, length(s.ws), join(cols, ","),
                    s.Q[1, 1], s.Q[1, 2], s.Q[2, 2], k == 1)
                wstr = join([@sprintf("μ=%.3f σ=%.3f V=%.4g π=%.3g",
                    s.ws[j][1], s.ws[j][2], s.ws[j][3], s.π[j]) for j in eachindex(s.ws)], " | ")
                @printf("d=%g P=%-7.0f sol %d  F=%.6g  L3=%.4f f1=%.4f  %s\n", c.d, P, k, s.F, L3, f1, wstr)
            end
            isempty(sols) && @printf("d=%g P=%-7.0f no converged fixed point\n", c.d, P)
            flush(stdout)
        end
    end
    println("saved ", out)
end

"""Steepwell-exponent ray at a chosen Finv (not the steepwell value 4.6e4).

Same exponents as red_robin_alpha_beta_invariant_steepwell.py:
  d~β^{1/2}, N~β^{1/4}, sa0~β^{-3/4}, α=α0 β^{5/4}, P~α^{3/4}, κ~κ0(α/β)^{3/4}
so Finv is flat in β. Base (β=1) is retargeted so Finv = --Finv (default 200),
which is inside the pure-cubic outer-Hamiltonian window.

Also reports endpoint Hamiltonians at π=0 and π≈1 (self-consistent Q at that π),
and the linear undistributed ratio R_lin = 2 Finv / τ.
"""
function ray_scan_main()
    Finv_target = parse(Float64, argval("--Finv", "200"))
    eps_task = parse(Float64, argval("--eps", "1"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n1", "1000"))  # N0 at β=1
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    α0 = parse(Float64, argval("--alpha0", "1"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    # chi flag: default 1 (steepwell / sigma_a). Pass --chi -1 for classic chi=N.
    chi_flag = parse(Float64, argval("--chi", "1"))
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    P0 = κ0 * sqrt(π * d0 * n0 * Finv_target / (2 * sa00))
    betas = parse_list(argval("--betas", "1,2,3.162,10,31.623,100"))
    chi_tag = chi_flag < 0 ? "chiN" : "chi$(chi_flag)"
    out = argval("--out", "tmp_pure_cubic_ray_Finv$(Int(round(Finv_target)))_$(chi_tag).csv")
    chi_label = chi_flag < 0 ? "N" : string(chi_flag)
    @printf("ray-scan pure cubic y=(0,eps=%.4g)  target Finv=%.4g  Finv3=%.4g  chi=%s\n",
        eps_task, Finv_target, Finv_target * eps_task^2, chi_label)
    @printf("beta=1 base: d0=%g N0=%g sa0=%g kappa0=%g P0=%.2f  (steepwell exponents)\n",
        d0, n0, sa00, κ0, P0)
    io = open(out, "w")
    println(io, "beta,alpha,d,P,N,chi,sa0,kappa,Finv,Finv3,tau,R_lin,pi_out,L3,dV_raw,V_in,V_out,mu_in,mu_out,dV_raw_pi0,dV_raw_pi1,Vout_pi1,pi_max,dF_hi,nwells")
    for β in betas
        α = α0 * β^γ
        d = round(d0 * β^εd)
        N = round(n0 * β^ν)
        sa0 = sa00 * β^ρ
        P = P0 * α^λ
        κ = κ0 * (α / β)^ω
        local chi = chi_flag < 0 ? N : chi_flag
        c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
        Finv = invariant_F(P, κ, c)
        τ = c.T
        R_lin = 2 * Finv / τ
        eq = equilibrium_at(c, P, κ)
        gridπ = walk_pi(c, P, κ)
        s0 = isempty(gridπ) || gridπ[1][1] != 0.0 ? nothing : gridπ[1][2]
        s1 = isempty(gridπ) ? nothing : gridπ[end][2]
        po_max = isempty(gridπ) ? NaN : gridπ[end][1]
        dV0 = s0 === nothing ? NaN : s0.dV_raw
        dV1 = s1 === nothing ? NaN : s1.dV_raw
        V1 = s1 === nothing ? NaN : s1.wout[3]
        F_pi0 = s0 === nothing ? NaN : s0.F
        F_pimax = s1 === nothing ? NaN : s1.F
        dF_hi = (isfinite(F_pi0) && isfinite(F_pimax)) ? (F_pimax - F_pi0) : NaN
        if eq === nothing
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,%.10g,%.10g,%.10g,%.10g,%.10g,0\n",
                β, α, d, P, N, chi, sa0, κ, Finv, Finv * eps_task^2, τ, R_lin, dV0, dV1, V1, po_max, dF_hi)
            @printf("beta=%-7.3f d=%-3.0f chi=%-5.0f Finv=%.3g R_lin=%.3g  no eq  dV_raw(pi0)=%.3g pi_max=%.3g dF_hi=%.3g\n",
                β, d, chi, Finv, R_lin, dV0, po_max, dF_hi)
            continue
        end
        po, s = eq.po, eq.s
        mark = s.dV_raw < 0 ? "OUTER_HAM" : "inner_ham"
        @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
            β, α, d, P, N, chi, sa0, κ, Finv, Finv * eps_task^2, τ, R_lin,
            po, s.L3, s.dV_raw, s.win[3], s.wout[3], s.win[1], s.wout[1], dV0, dV1, V1,
            po_max, dF_hi, s.nwells)
        @printf("beta=%-7.3f d=%-3.0f chi=%-5.0f P=%-7.1f N=%-5.0f sa0=%.4g Finv=%.3g R_lin=%.2g  pi_out=%.3g L3=%.3f dV_raw=%+.3g  pi_max=%.3g dF_hi=%+.3g  %s\n",
            β, d, chi, P, N, sa0, Finv, R_lin, po, s.L3, s.dV_raw, po_max, dF_hi, mark)
        flush(stdout)
    end
    close(io)
    println("saved ", out)
end

"""Dump F(π), dV_raw(π) along a warm walk at one (Finv, β) or fixed (d,N,P,κ)."""
function pi_scan_main()
    Finv_target = parse(Float64, argval("--Finv", "200"))
    eps_task = parse(Float64, argval("--eps", "1"))
    β = parse(Float64, argval("--beta", "10"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n1", "1000"))
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    α0 = parse(Float64, argval("--alpha0", "1"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    χ = parse(Float64, argval("--chi", "1"))
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    P0 = κ0 * sqrt(π * d0 * n0 * Finv_target / (2 * sa00))
    α = α0 * β^γ
    d = round(d0 * β^εd)
    N = round(n0 * β^ν)
    sa0 = sa00 * β^ρ
    P = P0 * α^λ
    κ = κ0 * (α / β)^ω
    # χ=N classic option: set --chi equal to N via --chi -1
    chi = χ < 0 ? N : χ
    c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    Finv = invariant_F(P, κ, c)
    R_lin = 2 * Finv / c.T
    out = argval("--out", "tmp_pure_cubic_pi_Finv$(Int(round(Finv_target)))_beta$(β).csv")
    @printf("π-scan pure cubic y=(0,ε=%.4g) β=%.4g d=%g N=%g χ=%g sa0=%.4g P=%.2f κ=%.4g Finv=%.4g R_lin=%.3g\n",
        eps_task, β, d, N, chi, sa0, P, κ, Finv, R_lin)
    grid = walk_pi(c, P, κ)
    local Fref = isempty(grid) ? 0.0 : grid[1][2].F
    io = open(out, "w")
    println(io, "pi_out,Fmf,dF_from0,dV_raw,dV,V_in,V_out,mu_in,mu_out,sig_in,sig_out,f1,L3,nwells,Q11,Q13,Q33")
    for (po, s) in grid
        dF = s.F - Fref
        @printf(io, "%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g\n",
            po, s.F, dF, s.dV_raw, s.dV, s.win[3], s.wout[3],
            s.win[1], s.wout[1], s.win[2], s.wout[2], s.f1, s.L3, s.nwells,
            s.Q[1, 1], s.Q[1, 2], s.Q[2, 2])
        @printf("π=%-8.5g  ΔF=%+.5g  dV_raw=%+.4g  dV=%+.4g  μ_out=%.3f L3=%.3f nwells=%d\n",
            po, dF, s.dV_raw, s.dV, s.wout[1], s.L3, s.nwells)
    end
    close(io)
    if !isempty(grid)
        best = grid[argmin([g[2].F for g in grid])]
        cross = nothing
        for j in 1:length(grid)-1
            (pa, sa), (pb, sb) = grid[j], grid[j+1]
            (sa.dV < 0 && sb.dV >= 0) || continue
            cross = (0.5 * (pa + pb), sa, sb)
            break
        end
        @printf("min_F at π_out=%.4g  ΔF=%.4g  dV_raw=%+.4g  π_max_converged=%.4g\n",
            best[1], best[2].F - Fref, best[2].dV_raw, grid[end][1])
        if cross !== nothing
            @printf("Laplace dV-cross near π≈%.4g  (dV lo=%+.3g hi=%+.3g)\n",
                cross[1], cross[2].dV, cross[3].dV)
        else
            dVend = grid[end][2].dV
            @printf("no dV-cross on walk; last π=%.4g dV=%+.4g (outer %s)\n",
                grid[end][1], dVend, dVend < 0 ? "still deeper" : "shallower/gone")
        end
    else
        println("no π points converged")
    end
    println("saved ", out)
end

"""Dense Finv sweep at fixed β on the steepwell ray; hunt max π_out and OUTER_HAM."""
function ray_F_scan_main()
    eps_task = parse(Float64, argval("--eps", "1"))
    β = parse(Float64, argval("--beta", "10"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n1", "1000"))
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    α0 = parse(Float64, argval("--alpha0", "1"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    χ = parse(Float64, argval("--chi", "1"))
    Fs = parse_list(argval("--Fs", "50,80,100,120,150,180,200,250,300,400,600,800,1000,2000,5000,10000,46000"))
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    α = α0 * β^γ
    d = round(d0 * β^εd)
    N = round(n0 * β^ν)
    sa0 = sa00 * β^ρ
    κ = κ0 * (α / β)^ω
    chi = χ < 0 ? N : χ
    c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    out = argval("--out", "tmp_pure_cubic_rayF_beta$(β)_eps$(eps_task).csv")
    @printf("ray-F-scan pure cubic y=(0,ε=%.4g) β=%.4g d=%g N=%g χ=%g sa0=%.4g κ=%.4g\n",
        eps_task, β, d, N, chi, sa0, κ)
    open(out, "w") do io
        println(io, "Finv,Finv3,P,R_lin,pi_out,L3,dV_raw,V_in,V_out,mu_in,mu_out,pi_max,dF_hi,dV_raw_pi0,dV_raw_pimax,nwells")
        for Finv_t in Fs
            P0 = κ0 * sqrt(π * d0 * n0 * Finv_t / (2 * sa00))
            P = P0 * α^λ
            Finv = invariant_F(P, κ, c)
            R_lin = 2 * Finv / c.T
            eq = equilibrium_at(c, P, κ)
            gridπ = walk_pi(c, P, κ)
            s0 = isempty(gridπ) || gridπ[1][1] != 0.0 ? nothing : gridπ[1][2]
            s1 = isempty(gridπ) ? nothing : gridπ[end][2]
            po_max = isempty(gridπ) ? NaN : gridπ[end][1]
            dV0 = s0 === nothing ? NaN : s0.dV_raw
            dV1 = s1 === nothing ? NaN : s1.dV_raw
            dF_hi = (s0 !== nothing && s1 !== nothing) ? (s1.F - s0.F) : NaN
            if eq === nothing
                @printf(io, "%.10g,%.10g,%.10g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,%.10g,%.10g,%.10g,%.10g,0\n",
                    Finv, Finv * eps_task^2, P, R_lin, po_max, dF_hi, dV0, dV1)
                @printf("Finv=%.4g  no eq  π_max=%.3g dV0=%.3g dF_hi=%.3g\n", Finv, po_max, dV0, dF_hi)
                continue
            end
            po, s = eq.po, eq.s
            mark = s.dV_raw < 0 ? "OUTER_HAM" : "inner_ham"
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                Finv, Finv * eps_task^2, P, R_lin, po, s.L3, s.dV_raw, s.win[3], s.wout[3],
                s.win[1], s.wout[1], po_max, dF_hi, dV0, dV1, s.nwells)
            @printf("Finv=%.4g Finv3=%.4g P=%-7.1f π_out=%.3g L3=%.3f dV_raw=%+.3g π_max=%.3g ΔF_hi=%+.3g  %s\n",
                Finv, Finv * eps_task^2, P, po, s.L3, s.dV_raw, po_max, dF_hi, mark)
            flush(stdout)
        end
    end
    println("saved ", out)
end

"""Dump the explicit single-site cavity action V(w) at χ=N equilibrium.

This is the theoretical −log p (up to an additive constant) written by the
matrix Laplace theory after integrating out A:

    V(w) = (d/s0)(1/2) w² + (1/2) c(w)ᵀ G c(w)

with c = (He1, He3) features and G from the pure-cubic teacher y=(0,ε).
"""
function dump_action_main()
    Finv_target = parse(Float64, argval("--Finv", "200"))
    eps_task = parse(Float64, argval("--eps", "1"))
    β = parse(Float64, argval("--beta", "10"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n1", "1000"))
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    α0 = parse(Float64, argval("--alpha0", "1"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    chi_flag = parse(Float64, argval("--chi", "-1"))  # default classic χ=N
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    P0 = κ0 * sqrt(π * d0 * n0 * Finv_target / (2 * sa00))
    α = α0 * β^γ
    d = round(d0 * β^εd)
    N = round(n0 * β^ν)
    sa0 = sa00 * β^ρ
    P = P0 * α^λ
    κ = κ0 * (α / β)^ω
    chi = chi_flag < 0 ? N : chi_flag
    c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    Finv = invariant_F(P, κ, c)
    eq = equilibrium_at(c, P, κ)
    eq === nothing && error("no equilibrium")
    po, s = eq.po, eq.s
    G = amp(c, s.Q, P, κ)
    Vfn = potential(c, G)
    ws = wells(Vfn)
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN")
    mkpath(out_dir)
    chi_tag = chi_flag < 0 ? "N" : string(Int(chi))
    beta_tag = replace(string(β), "." => "p")
    stem = "Finv$(Int(round(Finv_target)))_beta$(beta_tag)_chi$(chi_tag)"
    wells_path = joinpath(out_dir, "tmp_pure_cubic_wells_$(stem).csv")
    act_path = joinpath(out_dir, "tmp_pure_cubic_action_$(stem).csv")
    meta_path = joinpath(out_dir, "tmp_pure_cubic_meta_$(stem).txt")
    open(wells_path, "w") do io
        println(io, "role,mu,sig,V,multiplicity,mass")
        for (j, w) in enumerate(ws)
            role = j == 1 ? "inner" : (j == length(ws) ? "outer" : "mid")
            mass = role == "inner" ? (1 - po) : (role == "outer" ? po : 0.0)
            @printf(io, "%s,%.10g,%.10g,%.10g,%d,%.10g\n",
                role, w[1], w[2], w[3], w[4], mass)
        end
    end
    wmax = max(abs(s.wout[1]) + 4 * s.wout[2], abs(s.win[1]) + 4, 8.0)
    wgrid = range(-1.15 * wmax, 1.15 * wmax; length=4001)
    Vmin = minimum(Vfn.(wgrid))
    open(act_path, "w") do io
        println(io, "w,V,V_shifted,prior,data")
        for w in wgrid
            c1, c3 = FCS2_VGA.hermite_point_features(w, c.T)
            prior = (c.d / c.s0) * 0.5 * w^2
            data = 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
            V = prior + data
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n", w, V, V - Vmin, prior, data)
        end
    end
    open(meta_path, "w") do io
        @printf(io, "pure cubic y=(0,eps=%.4g)  chi=%g  beta=%.4g\n", eps_task, chi, β)
        @printf(io, "d=%g N=%g sa0=%.6g P=%.6g kappa=%.6g Finv=%.6g\n", d, N, sa0, P, κ, Finv)
        @printf(io, "pi_out=%.6g  L3=%.6g  dV_raw=%+.6g\n", po, s.L3, s.dV_raw)
        @printf(io, "mu_in=%.6g sig_in=%.6g V_in=%.6g\n", s.win[1], s.win[2], s.win[3])
        @printf(io, "mu_out=%.6g sig_out=%.6g V_out=%.6g\n", s.wout[1], s.wout[2], s.wout[3])
        @printf(io, "G11=%.6g G13=%.6g G33=%.6g\n", G[1, 1], G[1, 2], G[2, 2])
        @printf(io, "wells=%s\naction=%s\n", wells_path, act_path)
    end
    @printf("pi_out=%.4g  mu_in=%.4g  mu_out=±%.4g  V_in=%.4g  V_out=%.4g  dV_raw=%+.4g\n",
        po, s.win[1], s.wout[1], s.win[3], s.wout[3], s.dV_raw)
    println("saved ", wells_path)
    println("saved ", act_path)
    println("saved ", meta_path)
end

if "--ray-scan" in ARGS
    ray_scan_main()
elseif "--ray-F-scan" in ARGS
    ray_F_scan_main()
elseif "--pi-scan" in ARGS
    pi_scan_main()
elseif "--dump-action" in ARGS || "--dump-pdf" in ARGS
    dump_action_main()
elseif "--F-scan" in ARGS
    f_scan_main()
elseif "--scan" in ARGS
    scan_main()
else
    main()
end
