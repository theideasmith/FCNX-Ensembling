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

"""Teacher (He1, He3). Default pure cubic; `--he1 1` is the matrix-Laplace teacher (1, ε)."""
const TEACHER_HE1 = Ref(0.0)
teacher_y(c::Cfg) = [TEACHER_HE1[], c.eps]

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
    v = try
        V.(WGRID)
    catch
        return Tuple{Float64,Float64,Float64,Int}[]
    end
    any(x -> !isfinite(x), v) && return Tuple{Float64,Float64,Float64,Int}[]
    out = Tuple{Float64,Float64,Float64,Int}[]
    if v[1] < v[2]
        h0 = try; d2(0.0); catch; NaN; end
        if isfinite(h0) && h0 > 0
            push!(out, (0.0, 1 / sqrt(h0), v[1], 1))
        end
    end
    for j in 2:length(WGRID)-1
        (v[j] < v[j-1] && v[j] <= v[j+1]) || continue
        w = WGRID[j]
        for _ in 1:30
            h = try; d2(w); catch; NaN; end
            (isfinite(h) && h > 0) || break
            g = try; d1(w); catch; NaN; end
            isfinite(g) || break
            w -= g / h
        end
        h = try; d2(w); catch; NaN; end
        if !(isfinite(h) && h > 0 && w > 0 && abs(w - WGRID[j]) < 0.05)
            w = WGRID[j]
            h = try; d2(w); catch; NaN; end
        end
        (isfinite(h) && h > 0) || continue
        Vm = try; V(w); catch; NaN; end
        isfinite(Vm) || continue
        push!(out, (w, 1 / sqrt(h), Vm, 2))
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

"""Returns (f1, L3). f1 is induced He1 (teacher has none); L3 = f3/ε.

χ=N path: readout prior a0/(n1 χ) rescales the ridge to ρ' = κ/(a0 P).
"""
function learnabilities(c::Cfg, Q, P, κ)
    ρp = κ / (c.a0 * P)
    y = teacher_y(c)
    f = Q * ((Q + ρp * I) \ y)
    f[1], f[2] / c.eps
end

"""NNGP (χ=1). Ridge is κ/(a0 P): a0 is not replaced by a0/N, and κ is not multiplied by N."""
function learnabilities_nngp(c::Cfg, Q, P, κ)
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
        ev = eigvals(Symmetric(Q))
        scale = max(1.0, maximum(abs.(ev)))
        minimum(ev) < -1e-8 * scale && return nothing
        ev = max.(ev, 1e-14 * scale)
        F = c.n1 * (e + s - 0.5 * tr(G * Q)) + 0.5χp * dot(y, (Q + ρp * I) \ y) + 0.5 * sum(log.(ev))
        f1, L3 = learnabilities(c, Q, P, κ)
        dV = wout[3] - win[3] - log(wout[2] / win[2])
        return (Q=Q, F=F, win=win, wout=wout, f1=f1, L3=L3, dV=dV, nwells=length(ws),
            dV_raw=wout[3] - win[3])
    end
    nothing
end

"""Try fixed-π solve with several seeds / damping schedules."""
function try_fixed_pi(c::Cfg, po, Qprev, P, κ; win=nothing, wout=nothing)
    seeds = Any[Qprev, seed_Q(c, 0.0, 0.5), seed_Q(c, 2.0, 0.3),
        seed_Q(c, 3.0, 0.25), seed_Q(c, 4.0, 0.2),
        (1 - po) * seed_Q(c, 0.0, 0.4) + po * seed_Q(c, 2.5, 0.25)]
    if win !== nothing && wout !== nothing
        push!(seeds, (1 - po) * seed_Q(c, win[1], win[2]) + po * seed_Q(c, wout[1], wout[2]))
        push!(seeds, (1 - po) * seed_Q(c, win[1], win[2]) +
            po * seed_Q(c, wout[1], max(wout[2], 0.12)))
    end
    for Q0 in seeds
        for (η, iters) in ((0.3, 3000), (0.15, 6000), (0.05, 8000), (0.02, 10000))
            s = solve_fixed_pi(c, po, Q0, P, κ; iters=iters, η=η, tol=1e-9)
            s === nothing || return s
        end
    end
    nothing
end

"""Walk π from 0→1 with warm Q; return densest grid of converged (po, s).

Hardening vs early death:
1. binary refine between last-good and first-fail;
2. adaptive microsteps (Δπ ↓ to 2e-4) with slow damping from the last well
   locations — recovers cases where a coarse grid step overshoots;
3. stops when the outer well annihilates under forced π (σ blows / well merges),
   which is a real boundary of the two-well ansatz, not a seed bug.
"""
function walk_pi(c::Cfg, P, κ; pos=nothing)
    pos === nothing && (pos = vcat(0.0, 10 .^ range(-5, -0.7; length=28),
        range(0.25, 0.98; length=30), 0.99, 0.995, 0.999))
    grid = Tuple{Float64,Any}[]
    Qprev = seed_Q(c, 0.0, 0.5)
    failed_po = nothing
    for po in pos
        win = isempty(grid) ? nothing : grid[end][2].win
        wout = isempty(grid) ? nothing : grid[end][2].wout
        s = try_fixed_pi(c, po, Qprev, P, κ; win=win, wout=wout)
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
        for _ in 1:30
            m = 0.5 * (lo + hi)
            t = try_fixed_pi(c, m, Qs, P, κ; win=grid[end][2].win, wout=grid[end][2].wout)
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
    # Adaptive microstep continuation past a coarse-grid death.
    if !isempty(grid)
        Qs = grid[end][2].Q
        pocur = grid[end][1]
        Δ = 0.01
        stagnant = 0
        while pocur < 0.99 && stagnant < 3 && Δ >= 2e-4
            ponext = min(pocur + Δ, 0.99)
            t = try_fixed_pi(c, ponext, Qs, P, κ;
                win=grid[end][2].win, wout=grid[end][2].wout)
            if t === nothing
                Δ *= 0.4
                stagnant += 1
                continue
            end
            push!(grid, (ponext, t))
            Qs, pocur = t.Q, ponext
            stagnant = 0
            Δ = min(Δ * 1.25, 0.02)
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

"""Free mixture Laplace solve (π from well masses, not forced)."""
function free_solve_candidates(c::Cfg, P, κ; Q_warm=nothing)
    out = NamedTuple[]
    seeds = Any[]
    Q_warm === nothing || push!(seeds, Q_warm)
    for (μ, σ) in ((0.0, 0.5), (0.0, 0.3), (1.5, 0.3), (2.0, 0.3), (2.5, 0.25), (3.0, 0.2))
        push!(seeds, seed_Q(c, μ, σ))
        for mix in (0.05, 0.1, 0.2, 0.4)
            push!(seeds, (1 - mix) * seed_Q(c, 0.0, 0.4) + mix * seed_Q(c, μ, σ))
        end
    end
    for Q0 in seeds
        s = solve(c, Q0, P, κ; iters=6000, η=0.15, tol=1e-9)
        s === nothing && continue
        isempty(s.ws) && continue
        # Re-evaluate as fixed-π at the mixture's outer mass so F lives on the same
        # ansatz as the π-walk (outer = last well weight).
        po = length(s.π) >= 2 ? float(s.π[end]) : 0.0
        t = try_fixed_pi(c, po, s.Q, P, κ; win=s.ws[1], wout=s.ws[end])
        t === nothing && continue
        push!(out, (po=po, s=t, source=:free))
    end
    out
end

"""Global free-energy minimum over π_out at fixed (P, κ).

Uses a warm π-walk (`walk_pi`) and selects argmin_π F(π). Also competes free
mixture solves (self-consistent π). Returns `at_edge=true` when the minimizer
is the last converged constrained-π point and no better free solve exists.
"""
function equilibrium_at(c::Cfg, P, κ; Q_warm=nothing)
    grid = collect(walk_pi(c, P, κ))
    # Guarantee a π=0 trial even if the walk failed immediately.
    if isempty(grid) || grid[1][1] > 0.0
        s0 = try_fixed_pi(c, 0.0, Q_warm === nothing ? seed_Q(c, 0.0, 0.5) : Q_warm, P, κ)
        s0 !== nothing && pushfirst!(grid, (0.0, s0))
    end
    isempty(grid) && return nothing

    Fs = [g[2].F for g in grid]
    jbest = argmin(Fs)
    if jbest > 1 && jbest < length(grid)
        lo, hi = grid[jbest - 1][1], grid[jbest + 1][1]
        Qs = grid[jbest][2].Q
        sm = grid[jbest][2]
        for _ in 1:25
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            t1 = try_fixed_pi(c, m1, Qs, P, κ; win=sm.win, wout=sm.wout)
            t2 = try_fixed_pi(c, m2, Qs, P, κ; win=sm.win, wout=sm.wout)
            (t1 === nothing || t2 === nothing) && break
            if t1.F < t2.F
                hi, sm, Qs = m2, t1, t1.Q
            else
                lo, sm, Qs = m1, t2, t2.Q
            end
            hi - lo < 1e-5 && break
        end
        grid[jbest] = (0.5 * (lo + hi), sm)
        Fs[jbest] = sm.F
        jbest = argmin(Fs)
    end

    po, s = grid[jbest]
    at_edge = (jbest == length(grid) && length(grid) > 1)
    source = :walk

    # Free mixture candidates — especially useful when the constrained walk dies
    # because forced-π annihilates the outer well while F is still falling.
    for cand in free_solve_candidates(c, P, κ; Q_warm=(Q_warm === nothing ? s.Q : Q_warm))
        if cand.s.F < s.F - 1e-6
            po, s = cand.po, cand.s
            at_edge = false
            source = :free
        elseif at_edge && abs(cand.s.F - s.F) < 1e-3 * max(abs(s.F), 1.0)
            # Prefer a true self-consistent free solve over an edge artifact
            # with essentially the same F.
            po, s = cand.po, cand.s
            at_edge = false
            source = :free
        end
    end

    F0 = grid[1][1] == 0.0 ? grid[1][2].F : NaN
    F1 = grid[end][2].F
    ΔF_ends = (isfinite(F0) && isfinite(F1)) ? (F1 - F0) : NaN
    (po=po, s=s, F0=F0, F1=F1, dF_outer_minus_inner=ΔF_ends, grid=grid,
        at_edge=at_edge, source=source)
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
    println(io, "beta,alpha,d,P,N,chi,sa0,kappa,Finv,Finv3,tau,R_lin,pi_out,L3,dV_raw,V_in,V_out,mu_in,mu_out,V_in_pi0,V_out_pi0,mu_out_pi0,dV_raw_pi0,dV_raw_pi1,Vout_pi1,pi_max,dF_hi,nwells")
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
        Vin0 = s0 === nothing ? NaN : s0.win[3]
        Vout0 = s0 === nothing ? NaN : s0.wout[3]
        mu0 = s0 === nothing ? NaN : s0.wout[1]
        V1 = s1 === nothing ? NaN : s1.wout[3]
        F_pi0 = s0 === nothing ? NaN : s0.F
        F_pimax = s1 === nothing ? NaN : s1.F
        dF_hi = (isfinite(F_pi0) && isfinite(F_pimax)) ? (F_pimax - F_pi0) : NaN
        if eq === nothing
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,0\n",
                β, α, d, P, N, chi, sa0, κ, Finv, Finv * eps_task^2, τ, R_lin,
                Vin0, Vout0, mu0, dV0, dV1, V1, po_max, dF_hi)
            @printf("beta=%-7.3f d=%-3.0f chi=%-5.0f Finv=%.3g  no eq  V_out(pi0)=%.3g dV_raw(pi0)=%.3g\n",
                β, d, chi, Finv, Vout0, dV0)
            continue
        end
        po, s = eq.po, eq.s
        mark = s.dV_raw < 0 ? "OUTER_HAM" : "inner_ham"
        @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
            β, α, d, P, N, chi, sa0, κ, Finv, Finv * eps_task^2, τ, R_lin,
            po, s.L3, s.dV_raw, s.win[3], s.wout[3], s.win[1], s.wout[1],
            Vin0, Vout0, mu0, dV0, dV1, V1, po_max, dF_hi, s.nwells)
        @printf("beta=%-7.3f d=%-3.0f chi=%-5.0f Finv=%.3g  pi_out=%.3g  V_in=%.4g V_out=%.4g dV=%+.3g | pi0: V_out=%.4g dV=%+.3g  %s\n",
            β, d, chi, Finv, po, s.win[3], s.wout[3], s.dV_raw, Vout0, dV0, mark)
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
    # Extend past the outer wells until the prior walls are visible (V back above 0
    # and still rising). Default half-width ~ max(3|μ_out|, 40); override with --wmax.
    wmax = parse(Float64, argval("--wmax", "nan"))
    if !isfinite(wmax)
        wmax = max(3.0 * abs(s.wout[1]), 40.0)
    end
    wgrid = range(-wmax, wmax; length=8001)
    vals = [begin
        c1, c3 = FCS2_VGA.hermite_point_features(w, c.T)
        prior = (c.d / c.s0) * 0.5 * w^2
        data = 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
        (w, prior + data, prior, data)
    end for w in wgrid]
    Vmin = minimum(t -> t[2], vals)
    open(act_path, "w") do io
        println(io, "w,V,V_shifted,prior,data")
        for (w, V, prior, data) in vals
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

"""Dump raw V(w) at the χ-equilibrium for every β on the steepwell ray.

Writes a long CSV: beta,w,V,prior,data,pi_out,mu_in,mu_out,V_in,V_out,dV_raw
for overlay plots colored by β.
"""
function dump_ray_actions_main()
    Finv_target = parse(Float64, argval("--Finv", "200"))
    eps_task = parse(Float64, argval("--eps", "1"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n1", "1000"))
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    α0 = parse(Float64, argval("--alpha0", "1"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    chi_flag = parse(Float64, argval("--chi", "-1"))
    wmax = parse(Float64, argval("--wmax", "40"))
    ngrid = parse(Int, argval("--ngrid", "2001"))
    betas = parse_list(argval("--betas", "1,2,3.162,10,31.623,100"))
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    P0 = κ0 * sqrt(π * d0 * n0 * Finv_target / (2 * sa00))
    chi_tag = chi_flag < 0 ? "chiN" : "chi$(Int(chi_flag))"
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN")
    mkpath(out_dir)
    out = argval("--out", joinpath(out_dir, "tmp_pure_cubic_ray_actions_Finv$(Int(round(Finv_target)))_$(chi_tag).csv"))
    @printf("ray-actions pure cubic y=(0,eps=%.4g) Finv=%.4g chi=%s  |w|<=%.3g\n",
        eps_task, Finv_target, chi_flag < 0 ? "N" : string(chi_flag), wmax)
    io = open(out, "w")
    println(io, "beta,alpha,d,P,N,chi,sa0,kappa,Finv,pi_out,mu_in,mu_out,V_in,V_out,dV_raw,w,V,prior,data")
    wgrid = range(-wmax, wmax; length=ngrid)
    for β in betas
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
        if eq === nothing
            @printf("beta=%.4g  no equilibrium — skip\n", β)
            continue
        end
        po, s = eq.po, eq.s
        G = amp(c, s.Q, P, κ)
        for w in wgrid
            c1, c3 = FCS2_VGA.hermite_point_features(w, c.T)
            prior = (c.d / c.s0) * 0.5 * w^2
            data = 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
            V = prior + data
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g,%.10g,%.10g,%.10g\n",
                β, α, d, P, N, chi, sa0, κ, Finv, po, s.win[1], s.wout[1],
                s.win[3], s.wout[3], s.dV_raw, w, V, prior, data)
        end
        @printf("beta=%-7.3f d=%-3.0f chi=%-5.0f Finv=%.3g pi_out=%.3g V_in=%.4g V_out=%.4g dV=%+.3g\n",
            β, d, chi, Finv, po, s.win[3], s.wout[3], s.dV_raw)
        flush(stdout)
    end
    close(io)
    println("saved ", out)
end

"""Steepwell-ray point at a chosen d, with Finv held at Finv_target.

Same exponents as ray_scan_main:
  d = 5 √β,  N ~ β^{1/4},  sa0 ~ β^{-3/4},  α_steep = β^{5/4},
  P_ray = P0 α_steep^{3/4},  κ = κ0 (α_steep/β)^{3/4},  χ = N.
P0 is chosen so Finv(P_ray) = Finv_target.
"""
function ray_at_d(d_target::Float64, Finv_target::Float64)
    d0, n0, sa00 = 5.0, 1000.0, 0.03
    κ0, α0, γ = 1.0 / n0, 1.0, 1.25
    εd, ν, ω = 0.5, 0.25, 0.75
    ρ, λ = -ω, ω
    β = (d_target / d0)^(1 / εd)
    αs = α0 * β^γ
    d = round(d0 * β^εd)
    N = round(n0 * β^ν)
    sa0 = sa00 * β^ρ
    P0 = κ0 * sqrt(π * d0 * n0 * Finv_target / (2 * sa00))
    P = P0 * αs^λ
    κ = κ0 * (αs / β)^ω
    (beta=β, alpha_steep=αs, d=d, N=N, sa0=sa0, kappa=κ, P_ray=P, P0=P0)
end

"""At each d on the Finv-flat ray, dump the equilibrium action V(w) and sweep
matrix learnability L3 against α defined by P = d^α.

χ = N, teacher y = (0, ε). (N, sa0, κ) stay on the ray; only P moves with α.
"""
function d_alpha_scan_main()
    Finv_target = parse(Float64, argval("--Finv", "200"))
    eps_task = parse(Float64, argval("--eps", "1"))
    d_one = argval("--d", "")
    ds = isempty(d_one) ? parse_list(argval("--ds", "20,32,50,80,125,200")) : [parse(Float64, d_one)]
    alphas = parse_list(argval("--alphas", join(string.(range(0.8, 2.6; step=0.05)), ",")))
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN/dscale_20_200")
    mkpath(out_dir)
    wmax_fixed = parse(Float64, argval("--wmax", "30"))
    ngrid = parse(Int, argval("--ngrid", "1601"))
    for d_target in ds
        rp = ray_at_d(d_target, Finv_target)
        d, N, sa0, κ = rp.d, rp.N, rp.sa0, rp.kappa
        chi = N
        tag = "d$(Int(d))"
        act_path = joinpath(out_dir, "tmp_pure_cubic_action_$(tag).csv")
        learn_path = joinpath(out_dir, "tmp_pure_cubic_learn_$(tag).csv")
        @printf("\n=== d=%g  β=%.4g  N=%g  sa0=%.4g  κ=%.4g  P_ray=%.4g  α_ray=%.4f ===\n",
            d, rp.beta, N, sa0, κ, rp.P_ray, log(rp.P_ray) / log(d))
        # Equilibrium action at the Finv-flat point (the plotted theory).
        c_ray = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
        eq_ray = equilibrium_at(c_ray, rp.P_ray, κ)
        if eq_ray === nothing
            @printf("d=%g  no ray equilibrium — skip\n", d)
            continue
        end
        po, s = eq_ray.po, eq_ray.s
        G = amp(c_ray, s.Q, rp.P_ray, κ)
        Finv = invariant_F(rp.P_ray, κ, c_ray)
        wgrid = range(-wmax_fixed, wmax_fixed; length=ngrid)
        open(act_path, "w") do io
            println(io, "d,beta,N,chi,sa0,kappa,P,alpha,Finv,pi_out,L3,mu_in,mu_out,V_in,V_out,dV_raw,w,V,prior,data")
            αP = log(rp.P_ray) / log(d)
            for w in wgrid
                c1, c3 = FCS2_VGA.hermite_point_features(w, c_ray.T)
                prior = (c_ray.d / c_ray.s0) * 0.5 * w^2
                data = 0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c3 + G[2, 2] * c3^2)
                @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g,%.10g,%.10g,%.10g\n",
                    d, rp.beta, N, chi, sa0, κ, rp.P_ray, αP, Finv, po, s.L3,
                    s.win[1], s.wout[1], s.win[3], s.wout[3], s.dV_raw,
                    w, prior + data, prior, data)
            end
        end
        @printf("action  Finv=%.4g  π_out=%.4g  L3=%.4f  μ_in=%.3f  μ_out=%.3f  dV_raw=%+.4g\n",
            Finv, po, s.L3, s.win[1], s.wout[1], s.dV_raw)
        println("saved ", act_path)
        if "--actions-only" in ARGS
            continue
        end
        # Learnability vs α, P = d^α. Include the ray sample size exactly.
        P_of = Dict{Float64,Float64}()
        for α in alphas
            P_of[float(α)] = d^α
        end
        α_ray = log(rp.P_ray) / log(d)
        P_of[α_ray] = rp.P_ray
        αs = sort(collect(keys(P_of)))
        Qwarm = s.Q
        open(learn_path, "w") do io
            println(io, "d,N,sa0,kappa,alpha,P,Finv,pi_out,L3,f1,mu_in,mu_out,V_in,V_out,dV_raw,nwells,ray")
            for α in αs
                P = P_of[α]
                c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
                eq = try
                    equilibrium_at(c, P, κ; Q_warm=Qwarm)
                catch err
                    @printf("  α=%.3f  P=%.4g  failed: %s\n", α, P, err)
                    nothing
                end
                ray_flag = abs(P - rp.P_ray) / rp.P_ray < 1e-9 ? 1 : 0
                if eq === nothing
                    @printf(io, "%.10g,%.10g,%.10g,%.10g,%.6g,%.10g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,%d\n",
                        d, N, sa0, κ, α, P, ray_flag)
                    @printf("  α=%.3f  P=%.4g  no equilibrium\n", α, P)
                    continue
                end
                spo, ss = eq.po, eq.s
                Qwarm = ss.Q
                Finvα = invariant_F(P, κ, c)
                @printf(io, "%.10g,%.10g,%.10g,%.10g,%.6g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d\n",
                    d, N, sa0, κ, α, P, Finvα, spo, ss.L3, ss.f1,
                    ss.win[1], ss.wout[1], ss.win[3], ss.wout[3], ss.dV_raw, ss.nwells, ray_flag)
                @printf("  α=%.3f  P=%-10.4g  Finv=%-8.3g  L3=%.4f  π_out=%.3f  dV_raw=%+.3g%s\n",
                    α, P, Finvα, ss.L3, spo, ss.dV_raw, ray_flag == 1 ? "  [ray]" : "")
                flush(io)
                flush(stdout)
            end
        end
        println("saved ", act_path)
        println("saved ", learn_path)
    end
end

"""Central-node pure-cubic action along the ρ-ray (P, sa0 fixed; κ = ρ sa0 P).

Both χ=N and the NNGP use ridge ρ' = κ/(sa0 P). The NNGP does not replace sa0 by
sa0/N and does not multiply κ by N. P_eff = 1/ρ' and α from d^α = P_eff.
"""
function rho_learn_main()
    d = parse(Float64, argval("--d", "16"))
    P = parse(Float64, argval("--P", "1000"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps_task = parse(Float64, argval("--eps", "1"))
    n1 = parse(Float64, argval("--n1", "200"))
    chi_flag = parse(Float64, argval("--chi", string(n1)))
    chi = chi_flag < 0 ? n1 : chi_flag
    # NNGP: χ=1, prior kernel. Ridge κ/(sa0 P), with no N on κ and no N on sa0.
    nngp = "--nngp" in ARGS || chi == 1
    σ_default = nngp ? 1 / sqrt(d) : 0.5
    σk = parse(Float64, argval("--sigma", string(σ_default)))
    self_c = "--self-consistent" in ARGS && !nngp
    rhos = parse_list(argval("--rhos", join(string.(exp.(range(log(1e-6), log(10.0); length=140))), ",")))
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN/rho_ray_learn")
    mkpath(out_dir)
    c = Cfg(d, n1, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    if "--match" in ARGS
        # Compare shifted V(w) to the saved ρ-ray at two ridges.
        ref = argval("--ref", "action_h0_activation_plots/PureCubicChiN/tmp_pure_cubic_rho_ray_actions.csv")
        for ρ in (0.15, 0.10)
            κ = ρ * sa0 * P
            s = solve_fixed_pi(c, 0.0, seed_Q(c, 0.0, 0.5), P, κ)
            s === nothing && (@printf("match ρ=%.3g no solution\n", ρ); continue)
            G = amp(c, s.Q, P, κ)
            Vfn = potential(c, G)
            V0 = Vfn(0.0)
            @printf("match N=%g ρ=%.3g  μ_out=%.3f ΔV=%+.3f sig=%.3f G=[%.4g %.4g; %.4g %.4g]\n",
                n1, ρ, s.wout[1], s.wout[3] - s.win[3], s.win[2], G[1,1], G[1,2], G[2,1], G[2,2])
            for σ in (0.25, 0.5, 0.7, 0.9, 1.1, 1.4)
                Gf = amp(c, seed_Q(c, 0.0, σ), P, κ)
                Vf = potential(c, Gf)
                z = Vf(0.0)
                @printf("  frozen σ=%.3f  V(1.52)=%+.3f V(2.06)=%+.3f V(2.43)=%+.3f  G=[%.4g %.4g; %.4g %.4g]\n",
                    σ, Vf(1.52) - z, Vf(2.06) - z, Vf(2.43) - z, Gf[1,1], Gf[1,2], Gf[2,1], Gf[2,2])
            end
        end
        println("ref file ", ref)
        return
    end
    chi_tag = nngp ? "nngp" : (chi == n1 ? "chiN" : "chi$(chi)")
    mode_tag = self_c ? "selfconsistent" : "frozen"
    out = joinpath(out_dir, "tmp_pure_cubic_rho_learn_d$(Int(d))_$(chi_tag)_$(mode_tag).csv")
    Q = seed_Q(c, 0.0, σk)
    @printf("ρ-ray learnability  d=%g N=%g χ=%g P=%g sa0=%g ε=%g  %s σ=%g\n",
        d, n1, chi, P, sa0, eps_task, nngp ? "NNGP prior" : (self_c ? "self-consistent from" : "frozen"), σk)
    if nngp
        @printf("NNGP ridge ρ'=κ/(sa0 P)  (no ×N on κ, no /N on sa0);  P_eff=1/ρ';  α = log(P_eff)/log(d)\n")
    else
        @printf("ridge ρ'=κ/(sa0 P);  P_eff=1/ρ';  α = log(P_eff)/log(d)\n")
    end
    @printf("Q0 = [%.5g %.5g; %.5g %.5g]\n", Q[1,1], Q[1,2], Q[2,1], Q[2,2])
    io = open(out, "w")
    println(io, "rho,kappa,P,sa0,d,N,chi,sigma,P_eff,alpha,L3,f1,mu_out,V_out,dV_raw,nwells")
    Qwarm = Q
    for ρ in sort(rhos; rev=true)
        κ = ρ * sa0 * P
        if self_c
            s = solve_fixed_pi(c, 0.0, Qwarm, P, κ)
            if s === nothing
                s = solve_fixed_pi(c, 0.0, seed_Q(c, 0.0, σk), P, κ)
            end
            if s === nothing
                @printf("  ρ=%.4g  no central-node solution\n", ρ)
                continue
            end
            Qwarm = s.Q
            Qρ, σ_here = s.Q, s.win[2]
            f1, L3 = s.f1, s.L3
        else
            Qρ, σ_here = Q, σk
            f1, L3 = nngp ? learnabilities_nngp(c, Qρ, P, κ) : learnabilities(c, Qρ, P, κ)
        end
        # Well geometry only where the ρ-ray figure lives. At tiny ρ the
        # potential is huge and the crossing is already known from that figure.
        mu_out, V_out, dV, nwells = 0.0, 0.0, NaN, 0
        if 0.05 <= ρ <= 0.4 && !self_c && !nngp
            G = amp(c, Qρ, P, κ)
            Vfn = potential(c, G)
            V0 = Vfn(0.0)
            ws = wells(Vfn)
            nwells = length(ws)
            if nwells >= 2
                mu_out = ws[end][1]
                V_out = ws[end][3] - V0
                dV = V_out
            end
        end
        Peff = 1.0 / ρ
        α = log(Peff) / log(d)
        @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
            ρ, κ, P, sa0, d, n1, chi, σ_here, Peff, α, L3, f1, mu_out, V_out, dV, nwells)
        if self_c || ρ < 3e-4 || abs(log(ρ / 0.14)) < 0.15
            @printf("  ρ=%.4g  P_eff=%.3g  α=%.3f  L3=%.4f  σ=%.4f\n", ρ, Peff, α, L3, σ_here)
        end
        flush(io)
    end
    close(io)
    println("saved ", out)
end

"""Raw V(w) along the ρ-ray (d, N, P, sa0 fixed; κ = ρ sa0 P; χ = N), from the solver itself.

Two self-consistent solutions per ρ, each written to its own long CSV:
  center       solve_fixed_pi at π_out = 0: Q is built from the central well of its own V.
  equilibrium  equilibrium_at: π_out at the lowest mean-field free energy.
V(0) = 0 for both (prior and He1/He3 features vanish at w = 0), so V is unshifted.
"""
function rho_ray_actions_main()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    P = parse(Float64, argval("--P", "1000"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps_task = parse(Float64, argval("--eps", "1"))
    wmax = parse(Float64, argval("--wmax", "12"))
    ngrid = parse(Int, argval("--ngrid", "4001"))
    rhos = sort(parse_list(argval("--rhos",
        "0.3,0.26,0.22,0.2,0.18,0.17,0.16,0.15,0.145,0.14,0.135,0.13,0.125,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05")); rev=true)
    modes = split(argval("--modes", "equilibrium"), ",")
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN")
    mkpath(out_dir)
    c = Cfg(d, n1, n1, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    wgrid = range(-wmax, wmax; length=ngrid)
    header = "rho,kappa,P,sa0,d,N,chi,pi_out,mu_in,sig_in,mu_out,sig_out,V_in,V_out,dV_raw,L3,F,nwells,at_edge,w,V"
    phase(s) = s.nwells < 2 ? "no_outer" : (s.dV_raw > 0 ? "inner_deeper" : "outer_deeper")
    @printf("ρ-ray actions (solver)  d=%g N=χ=%g P=%g sa0=%g ε=%g  y=(0,ε)  modes=%s\n",
        d, n1, P, sa0, eps_task, join(modes, ","))
    for mode in modes
        mode in ("center", "equilibrium") || error("unknown mode $mode")
        act_path = joinpath(out_dir, "tmp_pure_cubic_rho_ray_actions_$(mode).csv")
        sum_path = joinpath(out_dir, "tmp_pure_cubic_rho_ray_summary_$(mode).csv")
        io, sio = open(act_path, "w"), open(sum_path, "w")
        println(io, header)
        println(sio, "rho,kappa,pi_out,mu_in,sig_in,mu_out,sig_out,V_in,V_out,dV_raw,L3,F,nwells,at_edge,phase")
        Qwarm = seed_Q(c, 0.0, 0.5)
        for ρ in rhos
            κ = ρ * sa0 * P
            local po, s, at_edge, dF0, src
            at_edge = false
            dF0 = NaN
            src = "center"
            if mode == "center"
                s = nothing
                for Q0 in (Qwarm, seed_Q(c, 0.0, 0.5), seed_Q(c, 0.0, 0.3))
                    s = solve_fixed_pi(c, 0.0, Q0, P, κ)
                    s === nothing || break
                end
                po = 0.0
            else
                eq = equilibrium_at(c, P, κ; Q_warm=Qwarm)
                s = eq === nothing ? nothing : eq.s
                po = eq === nothing ? NaN : eq.po
                at_edge = eq === nothing ? false : eq.at_edge
                dF0 = (eq !== nothing && isfinite(eq.F0)) ? (eq.s.F - eq.F0) : NaN
                src = eq === nothing ? "none" : string(eq.source)
            end
            if s === nothing
                @printf("  %-11s ρ=%-6.4g κ=%-6.4g  no solution\n", mode, ρ, κ)
                continue
            end
            Qwarm = s.Q
            G = amp(c, s.Q, P, κ)
            Vfn = potential(c, G)
            Vout = s.nwells < 2 ? NaN : s.wout[3]
            muout = s.nwells < 2 ? NaN : s.wout[1]
            sigout = s.nwells < 2 ? NaN : s.wout[2]
            dV = s.nwells < 2 ? NaN : s.dV_raw
            edge_i = at_edge ? 1 : 0
            meta = @sprintf("%.10g,%.10g,%.10g,%.10g,%g,%g,%g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g,%d,%d",
                ρ, κ, P, sa0, d, n1, n1, po, s.win[1], s.win[2], muout, sigout, s.win[3], Vout, dV, s.L3, s.F, s.nwells, edge_i)
            for w in wgrid
                @printf(io, "%s,%.10g,%.10g\n", meta, w, Vfn(w))
            end
            @printf(sio, "%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g,%d,%d,%s\n",
                ρ, κ, po, s.win[1], s.win[2], muout, sigout, s.win[3], Vout, dV, s.L3, s.F, s.nwells, edge_i, phase(s))
            flush(io); flush(sio)
            edge_tag = at_edge ? " EDGE" : ""
            @printf("  %-11s ρ=%-6.4g π_out=%-7.4g ΔF=%+-9.4g μ_out=%-6.3f V_in=%-8.4g V_out=%-8.4g dV_raw=%+-8.4g  %s%s [%s]\n",
                mode, ρ, po, dF0, muout, s.win[3], Vout, dV, phase(s), edge_tag, src)
            flush(stdout)
        end
        close(io); close(sio)
        println("saved ", act_path, " and ", sum_path)
    end
end

"""F(π_out) along a warm π walk at each ρ of the ρ-ray (same Cfg as --rho-ray-actions).

A first-order transition shows up as two separated local minima of F(π) exchanging order.
"""
function rho_pi_scan_main()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    P = parse(Float64, argval("--P", "1000"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps_task = parse(Float64, argval("--eps", "1"))
    rhos = sort(parse_list(argval("--rhos", "0.15,0.147,0.145,0.14,0.13,0.12,0.1,0.09,0.08,0.06")); rev=true)
    out = argval("--out", "action_h0_activation_plots/PureCubicChiN/tmp_pure_cubic_rho_pi_scan.csv")
    c = Cfg(d, n1, n1, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    io = open(out, "w")
    println(io, "rho,kappa,pi_out,F,dF_from0,dV_raw,dV,V_in,V_out,mu_in,mu_out,sig_in,sig_out,L3,nwells")
    for ρ in rhos
        κ = ρ * sa0 * P
        grid = walk_pi(c, P, κ)
        isempty(grid) && (@printf("ρ=%.4g  no π points\n", ρ); continue)
        F0 = grid[1][2].F
        Fs = [g[2].F for g in grid]
        mins = [j for j in eachindex(Fs) if (j == 1 || Fs[j] <= Fs[j-1]) && (j == length(Fs) || Fs[j] <= Fs[j+1])]
        for (po, s) in grid
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                ρ, κ, po, s.F, s.F - F0, s.dV_raw, s.dV, s.win[3], s.wout[3],
                s.win[1], s.wout[1], s.win[2], s.wout[2], s.L3, s.nwells)
        end
        flush(io)
        @printf("ρ=%-6.4g  π walk to %.4g (%d pts)  local minima of F(π): %s\n", ρ, grid[end][1], length(grid),
            join([@sprintf("π=%.4g ΔF=%+.4g", grid[j][1], Fs[j] - F0) for j in mins], " | "))
        flush(stdout)
    end
    close(io)
    println("saved ", out)
end

"""Continue one F(π) local minimum by warm fixed-π search near `piprev`.

Does not re-walk the full π axis (that re-discovers other basins and jumps).
Builds F(π) on a neighborhood grid, takes the local min of F nearest to
`piprev` (not the global argmin in the window — that jumps basins at FO),
then ternary-refines.
"""
function continue_pi_min(c::Cfg, P, κ, piprev, Qprev; window=0.04, ngrid=25,
        win=nothing, wout=nothing)
    lo = max(0.0, piprev - window)
    hi = min(0.99, piprev + window)
    hi <= lo && return nothing
    pos = collect(range(lo, hi; length=ngrid))
    Fs = fill(Inf, length(pos))
    Ss = Vector{Any}(undef, length(pos))
    for (i, po) in enumerate(pos)
        t = try_fixed_pi(c, po, Qprev, P, κ; win=win, wout=wout)
        t === nothing && continue
        Fs[i] = t.F
        Ss[i] = t
    end
    # local minima on the neighborhood grid
    locs = Int[]
    for i in eachindex(Fs)
        isfinite(Fs[i]) || continue
        left = (i == 1) || !(isfinite(Fs[i - 1]) && Fs[i - 1] < Fs[i])
        right = (i == length(Fs)) || !(isfinite(Fs[i + 1]) && Fs[i + 1] < Fs[i])
        left && right && push!(locs, i)
    end
    # fallback: any finite point closest to piprev
    isempty(locs) && (locs = findall(isfinite, Fs))
    isempty(locs) && return nothing
    j = locs[argmin(abs.(pos[locs] .- piprev))]
    # ternary refine in a cell around j
    a = j == 1 ? lo : 0.5 * (pos[j - 1] + pos[j])
    b = j == length(pos) ? hi : 0.5 * (pos[j] + pos[j + 1])
    sm = Ss[j]
    Qs = sm.Q
    for _ in 1:20
        b - a < 1e-5 && break
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        t1 = try_fixed_pi(c, m1, Qs, P, κ; win=sm.win, wout=sm.wout)
        t2 = try_fixed_pi(c, m2, Qs, P, κ; win=sm.win, wout=sm.wout)
        if t1 === nothing && t2 === nothing
            break
        elseif t1 === nothing
            a, sm, Qs = m1, t2, t2.Q
        elseif t2 === nothing
            b, sm, Qs = m2, t1, t1.Q
        elseif t1.F < t2.F
            b, sm, Qs = m2, t1, t1.Q
        else
            a, sm, Qs = m1, t2, t2.Q
        end
    end
    po = 0.5 * (a + b)
    (po=po, s=sm, at_edge=false, F=sm.F, F0=NaN, dF=NaN)
end

"""Local minima of F(π) on a π-walk grid (non-edge preferred)."""
function walk_local_mins(grid)
    isempty(grid) && return NamedTuple[]
    Fs = [g[2].F for g in grid]
    out = NamedTuple[]
    for j in eachindex(Fs)
        left = (j == 1) || Fs[j] <= Fs[j - 1]
        right = (j == length(Fs)) || Fs[j] <= Fs[j + 1]
        left && right || continue
        at_edge = (j == length(grid) && length(grid) > 1)
        push!(out, (po=grid[j][1], s=grid[j][2], at_edge=at_edge, F=Fs[j]))
    end
    out
end

"""Continuous F(π) branches vs ρ via warm π-neighborhood continuation.

Global argmin F jumps at the FO; each metastable basin is tracked separately so
L3 / V / π stay continuous along a branch. A second branch is spawned only when
a walk local-min sits Δπ ≥ `sep` from every continued branch.
"""
function rho_branch_scan_main()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    P = parse(Float64, argval("--P", "1000"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps_task = parse(Float64, argval("--eps", "1"))
    rhos = sort(parse_list(argval("--rhos",
        "0.16,0.155,0.15,0.145,0.14,0.135,0.13,0.125,0.12,0.115,0.11,0.105,0.1,0.095,0.09,0.085,0.082,0.08,0.078,0.076,0.075,0.07,0.065,0.06")); rev=true)
    sep = parse(Float64, argval("--sep", "0.03"))
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin")
    mkpath(out_dir)
    out = argval("--out", joinpath(out_dir, "branch_local_mins.csv"))
    c = Cfg(d, n1, n1, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    @printf("ρ-branch continuation  d=%g N=χ=%g P=%g sa0=%g ε=%g  sep=%.3g\n",
        d, n1, P, sa0, eps_task, sep)

    # Live branches: (id, po, s). id is stable across ρ for CSV labeling.
    branches = NamedTuple[]
    next_id = 1
    io = open(out, "w")
    println(io, "rho,kappa,branch,pi_out,F,dF_from0,dV_raw,V_in,V_out,mu_out,L3,nwells,at_edge,is_global")
    for ρ in rhos
        κ = ρ * sa0 * P
        # F0 from π=0 for ΔF reporting
        s0 = try_fixed_pi(c, 0.0, seed_Q(c, 0.0, 0.5), P, κ)
        F0 = s0 === nothing ? NaN : s0.F

        continued = NamedTuple[]
        if isempty(branches)
            # Seed from free solves + walk local mins at the first ρ.
            seeds = NamedTuple[]
            for cand in free_solve_candidates(c, P, κ)
                push!(seeds, (po=cand.po, s=cand.s, at_edge=false, F=cand.s.F))
            end
            grid = walk_pi(c, P, κ)
            append!(seeds, walk_local_mins(grid))
            # Keep distinct basins (by π), lowest F first. Prefer non-edge, but
            # accept an edge seed when it is the only basin that converged.
            sort!(seeds; by=x -> (x.at_edge ? 1 : 0, x.F))
            for sd in seeds
                any(abs(sd.po - br.po) < sep for br in continued) && continue
                push!(continued, (id=next_id, po=sd.po, s=sd.s, at_edge=sd.at_edge, F=sd.F))
                next_id += 1
            end
        else
            for br in branches
                nxt = continue_pi_min(c, P, κ, br.po, br.s.Q; window=0.05,
                    win=br.s.win, wout=br.s.wout)
                nxt === nothing && continue
                push!(continued, (id=br.id, po=nxt.po, s=nxt.s, at_edge=false, F=nxt.F))
            end
            # Discover new basins from a fresh walk (FO birth of a second min).
            # Full walks are expensive: always near the FO window, otherwise every
            # other ρ step is enough to catch a new local min.
            do_walk = (ρ <= 0.10) || (mod(round(Int, ρ * 1000), 10) == 0)
            if do_walk
                grid = walk_pi(c, P, κ)
                for sd in walk_local_mins(grid)
                    sd.at_edge && continue
                    any(abs(sd.po - br.po) < sep for br in continued) && continue
                    Fref = isempty(continued) ? Inf : minimum(br.F for br in continued)
                    sd.F > Fref + 50.0 && continue
                    push!(continued, (id=next_id, po=sd.po, s=sd.s, at_edge=false, F=sd.F))
                    next_id += 1
                end
            end
        end

        # Dedup: if two continued branches collapsed, keep lower F / older id.
        sort!(continued; by=x -> (x.F, x.id))
        kept = NamedTuple[]
        for br in continued
            any(abs(br.po - k.po) < sep for k in kept) && continue
            push!(kept, br)
        end
        # Label by π order for readability: branch_1 = lowest π, etc. but keep
        # stable numeric id in the CSV `branch` field as "b<id>".
        sort!(kept; by=x -> x.po)
        branches = kept

        Fglob = isempty(branches) ? Inf : minimum(br.F for br in branches)
        parts = String[]
        for (k, br) in enumerate(branches)
            s = br.s
            Vout = s.nwells < 2 ? NaN : s.wout[3]
            muout = s.nwells < 2 ? NaN : s.wout[1]
            dV = s.nwells < 2 ? NaN : s.dV_raw
            dF = isfinite(F0) ? br.F - F0 : NaN
            is_global = br.F <= Fglob + 1e-9
            name = "b$(br.id)"
            @printf(io, "%.10g,%.10g,%s,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%d\n",
                ρ, κ, name, br.po, br.F, dF, dV, s.win[3], Vout, muout, s.L3, s.nwells,
                br.at_edge ? 1 : 0, is_global ? 1 : 0)
            push!(parts, @sprintf("%s:π=%.4g ΔF=%+.4g L3=%.4g%s", name, br.po, dF, s.L3,
                is_global ? " *" : ""))
        end
        flush(io)
        @printf("ρ=%-6.4g  %s\n", ρ, isempty(parts) ? "none" : join(parts, " | "))
        flush(stdout)
    end
    close(io)
    println("saved ", out)
end

"""Self-consistent multiwell equilibrium along the steepwell ray.

d = d0 β^{1/2}, N = N0 β^{1/4}, sa0 = sa00 β^{-3/4}, α = β^{5/4},
P = P0 α^{3/4}, κ = κ0 (α/β)^{3/4}. χ = N and χ = 1 both use this ray
and the Laplace amplitude (ridge κ/(sa0 P), no extra factor of N on κ or sa0).
σ is whatever the equilibrium wells come out to be.
"""
function theory_ray_main()
    TEACHER_HE1[] = parse(Float64, argval("--he1", "1"))
    eps_task = parse(Float64, argval("--eps", "0.1"))
    d0 = parse(Float64, argval("--d0", "5"))
    n0 = parse(Float64, argval("--n0", "1000"))
    sa00 = parse(Float64, argval("--sa0", "0.03"))
    κ0 = parse(Float64, argval("--kappa0", string(1 / n0)))
    P0 = parse(Float64, argval("--P0", "160"))
    γ = parse(Float64, argval("--gamma", "1.25"))
    εd, ν, ω = 0.5, 0.25, 0.75
    betas = parse_list(argval("--betas", "1,1.6,2.5,4,6.3,10,16,25,40,63,100"))
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN/theory_ray")
    mkpath(out_dir)
    he1 = TEACHER_HE1[]
    @printf("theory ray  y=(%.4g, %.4g)  d0=%g N0=%g sa00=%g κ0=%g P0=%g\n",
        he1, eps_task, d0, n0, sa00, κ0, P0)
    for (tag, chi_of) in (("chiN", N -> N), ("chi1", _ -> 1.0))
        out = joinpath(out_dir, "tmp_theory_ray_he$(he1)_eps$(eps_task)_$(tag).csv")
        io = open(out, "w")
        println(io, "beta,d,d_over_d0,N,chi,sa0,kappa,P,pi_out,L1,L3,mu_in,sig_in,mu_out,sig_out,nwells,Q11,Q13,Q33,F")
        Qwarm = nothing
        for β in betas
            α = β^γ
            d = d0 * β^εd
            N = n0 * β^ν
            sa0 = sa00 * β^(-ω)
            P = P0 * α^ω
            κ = κ0 * (α / β)^ω
            chi = chi_of(N)
            c = Cfg(d, N, chi, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
            eq = equilibrium_at(c, P, κ; Q_warm=Qwarm)
            if eq === nothing
                @printf("  %s β=%.3g  no equilibrium\n", tag, β)
                continue
            end
            Qwarm = eq.s.Q
            s = eq.s
            @printf(io, "%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%.8g,%.8g,%.8g,%.8g\n",
                β, d, d / d0, N, chi, sa0, κ, P, eq.po, s.f1, s.L3,
                s.win[1], s.win[2], s.wout[1], s.wout[2], s.nwells,
                s.Q[1, 1], s.Q[1, 2], s.Q[2, 2], s.F)
            flush(io)
            @printf("  %s β=%-6.3g d=%-6.3g d/d0=%-6.3g σ_in=%.3f σ_out=%.3f n=%d π=%.3f L1=%.3f L3=%.3f\n",
                tag, β, d, d / d0, s.win[2], s.wout[2], s.nwells, eq.po, s.f1, s.L3)
            flush(stdout)
        end
        close(io)
        println("saved ", out)
    end
end

"""Deep ρ continuation toward L3 → 1 (warm π-neighborhood + free solve).

Starts at the first ρ with a full `equilibrium_at`, then continues the basin with
`continue_pi_min` (no full π-walk each step). Every `--reeq-every` steps (and at
the end) re-runs `equilibrium_at` to check the global min has not jumped.
"""
function rho_deep_main()
    d = parse(Float64, argval("--d", "16"))
    n1 = parse(Float64, argval("--n1", "200"))
    P = parse(Float64, argval("--P", "1000"))
    sa0 = parse(Float64, argval("--sa0", "0.03"))
    eps_task = parse(Float64, argval("--eps", "1"))
    rhos = sort(parse_list(argval("--rhos",
        "0.07,0.06,0.05,0.04,0.03,0.025,0.02,0.015,0.012,0.01,0.008,0.006,0.005,0.004,0.003,0.0025,0.002,0.0015,0.001,0.0007,0.0005,0.0003,0.0002,0.0001,5e-5,2e-5,1e-5")); rev=true)
    reeq_every = parse(Int, argval("--reeq-every", "4"))
    seed_pi = parse(Float64, argval("--seed-pi", "0.17"))  # post-FO high-π basin
    out_dir = argval("--outdir", "action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3")
    mkpath(out_dir)
    out = argval("--out", joinpath(out_dir, "summary_deep_equilibrium.csv"))
    c = Cfg(d, n1, n1, sa0, eps_task, 1.0, 1.0 + 2.0 * (d - 1.0) / d)
    @printf("ρ-deep continuation  d=%g N=χ=%g P=%g sa0=%g ε=%g  seed-π=%g  %d ρ points\n",
        d, n1, P, sa0, eps_task, seed_pi, length(rhos))
    io = open(out, "w")
    println(io, "rho,kappa,pi_out,mu_in,sig_in,mu_out,sig_out,V_in,V_out,dV_raw,L3,F,nwells,at_edge,source,lam33_implied")
    br = nothing  # (po, s)
    for (k, ρ) in enumerate(rhos)
        κ = ρ * sa0 * P
        src = "continue"
        at_edge = false
        if br === nothing
            # Seed the post-FO high-π basin directly (global min at ρ≲0.075).
            seed = try
                continue_pi_min(c, P, κ, seed_pi, seed_Q(c, 2.3, 0.2); window=0.08)
            catch err
                @printf("  ρ=%.4g  seed continue threw %s\n", ρ, err)
                nothing
            end
            if seed === nothing
                eq = try
                    equilibrium_at(c, P, κ)
                catch err
                    @printf("  ρ=%.4g  equilibrium_at threw %s\n", ρ, err)
                    nothing
                end
                if eq === nothing
                    @printf("  ρ=%.4g  no seed equilibrium\n", ρ)
                    continue
                end
                br = (po=eq.po, s=eq.s)
                at_edge = eq.at_edge
                src = "seed-" * string(eq.source)
            else
                br = (po=seed.po, s=seed.s)
                src = "seed-pi"
            end
        elseif reeq_every > 0 && mod(k - 1, reeq_every) == 0
            eq = try
                equilibrium_at(c, P, κ; Q_warm=br.s.Q)
            catch err
                @printf("  ρ=%.4g  equilibrium_at threw %s\n", ρ, err)
                nothing
            end
            if eq !== nothing && abs(eq.po - br.po) <= 0.10
                br = (po=eq.po, s=eq.s)
                at_edge = eq.at_edge
                src = string(eq.source)
            elseif eq !== nothing && eq.s.F < br.s.F - 1.0
                # global jumped to a different basin — still record it
                br = (po=eq.po, s=eq.s)
                at_edge = eq.at_edge
                src = "jump-" * string(eq.source)
            end
        end
        if src == "continue"
            nxt = try
                continue_pi_min(c, P, κ, br.po, br.s.Q; window=0.05,
                    win=br.s.win, wout=br.s.wout)
            catch err
                @printf("  ρ=%.4g  continue threw %s\n", ρ, err)
                nothing
            end
            # free-solve competitor (may jump basins — keep only if π stays close)
            best = nxt
            try
                for cand in free_solve_candidates(c, P, κ; Q_warm=br.s.Q)
                    abs(cand.po - br.po) > 0.08 && continue
                    if best === nothing || cand.s.F < best.F
                        best = (po=cand.po, s=cand.s, at_edge=false, F=cand.s.F, F0=NaN, dF=NaN)
                        src = "free"
                    end
                end
            catch err
                @printf("  ρ=%.4g  free_solve threw %s\n", ρ, err)
            end
            if best === nothing
                @printf("  ρ=%.4g  branch died (π was %.4g)\n", ρ, br.po)
                # try full re-seed
                eq = try
                    equilibrium_at(c, P, κ)
                catch
                    nothing
                end
                eq === nothing && continue
                br = (po=eq.po, s=eq.s)
                at_edge = eq.at_edge
                src = "reseed-" * string(eq.source)
            else
                br = (po=best.po, s=best.s)
                at_edge = best.at_edge
                src == "continue" || nothing
            end
        end
        s = br.s
        Vout = s.nwells < 2 ? NaN : s.wout[3]
        muout = s.nwells < 2 ? NaN : s.wout[1]
        sigout = s.nwells < 2 ? NaN : s.wout[2]
        dV = s.nwells < 2 ? NaN : s.dV_raw
        lam = (s.L3 < 1 && s.L3 > 0) ? (s.L3 * ρ / (1 - s.L3)) : NaN
        @printf(io, "%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.10g,%d,%d,%s,%.6g\n",
            ρ, κ, br.po, s.win[1], s.win[2], muout, sigout, s.win[3], Vout, dV, s.L3, s.F,
            s.nwells, at_edge ? 1 : 0, src, lam)
        flush(io)
        @printf("  ρ=%-8.4g  π=%-7.4g  L3=%-7.4g  λ≈%-7.4g  V_out=%+-8.3g  n=%d  [%s]%s\n",
            ρ, br.po, s.L3, lam, Vout, s.nwells, src, at_edge ? " EDGE" : "")
        flush(stdout)
    end
    close(io)
    println("saved ", out)
end

if "--rho-deep" in ARGS
    rho_deep_main()
elseif "--rho-branch-scan" in ARGS
    rho_branch_scan_main()
elseif "--rho-pi-scan" in ARGS
    rho_pi_scan_main()
elseif "--rho-ray-actions" in ARGS
    rho_ray_actions_main()
elseif "--theory-ray" in ARGS
    theory_ray_main()
elseif "--rho-learn" in ARGS
    rho_learn_main()
elseif "--d-alpha-scan" in ARGS
    d_alpha_scan_main()
elseif "--ray-scan" in ARGS
    ray_scan_main()
elseif "--ray-F-scan" in ARGS
    ray_F_scan_main()
elseif "--pi-scan" in ARGS
    pi_scan_main()
elseif "--dump-ray-actions" in ARGS
    dump_ray_actions_main()
elseif "--dump-action" in ARGS || "--dump-pdf" in ARGS
    dump_action_main()
elseif "--F-scan" in ARGS
    f_scan_main()
elseif "--scan" in ARGS
    scan_main()
else
    main()
end
