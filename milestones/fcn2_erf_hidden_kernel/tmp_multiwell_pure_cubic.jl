# Pure-cubic multi-well matrix Laplace (He1–He3 basis, teacher y = (0, ε)).
#
# Same V and mixture as multiwell_laplace.jl, but the target in G, F, and
# learnability is y = (0, ε) instead of (1, ε). The linear channel stays in
# the 2×2 kernel (Onsager / induced He1); it just has no teacher weight.
#
#   julia tmp_multiwell_pure_cubic.jl --scan --d 10 --n1 400 --chi 400 --sa0 1 \
#       --eps 1 --Ps 50,100,200,500,1000,2000,5000,10000 --kappas 0.5
#
# Reports dV = V(μ_out) − V(μ_in) − log(σ_out/σ_in). Outer wells are the
# Hamiltonian global mins when dV < 0 (and π_out → 1 under Boltzmann masses).

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

const PO_GRID = vcat(0.0, 10 .^ range(-5, -1, length=17), range(0.12, 1.0, length=23))

function make_cfg()
    d = parse(Float64, argval("--d", "10"))
    s0 = 1.0
    Cfg(d, parse(Float64, argval("--n1", "400")), parse(Float64, argval("--chi", "400")),
        parse(Float64, argval("--sa0", "1")), parse(Float64, argval("--eps", "1")), s0,
        1.0 + 2.0 * (d - 1.0) * s0 / d)
end

function scan_main()
    c = make_cfg()
    Ps = parse_list(argval("--Ps", "50,100,200,500,1000,2000,5000,10000"))
    κs = parse_list(argval("--kappas", "0.5"))
    length(κs) == 1 && (κs = fill(κs[1], length(Ps)))
    out = argval("--out", "tmp_pure_cubic_scan_d$(Int(c.d)).csv")
    eqout = replace(out, ".csv" => "_equilibrium.csv")
    eqio = open(eqout, "w")
    println(eqio, "P,kappa,pi_out,F,f1,L3,mu_in,sig_in,V_in,mu_out,sig_out,V_out,dV,dV_raw,nwells")
    @printf("pure cubic y=(0, ε=%.4g); d=%g N=%g χ=%g sa0=%g\n", c.eps, c.d, c.n1, c.chi, c.a0)
    open(out, "w") do io
        println(io, "P,kappa,pi_out,F,f1,L3,mu_in,sig_in,V_in,mu_out,sig_out,V_out,dV,dV_raw,nwells,Q11,Q13,Q33")
        Qeq_prev = nothing
        for (P, κ) in zip(Ps, κs)
            Qprev = seed_Q(c, 0.0, 0.5)
            best = nothing
            grid = Tuple{Float64,Any}[]
            for po in PO_GRID
                s = nothing
                Q0s = Any[Qprev, seed_Q(c, 0.0, 0.5), seed_Q(c, 2.0, 0.3),
                    0.5 * seed_Q(c, 0.0, 0.5) + 0.5 * seed_Q(c, 2.0, 0.3)]
                Qeq_prev === nothing || push!(Q0s, Qeq_prev)
                for Q0 in Q0s
                    s = solve_fixed_pi(c, po, Q0, P, κ)
                    s === nothing || break
                end
                if s === nothing
                    @printf(io, "%.10g,%.10g,%.6g,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,NaN,NaN,NaN\n",
                        P, κ, po)
                    continue
                end
                Qprev = s.Q
                push!(grid, (po, s))
                @printf(io, "%.10g,%.10g,%.6g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g\n",
                    P, κ, po, s.F, s.f1, s.L3, s.win[1], s.win[2], s.win[3],
                    s.wout[1], s.wout[2], s.wout[3], s.dV, s.dV_raw, s.nwells,
                    s.Q[1, 1], s.Q[1, 2], s.Q[2, 2])
                (best === nothing || s.F < best[2].F) && (best = (po, s))
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
            if !isempty(mins)
                best = mins[argmin([m[2].F for m in mins])]
                po, s = best
                Qeq_prev = s.Q
                @printf(eqio, "%.10g,%.10g,%.6g,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n",
                    P, κ, po, s.F, s.f1, s.L3, s.win[1], s.win[2], s.win[3],
                    s.wout[1], s.wout[2], s.wout[3], s.dV, s.dV_raw, s.nwells)
                flush(eqio)
                # dV ≈ 0 at the free-energy root by construction; Hamiltonian
                # ordering is dV_raw = V(out) − V(in). Outer is global when dV_raw < 0.
                mark = s.dV_raw < 0 ? "OUTER_HAMILTONIAN_MIN" : "inner_hamiltonian_min"
                @printf("d=%g P=%-7.0f π_out=%.4g  L3=%.4f f1=%.4f  in μ=%.3f V=%.4g  out μ=%.3f V=%.4g  dV_raw=%.4g  %s\n",
                    c.d, P, po, s.L3, s.f1, s.win[1], s.win[3], s.wout[1], s.wout[3], s.dV_raw, mark)
            else
                @printf("d=%g P=%-7.0f nothing converged\n", c.d, P)
            end
            flush(stdout)
        end
    end
    close(eqio)
    println("saved ", out, " and ", eqout)
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

"--scan" in ARGS ? scan_main() : main()
