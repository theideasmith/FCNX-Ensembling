#!/usr/bin/env julia
# P-ray: He2 learnability L2 for y=(0,ε) vs y=(1,ε), He1–He2 Laplace.
# Full Q,G; ± parity mixture. Soft coupling only via shared V(w).
#
#   julia --project=../.. probe_he12_learnability_ray.jl

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, Printf, SpecialFunctions

const GH_X = Float64[-7.619048541679758, -6.510590157013654, -5.5787388058932015,
                     -4.734581334046055, -3.9439673506573163, -3.1890148165533896,
                     -2.458663611172368, -1.7452473208141268, -1.042945348802751,
                     -0.3469641570813559, 0.3469641570813559, 1.042945348802751,
                     1.7452473208141268, 2.458663611172368, 3.1890148165533896,
                     3.9439673506573163, 4.734581334046055, 5.5787388058932015,
                     6.510590157013654, 7.619048541679758]
const GH_W = Float64[3.1528387293827914e-13, 6.221607696776643e-10,
    1.5359340338199267e-07, 1.103448119312213e-05, 0.00032291959586810135,
    0.004587388253857086, 0.035087375128591974, 0.15417361128553891,
    0.4054203876842811, 0.6537112666702904, 0.6537112666702904,
    0.4054203876842811, 0.15417361128553891, 0.035087375128591974,
    0.004587388253857086, 0.00032291959586810135, 1.103448119312213e-05,
    1.5359340338199267e-07, 6.221607696776643e-10, 3.1528387293827914e-13] ./ sqrt(2π)

struct Cfg
    d::Float64; n1::Float64; chi::Float64; a0::Float64; eps::Float64
    s0::Float64; T::Float64; z::Float64
end

function features(w, c::Cfg)
    c1, _ = hermite_point_features(w, c.T)
    α = abs(w) / sqrt(c.T)
    e2 = 0.0
    for i in eachindex(GH_X)
        ξ = GH_X[i]
        e2 += GH_W[i] * erf(α * ξ + c.z) * (ξ * ξ - 1.0)
    end
    return c1, e2 / sqrt(2.0)
end

function kernel_block(μ, σ, c::Cfg)
    q11 = q22 = q12 = 0.0
    nodes, weights = FCS2_VGA.GH_NODES, FCS2_VGA.GH_WEIGHTS
    for i in 1:length(nodes)
        w = μ + σ * nodes[i]
        c1, c2 = features(w, c)
        q11 += weights[i] * c1 * c1
        q22 += weights[i] * c2 * c2
        q12 += weights[i] * c1 * c2
    end
    return [q11 q12; q12 q22]
end

function amp_y(c::Cfg, Q, P, κ, y)
    chi_p = c.chi / c.a0
    ρp = κ / (c.a0 * P)
    qscale = abs(Q[1, 1]) + abs(Q[2, 2])
    Qreg = Matrix(Q) + (1e-10 * (1 + qscale)) * I(2)
    v = (Qreg + ρp * I(2)) \ y
    G = -(chi_p^2) * (v * v') + chi_p * inv(Qreg)
    return (c.a0 / (c.n1 * c.chi)) * G
end

function potential(c::Cfg, G)
    w -> begin
        c1, c2 = features(w, c)
        (c.d / c.s0) * 0.5 * w * w +
            0.5 * (G[1, 1] * c1^2 + 2G[1, 2] * c1 * c2 + G[2, 2] * c2^2)
    end
end

const WGRID = range(0.0, 20.0; length=8001)

function wells(V)
    d1 = w -> ForwardDiff.derivative(V, w)
    d2 = w -> ForwardDiff.derivative(d1, w)
    v = try
        V.(WGRID)
    catch
        return Tuple{Float64,Float64,Float64,Int}[]
    end
    any(!isfinite, v) && return Tuple{Float64,Float64,Float64,Int}[]
    out = Tuple{Float64,Float64,Float64,Int}[]
    if v[1] <= v[2]
        h0 = try; d2(0.0); catch; NaN; end
        if isfinite(h0) && h0 > 0
            push!(out, (0.0, 1 / sqrt(h0), v[1], 1))
        end
    end
    for j in 2:length(WGRID)-1
        (v[j] < v[j-1] && v[j] <= v[j+1]) || continue
        w = Float64(WGRID[j])
        for _ in 1:30
            h = try; d2(w); catch; NaN; end
            (isfinite(h) && h > 0) || break
            g = try; d1(w); catch; NaN; end
            isfinite(g) || break
            w -= g / h
        end
        h = try; d2(w); catch; NaN; end
        (isfinite(h) && h > 0 && w > 1e-5) || continue
        Vm = try; V(w); catch; NaN; end
        isfinite(Vm) || continue
        push!(out, (w, 1 / sqrt(h), Vm, 2))
    end
    return out
end

function mixture(c::Cfg, ws)
    logZk = [log(m * s) - Vm for (μ, s, Vm, m) in ws]
    lmax = maximum(logZk)
    wt = exp.(logZk .- lmax)
    wt ./= sum(wt)
    Q = zeros(2, 2)
    for (k, (μ, s, _, _)) in enumerate(ws)
        if abs(μ) < 1e-12
            Q .+= wt[k] * kernel_block(0.0, s, c)
        else
            Q .+= wt[k] * 0.5 * (kernel_block(μ, s, c) + kernel_block(-μ, s, c))
        end
    end
    return Q, wt
end

function solve_y(c::Cfg, P, κ, y; iters=2000, η=0.25, tol=1e-8)
    Q = kernel_block(0.0, 0.3, c)
    for _ in 1:iters
        G = amp_y(c, Q, P, κ, y)
        Vfn = potential(c, G)
        ws = wells(Vfn)
        isempty(ws) && return nothing
        Qn, _ = mixture(c, ws)
        δ = norm(Qn - Q) / max(norm(Q), 1e-30)
        Q = (1 - η) * Q + η * Qn
        if δ < tol
            G = amp_y(c, Q, P, κ, y)
            Vfn = potential(c, G)
            ws = wells(Vfn)
            Q, _ = mixture(c, ws)
            return (Q=Q, G=G, V=Vfn, ws=ws)
        end
    end
    G = amp_y(c, Q, P, κ, y)
    Vfn = potential(c, G)
    ws = wells(Vfn)
    isempty(ws) && return nothing
    Q, _ = mixture(c, ws)
    return (Q=Q, G=G, V=Vfn, ws=ws)
end

function learnab(Q, y, P, κ, a0, eps)
    ρp = κ / (a0 * P)
    f = Q * ((Matrix(Q) + ρp * I) \ y)
    return f[1], f[2] / eps
end

function classify(ws)
    isempty(ws) && return "fail"
    μs = [w[1] for w in ws]
    has0 = any(abs(μ) < 1e-6 for μ in μs)
    secondary = any(μ > 1e-3 for μ in μs)
    !has0 && secondary && return "INNER_SPLIT"
    has0 && secondary && return "OUTER_PLUS_CENTER"
    has0 && return "center_only"
    return "other"
end

function run_ray(; d=150.0, n1=200.0, chi=200.0, a0=0.03, eps=1.0,
                 kappa=0.6, z=0.3, T=1.0,
                 Ps=(50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0))
    outdir = joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN")
    mkpath(outdir)
    c = Cfg(d, n1, chi, a0, eps, 1.0, T, z)
    rows = String[]

    @printf("P-ray He1–He2 Laplace  z=%g κ=%g χ=%g a0=%g ε=%g d=%g\n",
        z, kappa, chi, a0, eps, d)
    @printf("%-8s %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-8s %s\n",
        "P", "ρ", "L2(0,ε)", "L2(1,ε)", "ΔL2", "L2|Q0", "f1(1,ε)", "Q22_0/Q22_1", "nwell", "notes")

    for P in Ps
        ρ = kappa / (a0 * P)
        y0 = [0.0, eps]
        y1 = [1.0, eps]
        r0 = solve_y(c, P, kappa, y0)
        r1 = solve_y(c, P, kappa, y1)
        if r0 === nothing || r1 === nothing
            @printf("%-8.4g FAIL\n", P)
            continue
        end
        _, L2_0 = learnab(r0.Q, y0, P, kappa, a0, eps)
        f1_1, L2_1 = learnab(r1.Q, y1, P, kappa, a0, eps)
        _, L2_1_on_Q0 = learnab(r0.Q, y1, P, kappa, a0, eps)
        dL = L2_1 - L2_0
        n0, n1 = length(r0.ws), length(r1.ws)
        note = @sprintf("%s|%s", classify(r0.ws), classify(r1.ws))
        @printf("%-8.4g %-8.4g %-10.5f %-10.5f %-10.3g %-10.5f %-10.5f %-10s %-8s %s\n",
            P, ρ, L2_0, L2_1, dL, L2_1_on_Q0, f1_1,
            @sprintf("%.3g/%.3g", r0.Q[2, 2], r1.Q[2, 2]), "$n0/$n1", note)
        push!(rows, @sprintf("%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%d,%s",
            P, ρ, L2_0, L2_1, dL, L2_1_on_Q0, f1_1,
            r0.Q[2, 2], r1.Q[2, 2], r0.Q[1, 2], n0, n1, note))
        flush(stdout)
    end

    csv = joinpath(outdir, "he12_learnability_ray.csv")
    open(csv, "w") do io
        println(io, "P,rho,L2_y0,L2_y1,dL2,L2_y1_on_Q0,f1_y1,Q22_y0,Q22_y1,Q12_y0,nwells_y0,nwells_y1,note")
        foreach(r -> println(io, r), rows)
    end
    println("wrote $csv")
end

run_ray()
println("DONE")
