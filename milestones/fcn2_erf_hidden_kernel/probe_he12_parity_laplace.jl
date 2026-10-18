#!/usr/bin/env julia
# He1–He2 matrix Laplace with FULL Q,G (no artificial zeroing).
#
# Parity alone: c1 odd, c2 even ⇒ c1 c2 odd ⇒ under ±-symmetric ν, Q12 → 0.
# Same amp as tmp_multiwell_pure_cubic: y=(0,ε), G from (Q+ρ'I)^{-1}y, no G12 hack.
#
#   julia --project=../.. probe_he12_parity_laplace.jl

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

"""c1 odd (stock erf), c2 even via ⟨erf(α(|w|)ξ+z), He2⟩/√2."""
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

"""Full 2×2 Q = E[c cᵀ] under N(μ,σ²). No off-diag zeroing."""
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

"""Same amp as multiwell pure-cubic — full G, no G12 hack."""
function amp(c::Cfg, Q, P, κ)
    chi_p = c.chi / c.a0
    ρp = κ / (c.a0 * P)
    y = [0.0, c.eps]
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
        push!(out, (w, 1 / sqrt(h), Vm, 2))  # multiplicity 2 for ± pair
    end
    return out
end

"""Mixture with ± multiplicity. Outer wells: average K(+μ)+K(-μ) so odd
integrands (c1 c2) cancel — parity, not an artificial Q12=0."""
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
            # ± pair represented by one μ≥0 well with multiplicity 2
            Q .+= wt[k] * 0.5 * (kernel_block(μ, s, c) + kernel_block(-μ, s, c))
        end
    end
    return Q, wt
end

function solve(c::Cfg, P, κ; iters=2000, η=0.25, tol=1e-8)
    Q = kernel_block(0.0, 0.3, c)
    local ws, G, Vfn, π
    for it in 1:iters
        G = amp(c, Q, P, κ)
        Vfn = potential(c, G)
        ws = wells(Vfn)
        isempty(ws) && return nothing
        Qn, π = mixture(c, ws)
        δ = norm(Qn - Q) / max(norm(Q), 1e-30)
        Q = (1 - η) * Q + η * Qn
        if δ < tol
            G = amp(c, Q, P, κ)
            Vfn = potential(c, G)
            ws = wells(Vfn)
            Q, π = mixture(c, ws)
            return (Q=Q, G=G, V=Vfn, ws=ws, π=π, iters=it)
        end
    end
    G = amp(c, Q, P, κ)
    Vfn = potential(c, G)
    ws = wells(Vfn)
    isempty(ws) && return nothing
    Q, π = mixture(c, ws)
    return (Q=Q, G=G, V=Vfn, ws=ws, π=π, iters=iters)
end

function classify(ws)
    isempty(ws) && return "fail"
    μs = [w[1] for w in ws]
    has0 = any(abs(μ) < 1e-6 for μ in μs)
    secondary = any(μ > 1e-3 for μ in μs)
    if !has0 && secondary
        return "INNER_SPLIT"
    elseif has0 && secondary
        return "OUTER_PLUS_CENTER"
    elseif has0
        return "center_only"
    else
        return "other"
    end
end

function run(; d=150.0, n1=200.0, chi=200.0, a0=0.03, P=1000.0,
             eps=1.0, zs=(0.0, 0.2, 0.5), rhos=0.15:-0.01:0.02)
    outdir = joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN")
    mkpath(outdir)
    action_rows = String[]
    sum_rows = String[]

    @printf("He1–He2 Laplace FULL Q,G (parity only)  d=%g χ=%g\n", d, chi)
    @printf("%-6s %-8s %-10s %-10s %-10s %-8s %-6s %s\n",
        "z", "ρ", "Q12", "G12", "V''(0)", "μ_in", "nwell", "note")

    for z in zs, ρ in rhos
        κ = ρ * a0 * P
        c = Cfg(d, n1, chi, a0, eps, 1.0, 1.0, z)
        r = solve(c, P, κ)
        if r === nothing
            @printf("%-6.2g %-8.3g FAIL\n", z, ρ)
            continue
        end
        vpp0 = try
            ForwardDiff.derivative(x -> ForwardDiff.derivative(r.V, x), 0.0)
        catch
            NaN
        end
        note = classify(r.ws)
        μin = r.ws[1][1]
        @printf("%-6.2g %-8.3g %-10.3g %-10.3g %-10.3g %-8.3g %-6d %s\n",
            z, ρ, r.Q[1, 2], r.G[1, 2], vpp0, μin, length(r.ws), note)

        for ww in range(-4.0, 4.0; length=401)
            push!(action_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%d,%s",
                z, ρ, ww, r.V(ww), length(r.ws), note))
        end
        push!(sum_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%s",
            z, ρ, r.Q[1, 1], r.Q[2, 2], r.Q[1, 2], r.G[1, 2], vpp0, μin,
            length(r.ws), note))
        flush(stdout)
    end

    open(joinpath(outdir, "he12_parity_actions.csv"), "w") do io
        println(io, "z,rho,w,V,nwells,note")
        foreach(r -> println(io, r), action_rows)
    end
    open(joinpath(outdir, "he12_parity_summary.csv"), "w") do io
        println(io, "z,rho,Q11,Q22,Q12,G12,Vpp0,mu_in,nwells,note")
        foreach(r -> println(io, r), sum_rows)
    end
    println("wrote he12_parity_*.csv")
end

run()
println("DONE")
