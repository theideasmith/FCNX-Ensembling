#!/usr/bin/env julia
# Decoupled He1–He2 matrix Laplace (NO mixing: Q12 = G12 = 0).
#
# Same equations as tmp_multiwell_pure_cubic.jl:
#   y = (0, ε),  G = a0/(n1 χ)[ -χ'² vvᵀ + χ' Q⁻¹ ],  V = prior + ½ cᵀ G c
# but on (c1, c2) with parity grading and forced off-diagonals zero.
#
# Features (definite parity in w):
#   c1(w) = stock erf He1  (odd)   — FCS2_VGA.hermite_point_features
#   c2(w) = ⟨erf(α(|w|) ξ + z), He₂(ξ)⟩ / √2!   (even; z opens the channel)
# Under ±-symmetric ν: Q12 = E[c1 c2] = 0 automatically; we also zero G12.
#
#   julia --project=../.. probe_he12_decoupled_laplace.jl

include(joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl"))
using .FCS2_VGA
using ForwardDiff, LinearAlgebra, Printf, SpecialFunctions

# Probabilists GH20 (numpy hermegauss), E[f]=Σ w f / √(2π)
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

"""Odd He1 (stock) + even He2 from ⟨erf(α(|w|)ξ+z), He2⟩/√2."""
function features(w, c::Cfg)
    c1, _ = hermite_point_features(w, c.T)
    α = abs(w) / sqrt(c.T)
    e2 = 0.0
    for i in eachindex(GH_X)
        ξ = GH_X[i]
        e2 += GH_W[i] * erf(α * ξ + c.z) * (ξ * ξ - 1.0)
    end
    c2 = e2 / sqrt(2.0)
    return c1, c2
end

"""Diagonal kernel block under N(μ,σ²) (one well); Q12 forced 0."""
function kernel_diag(μ, σ, c::Cfg)
    # GH on the well (reuse FCS2 nodes via Simple: 11-pt from module)
    q11 = 0.0
    q22 = 0.0
    # use same 11-pt as compute_lambda1 for consistency on the well
    nodes = FCS2_VGA.GH_NODES
    weights = FCS2_VGA.GH_WEIGHTS
    for i in 1:length(nodes)
        w = μ + σ * nodes[i]
        c1, c2 = features(w, c)
        q11 += weights[i] * c1 * c1
        q22 += weights[i] * c2 * c2
    end
    return [q11 0.0; 0.0 q22]
end

"""Same amp as multiwell pure-cubic, with y=(0,ε) and G12≡0."""
function amp(c::Cfg, Q, P, κ)
    chi_p = c.chi / c.a0
    ρp = κ / (c.a0 * P)
    Qd = [Q[1, 1] 0.0; 0.0 Q[2, 2]]
    y = [0.0, c.eps]
    qscale = abs(Qd[1, 1]) + abs(Qd[2, 2])
    Qreg = Qd + (1e-10 * (1 + qscale)) * I(2)
    # Guard silent He2: if Q22 tiny, teacher channel is invisible
    if Qreg[2, 2] < 1e-18
        # only Onsager on He1; He2 dead
        G = (c.a0 / (c.n1 * c.chi)) * [chi_p / Qreg[1, 1] 0.0; 0.0 0.0]
        return G
    end
    v = (Qreg + ρp * I(2)) \ y
    G = -(chi_p^2) * (v * v') + chi_p * inv(Qreg)
    G = (c.a0 / (c.n1 * c.chi)) * G
    return [G[1, 1] 0.0; 0.0 G[2, 2]]   # force no mixing
end

function potential(c::Cfg, G)
    w -> begin
        c1, c2 = features(w, c)
        (c.d / c.s0) * 0.5 * w * w + 0.5 * (G[1, 1] * c1^2 + G[2, 2] * c2^2)
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
    Q = sum(wt[k] * kernel_diag(ws[k][1], ws[k][2], c) for k in eachindex(ws))
    # parity: kill any numerical Q12
    return [Q[1, 1] 0.0; 0.0 Q[2, 2]], wt
end

function solve(c::Cfg, P, κ; iters=2000, η=0.25, tol=1e-8)
    # init: narrow central well
    Q = kernel_diag(0.0, 0.3, c)
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
    n = length(ws)
    has0 = any(abs(μ) < 1e-6 for μ in μs)
    # any secondary well at μ>0 with center still present = outer (no inner split)
    secondary = any(μ > 1e-3 for μ in μs)
    if !has0 && secondary
        return "INNER_SPLIT"   # left the origin
    elseif has0 && secondary
        return "OUTER_PLUS_CENTER"
    elseif secondary && !has0
        return "OUTER_ONLY"
    elseif has0 && n == 1
        return "center_only"
    else
        return "other"
    end
end

function run(; d=150.0, n1=200.0, chi=200.0, a0=0.03, P=1000.0,
             eps=1.0, s0=1.0, T=1.0, zs=(0.0, 0.2, 0.5),
             rhos=0.15:-0.01:0.02)
    outdir = joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN")
    mkpath(outdir)
    action_rows = String[]
    sum_rows = String[]

    @printf("Decoupled He1–He2 Laplace  d=%g χ=%g a0=%g P=%g ε=%g\n", d, chi, a0, P, eps)
    @printf("(Q12=G12=0 always)\n")
    @printf("%-6s %-8s %-8s %-10s %-8s %-6s %-8s %s\n",
        "z", "ρ", "Q22", "G22", "V''(0)", "nwell", "μ_in", "note")

    for z in zs
        for ρ in rhos
            κ = ρ * a0 * P
            c = Cfg(d, n1, chi, a0, eps, s0, T, z)
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
            μout = length(r.ws) > 1 ? r.ws[end][1] : NaN
            @printf("%-6.2g %-8.3g %-8.3g %-10.4g %-8.3g %-6d %-8.3g %s\n",
                z, ρ, r.Q[2, 2], r.G[2, 2], vpp0, length(r.ws), μin, note)

            for ww in range(-4.0, 4.0; length=401)
                push!(action_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%d,%s",
                    z, ρ, ww, r.V(ww), length(r.ws), note))
            end
            push!(sum_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%s",
                z, ρ, r.Q[1, 1], r.Q[2, 2], r.G[2, 2], vpp0, μin, length(r.ws), note))
            flush(stdout)
        end
    end

    open(joinpath(outdir, "he12_decoupled_actions.csv"), "w") do io
        println(io, "z,rho,w,V,nwells,note")
        for r in action_rows; println(io, r); end
    end
    open(joinpath(outdir, "he12_decoupled_summary.csv"), "w") do io
        println(io, "z,rho,Q11,Q22,G22,Vpp0,mu_in,nwells,note")
        for r in sum_rows; println(io, r); end
    end
    println("wrote he12_decoupled_*.csv")
end

# sanity: parity + z=0 ⇒ c2≈0; z≠0 ⇒ c2 even
c0 = Cfg(150.0, 200.0, 200.0, 0.03, 1.0, 1.0, 1.0, 0.0)
c2 = Cfg(150.0, 200.0, 200.0, 0.03, 1.0, 1.0, 1.0, 0.3)
for w in (0.0, 0.5, 1.0, -0.5, -1.0)
    a1, a2 = features(w, c0)
    b1, b2 = features(w, c2)
    @printf("w=%5.1f  z=0: c1=%8.4f c2=%9.2e   z=0.3: c1=%8.4f c2=%8.4f\n",
        w, a1, a2, b1, b2)
end
@printf("even check z=0.3: c2(1)=%.6f c2(-1)=%.6f\n",
    features(1.0, c2)[2], features(-1.0, c2)[2])

run()
println("DONE")
