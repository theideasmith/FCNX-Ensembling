#!/usr/bin/env julia
# Pure He2 (even) vs He3 (odd) single-channel cavity action: inner split?
#
#   φ₂(w) = w² - 1     (even; φ₂'(0)=0)
#   φ₃(w) = w³ - 3w    (odd;  φ₃'(0)=-3)
#   V(w)  = (d/2) w² + ½ G φ(w)²
#   G     = a0/(n1 χ) [ -χ'² v² + χ'/λ ], v=ε/(λ+ρ'), χ'=χ/a0   (δ=1)
#
#   julia --project=../.. probe_he2_teacher_action.jl

using Printf, ForwardDiff, Random

phi_he2(w) = w * w - 1.0
phi_he3(w) = w * w * w - 3.0 * w

function kernel_lambda(phi, μ, σ; nsamp=20000)
    # Monte Carlo E[φ(μ+σZ)²], Z~N(0,1)
    Random.seed!(0)
    s = 0.0
    for _ in 1:nsamp
        w = μ + σ * randn()
        s += phi(w)^2
    end
    return s / nsamp
end

function amp_G(; chi, n1, a0, kappa, P, eps, λ)
    chi_p = chi / a0
    ρp = kappa / (a0 * P)
    v = eps / (λ + ρp)
    return (a0 / (n1 * chi)) * (-(chi_p^2) * v * v + chi_p / max(λ, 1e-14))
end

potential(phi, G, d) = w -> 0.5 * d * w * w + 0.5 * G * phi(w)^2

function find_wells(V; wmax=8.0, n=6001)
    ws = range(0.0, wmax; length=n)
    vals = V.(ws)
    d2 = w -> ForwardDiff.derivative(x -> ForwardDiff.derivative(V, x), w)
    out = NamedTuple{(:μ, :σ, :V),Tuple{Float64,Float64,Float64}}[]
    if vals[1] <= vals[2]
        h0 = d2(0.0)
        h0 > 0 && push!(out, (μ=0.0, σ=1/sqrt(h0), V=vals[1]))
    end
    for j in 2:length(ws)-1
        (vals[j] < vals[j-1] && vals[j] <= vals[j+1]) || continue
        w = Float64(ws[j])
        for _ in 1:50
            h = d2(w); (isfinite(h) && h > 0) || break
            g = ForwardDiff.derivative(V, w); isfinite(g) || break
            w -= g / h
        end
        h = d2(w)
        (isfinite(h) && h > 0 && w > 1e-5) || continue
        push!(out, (μ=w, σ=1/sqrt(h), V=V(w)))
    end
    return out
end

function equilibrium(phi; d=150.0, n1=200.0, chi=200.0, a0=0.03, P=1000.0,
                     eps=1.0, ρ=0.05, iters=30)
    κ = ρ * a0 * P
    # start from a narrow central well
    μ, σ = 0.0, 0.3
    local wells, G, λ, V
    πout = 0.0
    for _ in 1:iters
        λ = kernel_lambda(phi, μ, σ)
        G = amp_G(; chi, n1, a0, kappa=κ, P, eps, λ)
        V = potential(phi, G, d)
        wells = find_wells(V)
        isempty(wells) && return nothing
        if length(wells) == 1
            μ, σ = wells[1].μ, wells[1].σ
            πout = 0.0
        else
            win, wout = wells[1], wells[end]
            # Boltzmann weights: center mult=1, outer mult=2
            lin = log(max(win.σ, 1e-14)) - win.V
            lout = log(max(2 * wout.σ, 1e-14)) - wout.V
            m = max(lin, lout)
            wn, wo = exp(lin - m), exp(lout - m)
            πout = wo / (wn + wo)
            λ = (1 - πout) * kernel_lambda(phi, win.μ, win.σ) +
                πout * kernel_lambda(phi, wout.μ, wout.σ)
            G = amp_G(; chi, n1, a0, kappa=κ, P, eps, λ)
            V = potential(phi, G, d)
            wells = find_wells(V)
            isempty(wells) && return nothing
            μ, σ = wells[1].μ, wells[1].σ
        end
    end
    vpp0 = ForwardDiff.derivative(x -> ForwardDiff.derivative(V, x), 0.0)
    μout = length(wells) > 1 ? wells[end].μ : NaN
    note = if length(wells) < 2
        "no_outer"
    elseif abs(wells[1].μ) < 1e-6
        "OUTER_ONLY"
    else
        "INNER_SPLIT"
    end
    return (nwells=length(wells), μin=wells[1].μ, μout=μout, πout=πout,
            Vpp0=vpp0, G=G, λ=λ, note=note)
end

function scan(phi, tag; d=150.0, eps=1.0, rhos=0.15:-0.01:0.02)
    @printf("\n=== %s  d=%g ε=%g ===\n", tag, d, eps)
    @printf("%-8s %-7s %-10s %-8s %-8s %-8s %s\n",
        "ρ", "nwells", "V''(0)", "μ_in", "μ_out", "π_out", "note")
    for ρ in rhos
        r = equilibrium(phi; d=d, eps=eps, ρ=ρ)
        if r === nothing
            @printf("%-8.3g  FAIL\n", ρ)
            continue
        end
        @printf("%-8.3g %-7d %-10.4g %-8.3g %-8.3g %-8.4g %s\n",
            ρ, r.nwells, r.Vpp0, r.μin, r.μout, r.πout, r.note)
        flush(stdout)
    end
end

# Analytic He2 curvature
@printf("He2 analytic: φ=w²-1 ⇒ φ²=(w²-1)²=1-2w²+w⁴\n")
@printf("  V=(d/2)w²+(G/2)φ² ⇒ V''(0)=d-2G\n")
@printf("  Teacher-driven G<0 ⇒ V''(0)=d+|2G|>0  (center ALWAYS a min)\n\n")

scan(phi_he2, "pure He2 (even) teacher")
scan(phi_he3, "pure He3 (odd) teacher — control")

# Sweep G directly for both
@printf("\n=== direct G sweep (d=150): does center leave 0? ===\n")
for (phi, tag) in ((phi_he2, "He2"), (phi_he3, "He3"))
    @printf("-- %s --\n", tag)
    for G in (0.0, -1.0, -10.0, -100.0, -1e3, -1e4)
        V = potential(phi, G, 150.0)
        vpp = ForwardDiff.derivative(x -> ForwardDiff.derivative(V, x), 0.0)
        wells = find_wells(V)
        @printf("  G=%9.3g  V''(0)=%10.4g  n=%d  μ_in=%.4g\n",
            G, vpp, length(wells), isempty(wells) ? NaN : wells[1].μ)
    end
end
println("\nDONE")
