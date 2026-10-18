#!/usr/bin/env julia
# Teacher y(x) = ε He₂(t̂·x), student = erf(α(w) ξ + z).
#
# Hermite convention matching FCS2_VGA hermite_point_features (b=0):
#   α(w) = w / √T
#   c_n(w) = E_ξ[erf(α ξ + z) He_n(ξ)] / √(n!)
# so c1|_{z=0} = √(4/π) w/√(T+2w²).
#
# For z=0, c2≡0 (parity) ⇒ pure He2 teacher is invisible.
# For z≠0, even channel opens; we solve the (He1,He2) matrix Laplace
# with y = (0, ε) and plot V(w).
#
#   julia --project=../.. probe_he2_target_teacher.jl

using LinearAlgebra, Printf, ForwardDiff, SpecialFunctions, Random

# Probabilists Hermite–Gauss 20-pt (numpy hermegauss). Weights integrate
# e^{-x²/2} dx, so E[f(Z)] = Σ w_i f(x_i) / √(2π).
const GH_X = Float64[-7.619048541679758, -6.510590157013654, -5.5787388058932015,
                     -4.734581334046055, -3.9439673506573163, -3.1890148165533896,
                     -2.458663611172368, -1.7452473208141268, -1.042945348802751,
                     -0.3469641570813559, 0.3469641570813559, 1.042945348802751,
                     1.7452473208141268, 2.458663611172368, 3.1890148165533896,
                     3.9439673506573163, 4.734581334046055, 5.5787388058932015,
                     6.510590157013654, 7.619048541679758]
const GH_W_RAW = Float64[3.1528387293827914e-13, 6.221607696776643e-10,
    1.5359340338199267e-07, 1.103448119312213e-05, 0.00032291959586810135,
    0.004587388253857086, 0.035087375128591974, 0.15417361128553891,
    0.4054203876842811, 0.6537112666702904, 0.6537112666702904,
    0.4054203876842811, 0.15417361128553891, 0.035087375128591974,
    0.004587388253857086, 0.00032291959586810135, 1.103448119312213e-05,
    1.5359340338199267e-07, 6.221607696776643e-10, 3.1528387293827914e-13]
const GH_W = GH_W_RAW ./ sqrt(2π)

He1(ξ) = ξ
He2(ξ) = ξ * ξ - 1.0

"""c1, c2 for erf(α ξ + z) with α = w/√T, orthonormal Hermite scale."""
function hermite_c12(w, T, z)
    α = w / sqrt(T)
    e1 = 0.0
    e2 = 0.0
    for i in eachindex(GH_X)
        ξ = GH_X[i]
        f = erf(α * ξ + z)
        e1 += GH_W[i] * f * He1(ξ)
        e2 += GH_W[i] * f * He2(ξ)
    end
    return e1 / 1.0, e2 / sqrt(2.0)   # /√(n!)
end

function Q_block(μ, σ, T, z; nsamp=8000)
    Random.seed!(0)
    q11 = q22 = q12 = 0.0
    for _ in 1:nsamp
        w = μ + σ * randn()
        c1, c2 = hermite_c12(w, T, z)
        q11 += c1 * c1
        q22 += c2 * c2
        q12 += c1 * c2
    end
    inv = 1 / nsamp
    return [q11*inv q12*inv; q12*inv q22*inv]
end

function amp_G(Q, y; chi, n1, a0, kappa, P)
    chi_p = chi / a0
    ρp = kappa / (a0 * P)
    # G = a0/(n1 χ) [ -χ'² v vᵀ + χ' Q⁻¹ ], v = (Q+ρ'I)⁻¹ y
    Qi = inv(Symmetric(Q + ρp * I))
    v = Qi * y
    Qinv = inv(Symmetric(Q + 1e-14 * I))
    return (a0 / (n1 * chi)) * (-(chi_p^2) * (v * v') + chi_p * Qinv)
end

function potential(G, T, z, d, s0)
    w -> begin
        c1, c2 = hermite_c12(w, T, z)
        c = [c1, c2]
        (d / (2 * s0)) * w * w + 0.5 * dot(c, G * c)
    end
end

function find_wells(V; wmin=-6.0, wmax=6.0, n=8001)
    ws = range(wmin, wmax; length=n)
    vals = V.(ws)
    d2 = w -> ForwardDiff.derivative(x -> ForwardDiff.derivative(V, x), w)
    out = NamedTuple{(:μ, :σ, :V),Tuple{Float64,Float64,Float64}}[]
    for j in 2:length(ws)-1
        (vals[j] < vals[j-1] && vals[j] < vals[j+1]) || continue
        w = Float64(ws[j])
        for _ in 1:40
            h = d2(w); (isfinite(h) && h > 0) || break
            g = ForwardDiff.derivative(V, w); isfinite(g) || break
            w -= g / h
        end
        h = d2(w)
        (isfinite(h) && h > 0) || continue
        push!(out, (μ=w, σ=1/sqrt(h), V=V(w)))
    end
    # dedupe
    uniq = typeof(out)()
    for well in sort(out; by=x -> x.μ)
        if isempty(uniq) || abs(well.μ - uniq[end].μ) > 1e-3
            push!(uniq, well)
        end
    end
    return uniq
end

function save_actions(path, rows)
    open(path, "w") do io
        println(io, "z,rho,w,V,nwells,note")
        for r in rows
            println(io, r)
        end
    end
end

function run(; d=150.0, n1=200.0, chi=200.0, a0=0.03, P=1000.0,
             eps=1.0, s0=1.0, T=1.0, zs=(0.0, 0.2, 0.5),
             rhos=(0.10, 0.05, 0.03, 0.02))
    outdir = joinpath(@__DIR__, "action_h0_activation_plots/PureCubicChiN")
    mkpath(outdir)
    csv_rows = String[]
    action_rows = String[]  # for python plot: z,rho,w,V

    @printf("Teacher y = (0, ε) in (He1, He2), ε=%g, d=%g\n", eps, d)
    @printf("%-6s %-8s %-8s %-8s %-8s %-6s %s\n",
        "z", "ρ", "Q22", "G22", "V''(0)", "nwell", "note")

    for z in zs
        for ρ in rhos
            κ = ρ * a0 * P
            # self-consistent: start narrow at 0
            μ, σ = 0.0, 0.35
            local G, Q, V, wells
            for _ in 1:25
                Q = Q_block(μ, σ, T, z)
                y = [0.0, eps]
                G = amp_G(Q, y; chi, n1, a0, kappa=κ, P)
                V = potential(G, T, z, d, s0)
                wells = find_wells(V)
                isempty(wells) && break
                # pick deepest well for next (μ,σ); mixture of all wells for Q
                # rebuild Q as mixture of Laplace wells with Boltzmann weights
                if length(wells) == 1
                    μ, σ = wells[1].μ, wells[1].σ
                else
                    logs = [log(max(w.σ, 1e-14)) - w.V for w in wells]
                    m = maximum(logs)
                    ws = exp.(logs .- m)
                    ws ./= sum(ws)
                    Q .= 0
                    for (πk, wk) in zip(ws, wells)
                        Q .+= πk * Q_block(wk.μ, wk.σ, T, z; nsamp=4000)
                    end
                    G = amp_G(Q, y; chi, n1, a0, kappa=κ, P)
                    V = potential(G, T, z, d, s0)
                    wells = find_wells(V)
                    isempty(wells) && break
                    # track near-zero well if present else deepest
                    near0 = argmin(w -> abs(w.μ), wells)
                    μ, σ = near0.μ, near0.σ
                end
            end
            if isempty(wells)
                @printf("%-6.2g %-8.3g FAIL\n", z, ρ)
                continue
            end
            vpp0 = try
                ForwardDiff.derivative(x -> ForwardDiff.derivative(V, x), 0.0)
            catch
                NaN
            end
            # classify: inner multimodality = ≥2 wells with |μ|<1.5
            inner = [w for w in wells if abs(w.μ) < 1.5]
            note = if z == 0.0
                "c2=0_invisible"
            elseif length(inner) >= 2
                "INNER_SPLIT"
            elseif any(abs(w.μ) > 1.5 for w in wells) && any(abs(w.μ) < 0.2 for w in wells)
                "OUTER_PLUS_CENTER"
            elseif any(abs(w.μ) > 1.5 for w in wells)
                "OUTER_ONLY"
            elseif length(wells) == 1 && abs(wells[1].μ) < 0.2
                "center_only"
            elseif length(wells) == 1
                "single_shifted"
            else
                "multi"
            end
            @printf("%-6.2g %-8.3g %-8.3g %-8.3g %-8.3g %-6d %s  wells=%s\n",
                z, ρ, Q[2, 2], G[2, 2], vpp0, length(wells), note,
                string(round.([w.μ for w in wells]; digits=3)))

            # dump V(w) for plotting
            for ww in range(-4.0, 4.0; length=401)
                vv = V(ww)
                push!(action_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%d,%s",
                    z, ρ, ww, vv, length(wells), note))
            end
            push!(csv_rows, @sprintf("%.6g,%.6g,%.8g,%.8g,%.8g,%d,%s",
                z, ρ, Q[2, 2], G[2, 2], vpp0, length(wells), note))
        end
    end

    open(joinpath(outdir, "he2_target_summary.csv"), "w") do io
        println(io, "z,rho,Q22,G22,Vpp0,nwells,note")
        for r in csv_rows
            println(io, r)
        end
    end
    open(joinpath(outdir, "he2_target_actions.csv"), "w") do io
        println(io, "z,rho,w,V,nwells,note")
        for r in action_rows
            println(io, r)
        end
    end
    println("wrote he2_target_*.csv")
end

# Sanity: z=0 ⇒ c2≈0, c1 matches theory
T = 1.0
for w in (0.5, 1.0, 2.0)
    c1, c2 = hermite_c12(w, T, 0.0)
    D = T + 2w^2
    c1th = sqrt(4 / π) * w / sqrt(D)
    @printf("sanity z=0 w=%g: c1=%.5f (th %.5f) c2=%.3e\n", w, c1, c1th, c2)
end
for z in (0.2, 0.5)
    c1, c2 = hermite_c12(1.0, T, z)
    @printf("sanity z=%g w=1: c1=%.4f c2=%.4f\n", z, c1, c2)
end

run()
println("DONE")
