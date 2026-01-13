using ForwardDiff
using LinearAlgebra
using Plots
using Colors
using Serialization
using Statistics
using LaTeXStrings
using Measures
using Printf

# # Try to enable a LaTeX-capable backend (PGFPlotsX) for high-quality LaTeX rendering.
# # If PGFPlotsX is not installed or LaTeX isn't available on the system, we'll fall back
# # to the default Plots backend but warn the user that LaTeX labels may not fully render.
# try
#     @eval begin
#         using PGFPlotsX
#         pgfplotsx()
#         println("Using PGFPlotsX backend for LaTeX rendering (requires LaTeX installed)")
#     end
# catch _
#     println("PGFPlotsX backend not available — LaTeX rendering may be limited. To enable, run:")
#     println("  julia -e 'using Pkg; Pkg.add(\"PGFPlotsX\")'")
#     println("and ensure a LaTeX distribution is installed (pdflatex).")
# end

include("./FCS.jl")
using .FCS
function polyfit(d, y; skip_first=5)
    mask = .!isnan.(y) .& isfinite.(y) .& (y .> 0)
    if count(mask) < 3
        return nothing
    end
    
    # Skip first few points for better fits
    mask[1:min(skip_first, length(mask))] .= false
    
    x = log10.(d[mask])
    ylog = log10.(y[mask])
    
    if length(x) < 2
        return nothing
    end
    
    # Fit without intercept: minimize sum((ylog - s*x)^2)
    # This gives s = sum(x * ylog) / sum(x^2)
    s = sum(x .* ylog) / sum(x .^ 2)
    
    # predicted y on full d grid: y = d^s (no multiplicative constant in log space)
    ypred = 10 .^ (s .* log10.(d))
    return (s, ypred)
end

#########################################################
# d-sweep eigensolver for fixed P
#
# This script sweeps over input dimension `d` for a fixed `P` and
# computes the mean-field / semi-empirical solutions for both the
# target ("T", delta=1.0) and perpendicular ("P", delta=0.0)
# eigenvalue equations using the solvers provided by `FCS.jl`.
#
# For each d the script records:
# - learnabilities mu1, mu3 (from FCS.compute_lK_ratio)
# - lH1, lH3 (target and perp)
# - lJ1, lJ3 (target and perp)
# - lK1, lK3 (readout eigenvalues from FCS.compute_lK)
#
# The script produces one figure per quantity-family and writes PNGs
# to `plots/d_sweep/`.
#########################################################

## User-editable sweep parameters
P_factor = 3                 # P = P_factor * d
P_power_factor = 2

κ = 1.0                      # kappa (noise)
ϵ = 0.03                     # epsilon in target
n_factor = 4.0                # n = n_factor * d
δ_target = 1.0               # delta for the target equations
δ_perp = 0.0                 # delta for the perpendicular equations
b = 4 / (3 * π)
lr = 1e-5
Tf = 2_000_000               # max iterations for solver

# Sweep range for d (input dimension)
d_values = 10 .^(collect(log10(10):0.05:log10(1000)))  # change as desired

# Output folder for plots
plot_dir = joinpath(@__DIR__, "..", "plots", "d_sweep")
mkpath(plot_dir)
results_file = joinpath(plot_dir, "d_sweep_results.jls")

sf(x) = x

# Storage arrays (we keep results for two P-variants per d)
ND = length(d_values)

# Variant 1: P = P_factor * d ("scale")
mu1_T_scale = fill(NaN, ND)
mu3_T_scale = fill(NaN, ND)
lH1_T_scale = fill(NaN, ND)
lH3_T_scale = fill(NaN, ND)
lJ1_T_scale = fill(NaN, ND)
lJ3_T_scale = fill(NaN, ND)
lK1_T_scale = fill(NaN, ND)
lK3_T_scale = fill(NaN, ND)

mu1_P_scale = fill(NaN, ND)
mu3_P_scale = fill(NaN, ND)
lH1_P_scale = fill(NaN, ND)
lH3_P_scale = fill(NaN, ND)
lJ1_P_scale = fill(NaN, ND)
lJ3_P_scale = fill(NaN, ND)
lK1_P_scale = fill(NaN, ND)
lK3_P_scale = fill(NaN, ND)

# Variant 2: P = 2 * d^1.5 ("power")
mu1_T_power = fill(NaN, ND)
mu3_T_power = fill(NaN, ND)
lH1_T_power = fill(NaN, ND)
lH3_T_power = fill(NaN, ND)
lJ1_T_power = fill(NaN, ND)
lJ3_T_power = fill(NaN, ND)
lK1_T_power = fill(NaN, ND)
lK3_T_power = fill(NaN, ND)

mu1_P_power = fill(NaN, ND)
mu3_P_power = fill(NaN, ND)
lH1_P_power = fill(NaN, ND)
lH3_P_power = fill(NaN, ND)
lJ1_P_power = fill(NaN, ND)
lJ3_P_power = fill(NaN, ND)
lK1_P_power = fill(NaN, ND)
lK3_P_power = fill(NaN, ND)

if false #isfile(results_file) 
    println("Loading existing results from: $results_file")
    results = deserialize(open(results_file))

    # assign arrays back into workspace (interpolate values so @eval doesn't look for local `v`)
    for (k,v) in results
        @eval $(Symbol(k)) = $(v)
    end
else
    for (i, d) in enumerate(d_values)
        P_scale = 3 * d
        P_power = P_power_factor * d^1.5

        P_list = [P_scale, P_power]
        n = n_factor * d
        n1 = n
        n2 = n
        
        χ = n2
        println("Solving for d=$d (n1=$n1, n2=$n2) — will try P variants: $(P_list)")
        # initial guess heuristics (scale with d)
        i0_target = [1 / sqrt(d), 1 / d^(3/2), 1 / sqrt(d), 1 / d^(3/2)]
        i0_perp = [4 / (3 * pi) * 1 / d, 1 / d^3, 4 / (3 * pi) * 1 / d, 1 / d^3]

        for (j, P) in enumerate(P_list)
            variant = j == 1 ? :scale : :power
            println("  Variant=$(variant) P=$(P)")

            # Target (delta = δ_target)
            try
                exp_sol_T = FCS.nlsolve_solver(i0_target;
                    chi=χ, d=d, kappa=κ, delta=δ_target,
                    epsilon=ϵ, n1=n, n2=n, b=b,
                    P=P, lr=lr, max_iter=Tf, verbose=false, anneal=true, anneal_steps=10_000)

                if exp_sol_T === nothing
                    println("    Target solve returned nothing for d=$d, variant=$(variant)")
                else
                    lJ1, lJ3, lH1, lH3 = exp_sol_T
                    # store based on variant
                    if variant == :scale
                        lJ1_T_scale[i] = lJ1
                        lJ3_T_scale[i] = lJ3
                        lH1_T_scale[i] = lH1
                        lH3_T_scale[i] = lH3
                    else
                        lJ1_T_power[i] = lJ1
                        lJ3_T_power[i] = lJ3
                        lH1_T_power[i] = lH1
                        lH3_T_power[i] = lH3
                    end

                    # compute learnabilities (ratio readout)
                    l1, l3 = FCS.compute_lK_ratio(exp_sol_T, P, n1, n2, χ, d, δ_target, κ, ϵ, b)
                    if variant == :scale
                        mu1_T_scale[i] = l1
                        mu3_T_scale[i] = l3
                    else
                        mu1_T_power[i] = l1
                        mu3_T_power[i] = l3
                    end

                    # readout eigenvalues
                    try
                        k1, k3 = FCS.compute_lK(exp_sol_T, P, n1, n2, χ, d, δ_target, κ, ϵ, b)
                        if variant == :scale
                            lK1_T_scale[i] = k1
                            lK3_T_scale[i] = k3
                        else
                            lK1_T_power[i] = k1
                            lK3_T_power[i] = k3
                        end
                    catch e
                        @warn "compute_lK target failed" exception=(e, catch_backtrace())
                    end
                end
            catch e
                @warn "Target solver failed for d=$d, variant=$(variant)" exception=(e, catch_backtrace())
            end

            # Perpendicular (delta = δ_perp)
            try
                exp_sol_P = FCS.nlsolve_solver(i0_perp;
                    chi=χ, d=d, kappa=κ, delta=δ_perp,
                    epsilon=ϵ, n1=n, n2=n, b=b,
                    P=P, lr=lr, max_iter=Tf, verbose=false, anneal=true, anneal_steps=10_000)

                if exp_sol_P === nothing
                    println("    Perp solve returned nothing for d=$d, variant=$(variant)")
                else
                    lJ1, lJ3, lH1, lH3 = exp_sol_P
                    if variant == :scale
                        lJ1_P_scale[i] = lJ1
                        lJ3_P_scale[i] = lJ3
                        lH1_P_scale[i] = lH1
                        lH3_P_scale[i] = lH3
                    else
                        lJ1_P_power[i] = lJ1
                        lJ3_P_power[i] = lJ3
                        lH1_P_power[i] = lH1
                        lH3_P_power[i] = lH3
                    end

                    l1p, l3p = FCS.compute_lK_ratio(exp_sol_P, P, n1, n2, χ, d, δ_perp, κ, ϵ, b)
                    if variant == :scale
                        mu1_P_scale[i] = l1p
                        mu3_P_scale[i] = l3p
                    else
                        mu1_P_power[i] = l1p
                        mu3_P_power[i] = l3p
                    end

                    try
                        k1p, k3p = FCS.compute_lK(exp_sol_P, P, n1, n2, χ, d, δ_perp, κ, ϵ, b)
                        if variant == :scale
                            lK1_P_scale[i] = k1p
                            lK3_P_scale[i] = k3p
                        else
                            lK1_P_power[i] = k1p
                            lK3_P_power[i] = k3p
                        end
                    catch e
                        @warn "compute_lK perp failed" exception=(e, catch_backtrace())
                    end
                end
            catch e
                @warn "Perp solver failed for d=$d, variant=$(variant)" exception=(e, catch_backtrace())
            end
        end
    end

    # serialize results for future runs
    results = Dict()
    for name in [
        :mu1_T_scale, :mu3_T_scale, :lH1_T_scale, :lH3_T_scale, :lJ1_T_scale, :lJ3_T_scale, :lK1_T_scale, :lK3_T_scale,
        :mu1_P_scale, :mu3_P_scale, :lH1_P_scale, :lH3_P_scale, :lJ1_P_scale, :lJ3_P_scale, :lK1_P_scale, :lK3_P_scale,
        :mu1_T_power, :mu3_T_power, :lH1_T_power, :lH3_T_power, :lJ1_T_power, :lJ3_T_power, :lK1_T_power, :lK3_T_power,
        :mu1_P_power, :mu3_P_power, :lH1_P_power, :lH3_P_power, :lJ1_P_power, :lJ3_P_power, :lK1_P_power, :lK3_P_power
    ]
        results[string(name)] = eval(name)
    end
    open(results_file, "w") do io
        serialize(io, results)
    end
    println("Serialized results to: $results_file")
end

# Keep legacy variable names (point to the "scale" variant) so plotting code below works
mu1_T = mu1_T_scale
mu3_T = mu3_T_scale
lH1_T = lH1_T_scale
lH3_T = lH3_T_scale
lJ1_T = lJ1_T_scale
lJ3_T = lJ3_T_scale
lK1_T = lK1_T_scale
lK3_T = lK3_T_scale

mu1_P = mu1_P_scale
mu3_P = mu3_P_scale
lH1_P = lH1_P_scale
lH3_P = lH3_P_scale
lJ1_P = lJ1_P_scale
lJ3_P = lJ3_P_scale
lK1_P = lK1_P_scale
lK3_P = lK3_P_scale

### Plotting helpers
function savefig_with_meta(fig, fname)
    outpath = joinpath(plot_dir, fname)
    println("Saving -> $outpath")
    savefig(fig, outpath)
end

default(titlefont=font(18), guidefont=font(16), tickfont=font(14), legendfontsize=20)

# Single combined figure for all eigenvalue traces
# Increase left margin so y-labels are not clipped when saving PNGs
# Legend at bottom left
plt_all = plot(yscale = :log10, xscale=:log10, xlabel=latexstring("d"), ylabel=latexstring("\\lambda"), legend=:outerright, size=(1400,900))

# helper to create LaTeX labels with interpolation
function latex_label(op, mode_sym, mode_idx, P_desc)
    mode = mode_sym == :star ? "\\star" : "\\perp"
    label= latexstring("\\lambda" * "_{$op}" * "^{($mode_idx)}$mode,\\; $P_desc)")
    println(label)
    return label
end

P_scale_desc =  string("P \\sim $(P_factor)\\,d")
P_power_desc = string("P \\sim $(P_power_factor)\\,d^{3/2}")

# Title / scaling text to include in plot titles (use LaTeXStrings for rendering)
n_scaling_text = string("n = $(n_factor)\\,d")
chi_scaling_text = string("\\chi = n")
# Super title as LaTeXString: Eigenvalue scaling (\lambda) with dimension d
full_plot_name = string("Eigenvalue scaling (") * latexstring("\\lambda") * string(") with dimension ") * latexstring("d")

# list of tuples: (ydata, operator, mode_sym, mode_idx, P_desc, style)
lines = []

push!(lines, (lH1_T_scale, "H", :star, 1, P_scale_desc, (:solid, :circle)))
push!(lines, (lH1_P_scale, "H", :perp, 1, P_scale_desc, (:solid, :cross)))
push!(lines, (lH1_T_power, "H", :star, 1, P_power_desc, (:dash, :star5)))
push!(lines, (lH1_P_power, "H", :perp, 1, P_power_desc, (:dash, :diamond)))

push!(lines, (lH3_T_scale, "H", :star, 3, P_scale_desc, (:solid, :utriangle)))
push!(lines, (lH3_P_scale, "H", :perp, 3, P_scale_desc, (:solid, :hexagon)))
push!(lines, (lH3_T_power, "H", :star, 3, P_power_desc, (:dash, :star5)))
push!(lines, (lH3_P_power, "H", :perp, 3, P_power_desc, (:dash, :hexagon)))

push!(lines, (lJ1_T_scale, "J", :star, 1, P_scale_desc, (:solid, :diamond)))
push!(lines, (lJ1_P_scale, "J", :perp, 1, P_scale_desc, (:solid, :x)))
push!(lines, (lJ1_T_power, "J", :star, 1, P_power_desc, (:dash, :cross)))
push!(lines, (lJ1_P_power, "J", :perp, 1, P_power_desc, (:dash, :dstar)))

push!(lines, (lJ3_T_scale, "J", :star, 3, P_scale_desc, (:solid, :diamond)))
push!(lines, (lJ3_P_scale, "J", :perp, 3, P_scale_desc, (:solid, :utriangle)))
push!(lines, (lJ3_T_power, "J", :star, 3, P_power_desc, (:dash, :star5)))
push!(lines, (lJ3_P_power, "J", :perp, 3, P_power_desc, (:dash, :hexagon)))

push!(lines, (lK1_T_scale, "K", :star, 1, P_scale_desc, (:solid, :circle)))
push!(lines, (lK1_P_scale, "K", :perp, 1, P_scale_desc, (:solid, :x)))
push!(lines, (lK1_T_power, "K", :star, 1, P_power_desc, (:dash, :star5)))
push!(lines, (lK1_P_power, "K", :perp, 1, P_power_desc, (:dash, :diamond)))

push!(lines, (lK3_T_scale, "K", :star, 3, P_scale_desc, (:solid, :diamond)))
push!(lines, (lK3_P_scale, "K", :perp, 3, P_scale_desc, (:solid, :utriangle)))
push!(lines, (lK3_T_power, "K", :star, 3, P_power_desc, (:dash, :cross)))
push!(lines, (lK3_P_power, "K", :perp, 3, P_power_desc, (:dash, :dstar)))

for (ydata, op, m, idx, pdesc, style) in lines
    (ls, mk) = style
    lbl = latex_label(op, m, idx, pdesc)
    plot!(plt_all, d_values, ydata; label=lbl, lw= ifelse(ls==:solid,2,1), linestyle=ls, marker=mk)
end

title!(plt_all, full_plot_name * "\n" * latexstring(n_scaling_text * string(", ") * chi_scaling_text) * string(" — scale & power P variants"))
savefig_with_meta(plt_all, "all_eigenvalues_Pvariants_nscale10.png")

###############################
# New: per-eigenvalue PNGs with two subplots (scale | power)
###############################
ops_map = Dict(
    "H1" => (lH1_T_scale, lH1_P_scale, lH1_T_power, lH1_P_power, string("\\lambda_{H}^{(1)}")),
    "H3" => (lH3_T_scale, lH3_P_scale, lH3_T_power, lH3_P_power, string("\\lambda_{H}^{(3)}")),
    "J1" => (lJ1_T_scale, lJ1_P_scale, lJ1_T_power, lJ1_P_power, string("\\lambda_{J}^{(1)}")),
    "J3" => (lJ3_T_scale, lJ3_P_scale, lJ3_T_power, lJ3_P_power, string("\\lambda_{J}^{(3)}")),
    "K1" => (lK1_T_scale, lK1_P_scale, lK1_T_power, lK1_P_power, string("\\lambda_{K}^{(1)}")),
    "K3" => (lK3_T_scale, lK3_P_scale, lK3_T_power, lK3_P_power, string("\\lambda_{K}^{(3)}"))
)
lH1_T_power
P_scale_label = latexstring("P \\sim $(P_factor)\\,d")

P_power_label = latexstring("P \\sim $(P_power_factor)\\,d^{3/2}")

# Fit helper: fit y = A * d^s in log10 space (returns (s, intercept)) and predicted values for grid
function fit_powerlaw(d, y)
    mask = .!isnan.(y) .& isfinite.(y) .& (y .> 0)
    if count(mask) < 3
        return nothing
    end
    x = log10.(d[mask])
    ylog = log10.(y[mask])
    xm = mean(x); ym = mean(ylog)
    s = sum((x .- xm) .* (ylog .- ym)) / sum((x .- xm) .^ 2)
    intercept = ym - s * xm
    # predicted y on full d grid
    ypred = 10 .^ (intercept .+ s .* log10.(d))
    return (s, intercept, ypred)
end

for (opname, tup) in ops_map
    (t_scale, p_scale, t_power, p_power, ylatex) = tup

    # two-panel subplot: left = P ~ d*scale, right = P ~ d^{3/2}
    # increase left margin to avoid clipping of y-label on the left subplot
    plt = plot(layout = (1,2), size=(1600,800), left_margin = 30mm)
    # overall super-title for the two-panel figure (LaTeX formatted)
    title!(plt, full_plot_name)

    # Left: scale (P ~ P_factor * d)
    lab_star = latexstring(string(ylatex * "^{\\star}")) * " target"
    lab_perp = latexstring(string(ylatex * "^{\\perp}")) * " perp" 
    plot!(plt[1], d_values, t_scale; label=lab_star, marker=:circle, lw=2, xscale=:log10, yscale=:log10)
    plot!(plt[1], d_values, p_scale; label=lab_perp, marker=:x, lw=2)
    xlabel!(plt[1], latexstring("d"))
    ylabel!(plt[1], latexstring(ylatex))
    title!(plt[1], latexstring(P_scale_label) * ", " * latexstring(n_scaling_text) * ", " * latexstring(chi_scaling_text))

    # fits (scale): target
    ft = fit_powerlaw(d_values[10:end], t_scale[10:end])
    if ft !== nothing
            (s, intercept, ypred) = ft
            plot!(plt[1], d_values[10:end], ypred; label=string("fit star: " * latexstring("d^{", round(s,digits=3), "}")), linestyle=:dot, color=:black, alpha=0.6)
            # annotate slope shifted left to avoid overlap (25% position)
            xann = d_values[Int(round(0.25*end))]
            yann = 10^(intercept + s * log10(xann))
            annotate!(plt[1], xann, yann, text("s=$(round(s,digits=3))", 18, :right))
    end
    # fits (scale): perp
    fp = fit_powerlaw(d_values[10:end], p_scale[10:end])
    if fp !== nothing
            (s2, intercept2, ypred2) = fp
            plot!(plt[1], d_values[10:end], ypred2; label=string("fit perp: " * latexstring("d^{", round(s2,digits=3), "}")), linestyle=:dot, color=:gray, alpha=0.6)
            xann = d_values[Int(round(0.15*end))]
            yann = 10^(intercept2 + s2 * log10(xann))
            annotate!(plt[1], xann, yann, text("s=$(round(s2,digits=3))", 18, :right))
    end

        # Place legend inside the left subplot at the bottom-left with smaller font
        plot!(plt[1], legend = :bottomleft, legendfontsize = 16)

    # Right: power (P ~ 2 * d^{3/2})
    lab_star_p = latexstring(string(ylatex) * "^{\\ast}") * " Target"
    lab_perp_p = latexstring(string(ylatex) * "^{\\perp}") * " Perp"

    plot!(plt[2], d_values, t_power; label=lab_star_p, marker=:star5, lw=2, xscale=:log10, yscale=:log10)
    plot!(plt[2], d_values, p_power; label=lab_perp_p, marker=:diamond, lw=2)
    xlabel!(plt[2], latexstring("d"))
    ylabel!(plt[2], latexstring(ylatex))
    title!(plt[2], latexstring(P_power_label) * ", " * latexstring(n_scaling_text) * ", " * latexstring(chi_scaling_text))

    # fits (power): target
    ftp = fit_powerlaw(d_values, t_power)
    if ftp !== nothing
            (s3, intercept3, ypred3) = ftp
            plot!(plt[2], d_values, ypred3; label=string("fit star: " * latexstring("d^{", round(s3,digits=3), "}")), linestyle=:dot, color=:black, alpha=0.6)
            xann = d_values[Int(round(0.25*end))]
            yann = 10^(intercept3 + s3 * log10(xann))
            annotate!(plt[2], xann, yann, text("s=$(round(s3,digits=3))", 18, :right))
    end
    # fits (power): perp
    fpp = fit_powerlaw(d_values, p_power)
    if fpp !== nothing
            (s4, intercept4, ypred4) = fpp
            plot!(plt[2], d_values, ypred4; label=string("fit perp: " * latexstring("d^{", round(s4,digits=3), "}")), linestyle=:dot, color=:gray, alpha=0.6)
            xann = d_values[Int(round(0.15*end))]
            yann = 10^(intercept4 + s4 * log10(xann))
            annotate!(plt[2], xann, yann, text("s=$(round(s4,digits=3))", 18, :right))
    end

        # Place legend inside the right subplot at the bottom-left with smaller font
        plot!(plt[2], legend = :bottomleft, legendfontsize = 16)

    outname = joinpath(plot_dir, string(opname, "_eigenvalues_Pscale_vs_Ppower.png"))
    println("Saving -> $outname")
    savefig(plt, outname)
end

println("Done. Per-eigenvalue PNGs saved under: $plot_dir")
