
using Optim
    ENV["GKSwstype"] = "100"
using Plots
using LaTeXStrings
using Statistics
using JSON
# Constants
const N = 1600
const P = 3000.0
const kappa = 0.14
const epsilon = 0.074
const d_range = 10:10:300
const reg = kappa / P

# --- Linear Regression Helper ---
function calculate_exponent(x, y)
    mask = (x .> 0) .& (y .> 0)
    lx = log.(x[mask])
    ly = log.(y[mask])
    return (length(lx) * sum(lx .* ly) - sum(lx) * sum(ly)) /
        (length(lx) * sum(lx .^ 2) - sum(lx)^2)
end

# --- GFL Action ---
function s_gfl(beta, d, chi)
    tr_sigma = (d - 1.0) / d + beta / d
    c1 = 4.0 / (pi * (1.0 + 2.0 * tr_sigma))
    c3 = 40.0 / (pi * (1.0 + 2.0 * tr_sigma)^3)   
    l1 = c1 * (beta / d)
    l3 = c3 * (beta / d)^3
    return N * (beta) / 2 + chi / (l1 + reg) + (chi * epsilon^2) / (l3 + reg) #+ 0.5 * log(beta)
end

# --- Grokking Action (Sparse Specialization) ---
function s_grok(M, N, d, chi, reg, epsilon)
    α = M / N # Fraction of dedicated neurons

    tr_sigma_active = 1.0

    c1 = 4.0 / (pi * (1.0 + 2.0 * tr_sigma_active))
    # Note: Added the factor of 3 to the denominator here as well
    c3 = 40.0 / (3.0 * pi * (1.0 + 2.0 * tr_sigma_active)^3)

    l1 = c1 * α
    l3 = c3 * α  # Notice this is NOT cubed. 1^3 * α = α.

    # Action = Weight Penalty (M active neurons in d dimensions) + Generalization Error
    return M * d /2 + chi / (l1 + reg) + (chi * epsilon^2) / (l3 + reg)
end
# --- Solver ---
function solve_regime(chi_val)
    gfl_S, grok_S, gfl_sig2, grok_sig2 = Float64[], Float64[], Float64[], Float64[]
    for d in d_range
        res_gfl = optimize(b -> s_gfl(b, d, chi_val), 1e-9, 100, Brent())
        b_opt = Optim.minimizer(res_gfl)
        push!(gfl_S, Optim.minimum(res_gfl))
        push!(gfl_sig2, b_opt / d)

        res_grok = optimize(m -> s_grok(m, N, d, chi_val, reg, epsilon), 1e-9, N, Brent())
        m_opt = Optim.minimizer(res_grok)
        push!(grok_S, Optim.minimum(res_grok))
        push!(grok_sig2, m_opt / N)
    end
    alpha_gfl = calculate_exponent(collect(d_range), gfl_sig2)
    alpha_grok = calculate_exponent(collect(d_range), grok_sig2)
    return gfl_S, grok_S, gfl_sig2, grok_sig2, alpha_gfl, alpha_grok
end

# --- Chi values and viridis colormap ---
chi_values = [1.0, 10.0, N / 10.0, Float64(N)]
chi_labels = [L"\chi=1", L"\chi=10", L"\chi=N/10", L"\chi=N"]
viridis_colors = cgrad(:viridis)[[0.1, 0.38, 0.65, 0.92]]

println("Running for all chi values...")
results = [solve_regime(c) for c in chi_values]
d_vec = collect(d_range)

lw_data = 3.5
l_style = (grid=true, framestyle=:box, titlefont=12, tickfont=10, legendfont=10)

dummy_label(p, label, ls) = plot!(p, [NaN], [NaN],
    color=:gray40, ls=ls, lw=lw_data, label=label)

# ── Panel 1: Action S ─────────────────────────────────────────────────────────
p1 = plot(title=L"Action $S$", ylabel=L"S", yscale=:log10; l_style...)

for (i, r) in enumerate(results)
    c = viridis_colors[i]
    plot!(p1, d_vec, r[1], color=c, ls=:dash, lw=lw_data, label=chi_labels[i])
    plot!(p1, d_vec, r[2], color=c, ls=:solid, lw=lw_data, label="")
end
dummy_label(p1, L"\textbf{---}\ \mathrm{GFL}", :dash)
dummy_label(p1, L"\textbf{—}\ \mathrm{Grokking}", :solid)

# ── Panel 2: λ_Σ* Scaling ────────────────────────────────────────────────────
p2 = plot(title=L"$\lambda_\Sigma^*$ Scaling with $d$",
    ylabel=L"\lambda_\Sigma^*", xlabel=L"d",
    xscale=:log10, yscale=:log10, legend=:bottomleft; l_style...)

# We'll place ratio labels at a fixed fractional position along d (75% of the way)
ratio_d_idx = round(Int, 0.25 * length(d_vec))
ratio_d_val = d_vec[ratio_d_idx]
print(d_vec[ratio_d_idx], "\n")

for (i, r) in enumerate(results)
    c = viridis_colors[i]
    gfl_sig2 = r[3]
    grok_sig2 = r[4]

    plot!(p2, d_vec, gfl_sig2, color=c, ls=:dash, lw=lw_data, label=chi_labels[i])
    plot!(p2, d_vec, grok_sig2, color=c, ls=:solid, lw=lw_data, label="")

    # Ratio GFL / Grokking at the annotation point
    ratio = gfl_sig2[ratio_d_idx] / grok_sig2[ratio_d_idx]
    # Place annotation midway (geometric mean) between the two curves at that d
    y_ann = sqrt(gfl_sig2[ratio_d_idx] * grok_sig2[ratio_d_idx]) *0.8
    annotate!(p2, ratio_d_val, y_ann,
        text(latexstring("\\times$(round(ratio, digits=5))"),
            c, :left, 9))
    
end

dummy_label(p2, L"\textbf{---}\ \mathrm{GFL}", :dash)
dummy_label(p2, L"\textbf{—}\ \mathrm{Grokking}", :solid)

fig = plot(p1, p2, layout=(1, 2), size=(1200, 540), margin=10Plots.mm)
# Render the plot
# display(current())
# save to file
savefig(fig, joinpath(@__DIR__, "gfl_vs_grokking.pdf"))
# Output json the action and lambda_sigma for d=150
d_target = 150
target_idx = findfirst(==(d_target), d_vec)
output = Dict()
for (i, r) in enumerate(results)
    output[chi_labels[i]] = Dict(
        "S_gfl" => r[1][target_idx],
        "S_grok" => r[2][target_idx],
        "lambda_sigma_gfl" => r[3][target_idx],
        "lambda_sigma_grok" => r[4][target_idx],
        "alpha_gfl" => r[5],
        "alpha_grok" => r[6],
    )
end
open(joinpath(@__DIR__, "gfl_vs_grokking.json"), "w") do io
    JSON.print(io, output, 4)
end
println("Results saved to gfl_vs_grokking.json")
