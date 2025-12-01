using ForwardDiff
using LinearAlgebra
using Plots  # for plotting

using ForwardDiff
using LinearAlgebra
using Plots
using Colors   # for distinguishable_colors

using ForwardDiff
using LinearAlgebra
using Plots
using Colors   # for distinguishable_colors
using ColorSchemes
using LaTeXStrings

include("./FCS.jl") # Load the module definition
using .FCS

plotting = true

# 5 log-scaled values of d from 25 to 25000
Nds = 5
d_vals = exp.(range(log(25), log(200), length=Nds))

kappa = 1.0
delta = 1.0
b = 4 / (3 * pi)
Nas = 20
alphas = range(0.0,2.5, length=Nas)

lbs = zeros(Float64, 2, length(d_vals), length(alphas))


for (i, d) in enumerate(d_vals)
    println("Processing d = ", d)
    i0 = []
    if delta == 0.0
        i0 = [1 / d, d^3, 1 / d, d^3]
    else
        i0 = [1 / d^0.5, 1 / d^1.5, 1 / d^0.5, 1 / d^1.5]
    end

    n = 10*d  # your scaling rule
    _, learn1, learn3 = FCS.sweep_learnabilities(
        i0,
        alphas=alphas, chi=n, d=d, kappa=kappa, delta=1.0,
        epsilon=0.04, n=n, b=b
    )
    lbs[1, i, :] = learn1
    lbs[2, i, :] = learn3
end

lbs[2,1,:]
lbs
# using Serialization

# struct ExperimentConfig
#     alphas::Vector{Float64}
#     d_vals::Vector{Float64}
#     kappa::Float64
#     chis::Vector{Float64}
#     chi_exponent::Float64
#     epsilon::Float64
#     n_factor::Float64
#     initial_guess::Vector{Float64}
#     Nas::Int
#     Nds::Int
#     lr::Float64
#     maxiter::Int
#     tol::Float64
# end

# # Parameters to store
# chi_exponent = 0.2
# n_factor = 5.0
# epsilon = 0.03
# lr = 1e-4
# maxiter = 5000
# tol = 1e-8

# # Build vector of chis corresponding to each d (chi = (n_factor * d)^chi_exponent)
# chis = [(n_factor * d)^chi_exponent for d in d_vals]

# config = ExperimentConfig(
#     collect(alphas),
#     d_vals,
#     kappa,
#     chis,
#     chi_exponent,
#     epsilon,
#     n_factor,
#     copy(initial_guess),
#     Nas,
#     Nds,
#     lr,
#     maxiter,
#     tol
# )

# # Serialize to file
# dmin = minimum(d_vals)
# dmax = maximum(d_vals)
# alpha_min = minimum(alphas)
# alpha_max = maximum(alphas)
# lH1 = initial_guess[2]

# outfile = "experiment_d$(round(dmin))-$(round(dmax))_alpha$(round(alpha_min, digits=2))-$(round(alpha_max, digits=2))_lH1$(round(lH1, digits=4)).jls"

# open(outfile, "w") do io
#     serialize(io, config)
# end

# println("Wrote experiment config -> ", outfile)


# println("Serialized experiment config -> experiment_config.jls")
# size(d_vals)
# l1Idxs = [findfirst((>)(0.5), lbs[1, i, :]) for i in 1:Nds]
# l3Idxs = [findfirst((>)(0.5), lbs[2, i, :]) for i in 1:Nds]

# # Mask alphas where l1Idxs and l3Idxs are non-nothing
# t1 = [alphas[l1Idxs[i]] for i in 1:Nds if !isnothing(l1Idxs[i])]
# t2 = [alphas[l3Idxs[i]] for i in 1:Nds if !isnothing(l3Idxs[i])]

# println("--------------")
# plt = plot(xscale=:log10, ylims=(0.0, 4.0))
# # Get the indices where l1Idxs and l3Idxs are non-nothing
# valid_idx1 = [i for i in 1:Nds if !isnothing(l1Idxs[i])]
# valid_idx2 = [i for i in 1:Nds if !isnothing(l3Idxs[i])]

# # Plot with the masked indices
# scatter!(plt, d_vals[valid_idx1], t1, label="t1", lw=2, linestyle=:solid)
# scatter!(plt, d_vals[valid_idx2], t2, label="t2", lw=2, linestyle=:dash)

# title!(plt, "Learnable α(d)")
# xlabel!(plt, "d")
# display(plt)
colors = []
try
    colors = [get(ColorSchemes.viridis, i) for i in range(0, 1, length=length(d_vals))]
catch
    colors = [get(ColorSchemes.viridis, 0)]
end

# Prepare slide-friendly fonts and LaTeX labels
title_f = 32
guide_f = 24
tick_f = 18
legend_f = 20
annot_f = 20

p = plot(
    xlabel = L"\alpha\; (P = d^\alpha)",
    ylabel = L"\text{Learnability}",
    title = L"\textbf{Learnabilities vs }\\ \alpha",
    legend = :outerright,
    lw = 3,
    titlefont = font(title_f),
    guidefont = font(guide_f),
    tickfont = font(tick_f),
    legendfont = font(legend_f),
)

for (i, d) in enumerate(d_vals)
    learn1 = lbs[1, i, :]
    learn3 = lbs[2, i, :]
    col = colors[i]

    mask1 = 0 .< learn1 .< 1
    mask3 = 0 .< learn3 .< 1

    # LaTeX legend labels including the rounded d value
    lbl1 = L"\lambda^{K1},\; d=$(round(d))"
    lbl3 = L"\lambda^{K3},\; d=$(round(d))"

    plot!(p, alphas[mask1], learn1[mask1], label=lbl1, lw=3, color=col, linestyle=:solid)
    plot!(p, alphas[mask3], learn3[mask3], label=lbl3, lw=3, color=col, linestyle=:dash)
end

# Add a boxed annotation with run parameters in LaTeX for slides
# Compose the LaTeX multiline string with parameter values
params_txt = L"\\begin{aligned} N_1 &= N_2 \\ n &= 10\\times d \\ kappa &= $(kappa) \\ delta &= $(delta) \\ epsilon &= 0.04 \\ b &= $(round(b, digits=4)) \\end{aligned}"

# Position the annotation near the upper-left of the plot area
x_text = minimum(alphas) + 0.02 * (maximum(alphas) - minimum(alphas))
y_text = 0.92 * maximum(lbs)
annot = text(params_txt, :left, annot_f)
annotate!(p, x_text, y_text, annot)

display(p)
plotting = true
# Save the plot to a file (high-res for slides)
if plotting
    savefig(p, "gd_proper_erfcubic_sweep_slide.png")
end