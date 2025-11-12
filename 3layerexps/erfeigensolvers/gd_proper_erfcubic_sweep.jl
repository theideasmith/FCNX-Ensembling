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
Nds = 20
d_vals = exp.(range(log(25), log(200), length=Nds))

kappa = 1.0
delta = 1.0
b = 4 / (3 * pi)
Nas = 30
alphas = range(0.1,4.0, length=Nas)
initial_guess = [0.01, 0.01, 0.01, 0.01]

lbs = zeros(Float64, 2, length(d_vals), length(alphas))


for (i, d) in enumerate(d_vals)
    n = 5*d  # your scaling rule
    _, learn1, learn3 = FCS.sweep_learnabilities(
        initial_guess,
        alphas=alphas, chi=n^0.2, d=d, kappa=kappa, delta=1.0,
        epsilon=0.03, n=n, b=b
    )
    lbs[1, i, :] = learn1
    lbs[2, i, :] = learn3
end

lbs[2,1,:]
lbs
using Serialization

struct ExperimentConfig
    alphas::Vector{Float64}
    d_vals::Vector{Float64}
    kappa::Float64
    chis::Vector{Float64}
    chi_exponent::Float64
    epsilon::Float64
    n_factor::Float64
    initial_guess::Vector{Float64}
    Nas::Int
    Nds::Int
    lr::Float64
    maxiter::Int
    tol::Float64
end

# Parameters to store
chi_exponent = 0.2
n_factor = 5.0
epsilon = 0.03
lr = 1e-4
maxiter = 5000
tol = 1e-8

# Build vector of chis corresponding to each d (chi = (n_factor * d)^chi_exponent)
chis = [(n_factor * d)^chi_exponent for d in d_vals]

config = ExperimentConfig(
    collect(alphas),
    d_vals,
    kappa,
    chis,
    chi_exponent,
    epsilon,
    n_factor,
    copy(initial_guess),
    Nas,
    Nds,
    lr,
    maxiter,
    tol
)

# Serialize to file
dmin = minimum(d_vals)
dmax = maximum(d_vals)
alpha_min = minimum(alphas)
alpha_max = maximum(alphas)
lH1 = initial_guess[2]

outfile = "experiment_d$(round(dmin))-$(round(dmax))_alpha$(round(alpha_min, digits=2))-$(round(alpha_max, digits=2))_lH1$(round(lH1, digits=4)).jls"

open(outfile, "w") do io
    serialize(io, config)
end

println("Wrote experiment config -> ", outfile)


println("Serialized experiment config -> experiment_config.jls")
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

colors = [get(ColorSchemes.viridis, i) for i in range(0, 1, length=length(d_vals))]

p = plot(
    xlabel="α (where P = d^α)",
    ylabel="Learnability",
    title="Learnabilities vs α for multiple d, chi=n^0.2, n=5d",
    legend=:outerright,
    lw=2
)


for (i, d) in enumerate(d_vals)
    
    learn1 = lbs[1, i, :]
    learn3 = lbs[2, i, :]
    col = colors[i]

    mask1 = 0 .< learn1 .< 1

    mask3 = 0 .< learn3 .< 1

    plot!(p, alphas[mask1], learn1[mask1],
        label="lK1, d=$(round(d))",
        lw=2, color=col, linestyle=:solid)

    plot!(p, alphas[mask3], learn3[mask3],
        label="lK3, d=$(round(d))",
        lw=2, color=col, linestyle=:dash)
end

display(p)
