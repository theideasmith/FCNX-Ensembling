using Plots
include("./FCS.jl")
using .FCS

d = 40.0
κ = 1.0
ϵ = 0.03
n = 400
χ = 400
b = 4 / (3π)
δ = 1.0



Ps = 10 .^ range(log10(d), log10(d^3), length=80)

ratio = Float64[]
for P in Ps
    params = FCS.ProblemParams(d, κ, ϵ, P, n, χ, b)
    i0 = [b / sqrt(d), 1 / d^(3 / 2), b / sqrt(d), 1 / d^(3 / 2)]
    sol = FCS.nlsolve_solver(i0, chi=χ, d=d, kappa=κ, delta=δ,
        epsilon=ϵ, n=n, b=b, P=P,
        lr=1e-6, max_iter=6_000_000, verbose=false,
        anneal=true, anneal_steps=30_000)
    lK1T, _ = FCS.compute_lK(sol, P, n, χ, d, δ, κ, ϵ, b)
    sol_perp = FCS.nlsolve_solver(i0, chi=χ, d=d, kappa=κ, delta=0.0,
        epsilon=ϵ, n=n, b=b, P=P,
        lr=1e-6, max_iter=6_000_000, verbose=false,
        anneal=true)
    lK1P, _ = FCS.compute_lK(sol_perp, P, n, χ, d, 0.0, κ, ϵ, b)
    push!(ratio, lK1T / lK1P)
end

plot(Ps, ratio, xscale=:log10, xlabel="P", ylabel="lK1T / lK1P",
    marker=:circle, ms=3, label=nothing, lw=2)