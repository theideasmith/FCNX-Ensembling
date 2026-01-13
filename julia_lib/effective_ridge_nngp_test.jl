using LinearAlgebra, Statistics, NLsolve, JSON

const d, P = 150, 6000
const κ_bare = 1.0

function arcsin_kernel(X)
    XXT = (X * X') / size(X, 2)
    d_vec = sqrt.(1 .+ 2 .* diag(XXT))
    arg = (2 .* XXT) ./ (d_vec * d_vec')
    return (2 / π) .* asin.(arg)
end

X = randn(P, d)
K = arcsin_kernel(X)
λ = real.(eigvals(K)) 
filter!(x -> x > 1e-10, λ) 
# Normalize eigenvalues
λ .= λ ./ length(λ)
# PRint first few largest eigenvalues for sanity check

function solve_kappa(λ, κ_b, P)
    function f!(F, x)
        κ = x[1]
        # The correction term: (1/P) * Σ (...)
        correction = sum((κ / P  .* λ) ./ (κ / P .+ λ))
        F[1] = κ - (κ_b + correction)
    end
    sol = nlsolve(f!, [κ_b + mean(λ)])
    return sol.zero[1]
end

κ_eff = solve_kappa(λ, κ_bare, P)
correction_val = κ_eff - κ_bare
pct_increase = (correction_val / κ_bare) * 100

println("--- Results ---")
println("κ_bare:      ", κ_bare)
println("κ_eff:       ", κ_eff)
println("Correction:  ", correction_val)
println("Increase:    ", round(pct_increase, digits=4), "%")
println("Avg λ:       ", mean(λ))

# Save analysis to disk
open("kappa_analysis.json", "w") do f
    JSON.print(f, Dict(
            "ratio" => correction_val / κ_eff,
            "λ_max" => maximum(λ),
            "λ_min" => minimum(λ)
        ), 4)
end