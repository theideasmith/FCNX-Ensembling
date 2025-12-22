# gd_proper_linear_eigsolver.jl
# Replicates the eigenvalue solve workflow for the linear (FCSLinear) case.
# Uses nlsolve_solver to find (lJ, lH) and computes the readout eigenvalue lK.

using Printf
using LinearAlgebra
include("./FCSLinear.jl")
using .FCSLinear

# -------------------------
# Helper: significant figures printer
# -------------------------
function sigfig(x::Float64; n::Int=6)
    if isnan(x) || isinf(x)
        return string(x)
    end
    if x == 0
        return "0" * (n > 1 ? "." * repeat("0", n - 1) : "")
    end
    order = floor(log10(abs(x)))
    if order >= n
        return string(round(Int, x))
    else
        int_digits = order >= 0 ? Int(order) + 1 : 1
        dec_digits = max(n - int_digits, 0)
        s = @sprintf("%.*f", dec_digits, x)
        s = rstrip(rstrip(s, '0'), '.')
        return s
    end
end

# -------------------------
# Parameters (provided)
# -------------------------
n1 = 50.0
n2 = 50.0
P = 50
d = 6
χ = 50
σ2 = 1.0
kappa = 1.0

# Map provided names to solver parameters
chi = χ
k = kappa
kappa = k  # keep name expected by solver
# In the linear solver, "delta" is the variance term.
delta = σ2

# Initial guess for [lJ, lH]
initial_guess = [1.0 / d^0.5, 1.0/d^0.5]

println("======================")
println("FCSLinear Eigen Solver")
println("======================")
println("Parameters:")
println("  n1 = $(Int(n1)), n2 = $(Int(n2))")
println("  P  = $(Int(P)), d  = $(Int(d))")
println("  chi = $(chi)")
println("  delta (σ²) = $(delta)")
println("  kappa (k) = $(kappa)")
println("  initial guess = $(initial_guess)")

sol = FCSLinear.nlsolve_solver(
    initial_guess;
    P = P,
    chi = chi,
    d = d,
    kappa = kappa,
    delta = delta,
    n1 = n1,
    n2 = n2,
    max_iter = 60_000,
    anneal_steps=3_000,
    lr = 1e-4,
    verbose = false,
    anneal=true
)

if sol === nothing
    println("Solver returned nothing (no physical solution found).")
    exit(1)
end

lJ, lH = sol
println("\nSolution:")
println("  lJ = $(sigfig(lJ))")
println("  lH = $(sigfig(lH))")

lK = FCSLinear.compute_lK(sol, P, n1, n2, chi, d, delta, kappa)
println("\nReadout eigenvalue lK:")
println("  lK = $(sigfig(lK[1]))")

# Sanity check: theoretical equilibrium variance for quadratic potential is T/γ.
# Here, noise scale uses kappa and delta only through FCS equations; no extra check needed.
discrepancy = kappa / (lH + kappa)
loss_term = P * discrepancy^2 * chi / kappa


println("\nReadout eigenvalue lK:")
println("  lK = $(sigfig(lK[1]))")
println("  Loss scale = $(sigfig(loss_term)))")

println("\nDone.")

println("\n======================")
# -------------------------
# Parameters (provided)
# -------------------------
n1 = 50.0
n2 = 50.0
P = 50
d = 6.0
χ = 1.0
σ2 = 1.0
kappa = 1.0


# Map provided names to solver parameters
chi = χ
k = kappa
kappa = k  # keep name expected by solver
# In the linear solver, "delta" is the variance term.
delta = σ2

# Initial guess for [lJ, lH]
initial_guess = [1.0 / d^0.5, 1.0 / d^0.5]


println("Parameters:")
println("  n1 = $(Int(n1)), n2 = $(Int(n2))")
println("  P  = $(Int(P)), d  = $(Int(d))")
println("  chi = $(chi)")
println("  delta (σ²) = $(delta)")
println("  kappa (k) = $(kappa)")
println("  initial guess = $(initial_guess)")

sol = FCSLinear.nlsolve_solver(
    initial_guess;
    P=P,
    chi=chi,
    d=d,
    kappa=kappa,
    delta=0.0,
    n1=n1,
    n2=n2,
    max_iter=60_000,
    anneal_steps=3_000,
    lr=1e-4,
    verbose=false,
    anneal=true
)

if sol === nothing
    println("Solver returned nothing (no physical solution found).")
    exit(1)
end

lJ, lH = sol
println("\nSolution:")
println("  lJ = $(sigfig(lJ))")
println("  lH = $(sigfig(lH))")

lK = FCSLinear.compute_lK(sol, P, n1, n2, chi, d, delta, kappa)
lJ, lH = sol

discrepancy = chi * kappa / (lH + kappa)
loss_term = P * discrepancy^2 / chi / kappa

println("\nReadout eigenvalue lK:")
println("  lK = $(sigfig(lK[1]))")
println("  Loss scale = $(sigfig(loss_term)))")
println("\n======================")

