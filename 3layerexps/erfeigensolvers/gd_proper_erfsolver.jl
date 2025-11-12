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
# using ColorScheme
using LaTeXStrings

include("./FCS.jl") # Load the module definition
using .FCS

using Printf
using Printf


function sigfig(x::Float64; n::Int=4)
    if isnan(x) || isinf(x)
        return string(x)
    end

    # Handle zero separately
    if x == 0
        return "0" * (n > 1 ? "." * "0"^(n - 1) : "")
    end

    # Determine magnitude and required decimal places
    order = floor(log10(abs(x)))
    if order >= n
        # Round to nearest integer if all digits are in integer part
        return string(round(Int, x))
    else
        # Number of digits after decimal = n - (digits in integer part)
        int_digits = order >= 0 ? Int(order) + 1 : 1
        dec_digits = n - int_digits
        dec_digits = max(dec_digits, 0)  # safety

        # Use %.*f with rounding
        s = @sprintf("%.*f", dec_digits, x)

        # Strip trailing zeros and decimal point if unnecessary
        s = rstrip(s, '0')
        s = rstrip(s, '.')
        return s
    end
end


sigfig(::Nothing; n::Int=4) = "nothing"

# === Create sf as a closure that captures precision at definition time ===
function make_sigfig_formatter(precision::Int)
    return x -> sigfig(x; n=precision)
end
precision = 8

sf = x -> x #sigfig(x; n=precision)

d = 25
κ = 1.0
ϵ = 0.00
P = 130
n=250
χ = 25
δ=1.0
b = 4/(3*π)
sf(1 / 40^3)


Tf = 6_000_00
lr = 1e-6
i0 = [1/d, 1/d^3, 1/d, 1/d^3]
exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=1.0, delta=δ,
    epsilon=ϵ, n=n, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, verbose=true, anneal=true
)
l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n, χ, d, δ, κ, ϵ, 4 / (3 * π))

lJ1, lJ3, lH1, lH3 = exp_sol
println("-------")
println("Semi MF Scaling Target Eigenvalues χ=$χ")
if exp_sol === nothing
    println("exp_sol is nothing")
else
    lJ1, lJ3, lH1, lH3 = exp_sol
    lV1 = (lH1 / lJ1^2 - 1 / lJ1)
    lWT = 1 / (25)
    println("lH1: $(sf(lH1)), lH3: $(sf(lH3)) ")
    println("lJ1: $(sf(lJ1)), lJ3: $(sf(lJ3)) ")
end

println("Learnabilities")
println("mu1 = $(sf(l1))")
println("mu3 = $(sf(l3))")

delta = 1.0
lH1
1 / lJ1
(lH1 / lJ1^2 )
lV1 = (lH1 / lJ1^2 - 1 / lJ1)
lV3 = ((lH3 / lJ3^2 - 1 / lJ3))
delta * b * (lV1) / n
lWT = 1 / (d + delta * b * (lV1) / n)
@printf("lWT: %.*g\n", 7, lWT)

lWP = 1/d
TrSigma = lWT * d
1 + 2 * TrSigma
b = 1/ (π * (1 + 2 * TrSigma)^3)
(1 / π) * (1/3)^3
(16 / (π * 27) * (15 * lWP^3))



16 * 15 * b * (1 / d^3)
1/d^3

b = 4/(3*π)
FCS.residuals(exp_sol, P, χ, d, κ, δ, ϵ, n, b)
exp_sol

δ = 0.0


exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=1.0, delta=δ,
    epsilon=ϵ, n=n, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, anneal=true
)
l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n, χ, d, δ, κ, ϵ, 4 / (3 * π))

println("----------")
println("Perpendicular Eigenvalues")
if exp_sol === nothing
    println("exp_sol is nothing")
else
    lJ1, lJ3, lH1, lH3 = exp_sol
    lV1 = (lH1 / lJ1^2 - 1 / lJ1)
    lWT = 1 / (25)
    println("lH1: $(sf(lH1)), lH3: $(sf(lH3)) ")
    println("lJ1: $(sf(lJ1)), lJ3: $(sf(lJ3)) ")
end

println("Learnabilities")
println("mu1 = $(sf(l1))")
println("mu3 = $(sf(l3))")

delta = 0.0
lH1
1 / lJ1
(lH1 / lJ1^2)
lV1 = (lH1 / lJ1^2 - 1 / lJ1)
lV3 = ((lH3 / lJ3^2 - 1 / lJ3))
delta * b * (lV1) / n
lWT = 1 / (d + delta * b * (lV1) / n)
@printf("lWT: %.*g\n", 7, lWT)