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

d = 40
κ = 1.0
ϵ = 0.03
P = 250
n1 = 400
n2 = 400
χ = 50
δ=1.0
b = 4/(3*π)
sf(1 / 40^3)

params = FCS.ProblemParams(d, κ, ϵ, P, n1, n2, χ, b)


Tf = 6_000_000
lr = 1e-6
i0 = [4 / (3 * pi) * 1 / d^0.5, 1 / d^(3 / 2), 4 / (3 * pi) * 1 / d^0.5, 1 / d^(3 / 2), 1/d]
# i0 = fill(0.1, 4)
exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=1.0, delta=δ,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, verbose=false, anneal=true, anneal_steps=30_000
)
l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))


println("#####################################################################")
println("##    &*&       THE GREAT FCN3-ERF EOS SOLVER            &*&.        ")
println("##                       Initializes                                 ")
println("#####################################################################")

lJ1, lJ3, lH1, lH3 = exp_sol
println("-------")
println("Semi MF Scaling Target Eigenvalues χ=$χ")
if exp_sol === nothing
    println("exp_sol is nothing")
else
    lJ1, lJ3, lH1, lH3 = exp_sol
    lV1 = (lH1 / lJ1^2 - 1 / lJ1)
    println("lH1: $(sf(lH1)), lH3: $(sf(lH3)) ")
    println("lJ1: $(sf(lJ1)), lJ3: $(sf(lJ3)) ")
end

delta = 1.0
lH1
1 / lJ1
(lH1 / lJ1^2)
lV1 = (lH1 / lJ1^2 - 1 / lJ1)
lV3 = ((lH3 / lJ3^2 - 1 / lJ3))
delta * b * n2 * (lV1) / n1
lWT = 1 / (d + delta * b * n2 * (lV1) / n1)
@printf("lWT: %.*g\n", 7, lWT)

println("lJ1/lWT: $(lJ1/(b*lWT))")
println("lJ3/lWT^3: $(lJ3/(15*16/(27*π) * lWT^3))")


println("Learnabilities")
println("mu1 = $(sf(l1))")
println("mu3 = $(sf(l3))")

lK1T, lK3T = FCS.compute_lK(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))

println("Readout Eigenvalues: ")
println("lK1: $lK1T")
println("lK3: $lK3T")


δ = 0.0

i0 = [4 / (3 * pi) * 1 / d, 1 / d^(3), 4 / (3 * pi) * 1 / d, 1 / d^(3), 1/d]

exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=1.0, delta=δ,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, anneal=true
)
l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))

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
delta * b * n2 * (lV1) / n1
lWT = 1 / (d + delta * b * n2 * (lV1) / n1)
@printf("lWT: %.*g\n", 7, lWT)
println("lJ1/lWT: $(lJ1/(b*lWT))")
println("lJ3/lWT^3: $(lJ3/(15*16/(27*π) * lWT^3))")

lK1P, lK3P = FCS.compute_lK(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))
println("----------")

println("Readout Eigenvalues: ")
println("lK1: $lK1P")
println("lK3: $lK3P")
println("----------")

println("Ratio of readout: ")
println("lK1T / lK1P: $(lK1T/lK1P)")
println("lK3T / lK3P: $(lK3T/lK3P)")

