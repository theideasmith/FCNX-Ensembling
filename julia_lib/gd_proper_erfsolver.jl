using ForwardDiff
using LinearAlgebra

include("./FCS.jl") # Load the module definition
using .FCS

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

d = 150
κ = 0.1
ϵ = 0.03 * 6^0.5
P = 3000
n1 = 1600
n2 = 1600
χ = 50
δ=1.0
b = 4/(3*π)
sf(1 / 40^3)

params = FCS.ProblemParams(d, κ, ϵ, P, n1, n2, χ, b)


Tf = 6_000_000
lr = 1e-8
i0 = [4 / (3 * pi) * 1 / d^0.5, 1 / d^(3 / 2), 4 / (3 * pi) * 1 / d^0.5, 1 / d^(3 / 2), 1/d]
# i0 = fill(0.1, 4)
exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=κ, delta=1.0,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, verbose=false, anneal=false, anneal_steps=100000, tol=1e-8
)


exp_sol_T = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=κ, delta=δ,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, verbose=false, anneal=true, anneal_steps=100000, tol=1e-12
)

exp_sol_P = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=κ, delta=0.0,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, anneal=true, tol=1e-12, anneal_steps = 100000
)
lJ1T, lJ3T, lH1T, lH3T, lWT = exp_sol_T
lJ1P, lJ3P, lH1P, lH3P, lWP = exp_sol_P
lJ1T / lJ1P

lWT / lWP

TrSigma = lWT + lWP * (d - 1)

lJ1T / (4 / (π * (1 + 2 * TrSigma))) - lWT
lJ1P / (4 / (π * (1 + 2 * TrSigma))) - lWP
b = 4 / π / (1 + 2 * TrSigma)

lWT
lWP

log10(lWT)
log10(lWP)

lWT^3
lWP^3

lWT^3 / lWP^3

lJ3T / lJ3P

log10(lWP)
log10(lWT)

lJ1T
lJ1P

lJ3T
lJ3P

lJ1P / b 
lWP
lH1T
lH3T
lJ1T  - b * lWT

lJ3T
gh = 1 / ((16 / (π * (1 + 2 * TrSigma)^3) * 15)) / 6
(lJ3T * gh)^(1.0/3)
lWT
(15 * 16 / (π * (1 + 2 * TrSigma)^3) ) * lWT^3 
gh = ((16 / (π * (1 + 2 * TrSigma)^3) * 15)) / 6

aghf = (lJ3T * gh)^(1.0 / 3)
(aghf^3 ) / gh
hg = 1 / gh 
hg * lWT^3
lJ3T

lJ3T

lJ1T / b 
lWT

lJ1P / b
lWP

lH1T
lH3T
lH1P
lH3P
lJ3T 
lJ3P
lWT
lWP
lWT^3 / lWP^3
(lWT / lWP)^3

FCS.residuals(exp_sol_P, P, χ, d, κ, 0.0, ϵ, n1, n2, b)
FCS.residuals(exp_sol_T, P, χ, d, κ, 1.0, ϵ, n1, n2, b)


1.14e-4 / 5.03e-6

FCS.residuals(exp_sol_T, P, χ, d, κ, 1.0, ϵ, n1, n2, b)



l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))


println("#####################################################################")
println("##    &*&       THE GREAT FCN3-ERF EOS SOLVER            &*&.        ")
println("##                       Initializes                                 ")
println("#####################################################################")

lJ1, lJ3, lH1, lH3, lWT = exp_sol
rj3 = (lJ3 - ((16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)))/6) 

println("-------")
println("Semi MF Scaling Target Eigenvalues χ=$χ")
if exp_sol === nothing
    println("exp_sol is nothing")
else
    lJ1, lJ3, lH1, lH3, lWT = exp_sol
    lV1 = -(lH1 / lJ1^2 - 1 / lJ1)
    println("lH1: $(sf(lH1)), lH3: $(sf(lH3)) ")
    println("lJ1: $(sf(lJ1)), lJ3: $(sf(lJ3)) ")
end

@printf("lWT: %.*g\n", 7, lWT)

println("lJ1/lWT: $(lJ1/(b*lWT))")
println("lJ3/lWT^3: $(lJ3/(15*16/(27*π) * lWT^3))")

lJ1, lJ3, lH1, lH3, lWT = exp_sol
delta = 1.0
lV1 = -(lH1 / lJ1^2 - 1 / lJ1)
lV3 = -(lH3 / lJ3^2 - 1 / lJ3)
lWT = 1 / (d + delta * b * n2 * (lV1) / n1)
lWP = 1 / d

TrSigma = lWT + lWP * (d - 1)
EChh = lH1 + lH3 +
       (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) +
       (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)
gammaYh2 = (4 / π) / (1 + 2 * EChh)
b = 4/π / (1+2 * TrSigma)
lK1 = gammaYh2 * lH1
lK3 = gammaYh2 * lH3

println("Learnabilities")
println("mu1 = $(sf(l1))")
println("mu3 = $(sf(l3))")

lK1T, lK3T = FCS.compute_lK(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))

println("Readout Eigenvalues: ")
println("lK1: $lK1T")
println("lK3: $lK3T")

lJ1
b * lWT

lJ3 - ((16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)))

δ = 0.0

i0 = [4 / (3 * pi) * 1 / d, 1 / d^(3), 4 / (3 * pi) * 1 / d, 1 / d^(3), 1/d]

exp_sol = FCS.nlsolve_solver(
    i0,
    chi=χ, d=d, kappa=2.0, delta=δ,
    epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
    P=P, lr=lr, max_iter=Tf, anneal=true
)
l1, l3 = FCS.compute_lK_ratio(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))

println("----------")
println("Perpendicular Eigenvalues")
if exp_sol === nothing
    println("exp_sol is nothing")
else
    lJ1, lJ3, lH1, lH3, lWP = exp_sol
    lV1 = -(lH1 / lJ1^2 - 1 / lJ1)
    lWT = 1 / (25)
    println("lH1: $(sf(lH1)), lH3: $(sf(lH3)) ")
    println("lJ1: $(sf(lJ1)), lJ3: $(sf(lJ3)) ")
end

println("Learnabilities")
println("mu1 = $(sf(l1))")
println("mu3 = $(sf(l3))")

delta = 0.0
@printf("lWP: %.*g\n", 7, lWP)
println("lJ1/lWP: $(lJ1/(b*lWP))")
println("lJ3/lWP^3: $(lJ3/(15*16/(27*π) * lWP^3))")

lK1P, lK3P = FCS.compute_lK(exp_sol, P, n1, n2, χ, d, δ, κ, ϵ, 4 / (3 * π))
println("----------")

println("Readout Eigenvalues: ")
println("lK1: $lK1P")
println("lK3: $lK3P")
println("----------")

println("Ratio of readout: ")
println("lK1T / lK1P: $(lK1T/lK1P)")
println("lK3T / lK3P: $(lK3T/lK3P)")

