
module FCS

export residuals, gradient_descent_solver, compute_lK_ratio, sweep_learnabilities, nlsolve_solver



using ForwardDiff
using LinearAlgebra
using Plots  # for plotting

using ForwardDiff
using LinearAlgebra
using Plots
using Colors   # for distinguishable_colors
using NLsolve
using ForwardDiff
using LinearAlgebra
using Plots
using Colors   # for distinguishable_colors
using ColorSchemes
using LaTeXStrings


is_physical(sol) = all(sol .> 0)

function get_eigenvalues(i0,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n=1.0, b=1.0,
    P=nothing)
    Tf = 6_000_000
    lr = 1e-4
    return gradient_descent_solver(
        i0,
        chi=chi, d=d, kappa=1.0, delta=delta,
        epsilon=ϵ, n=n, b=4 / (3 * π),
        P=P, lr=lr, max_iter=Tf, verbose=true
    )
end

function nlsolve_solver(initial_guess;
    anneal= false,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n=1.0, b=1.0,
    P=nothing,anneal_steps = 30000,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)
    x = copy(initial_guess)
    if P === nothing
        P = d^1.2 # default
    end


    function res_func!(F, x, c)
        F[:] = residuals(x, P, c, d, kappa, delta, epsilon, n, b)
    end

    result = nothing
    if anneal 
        chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
        prev_sol = x
        for (j, chit) in enumerate(chi_anneal_list)

            f1! = (F, x) -> res_func!(F, x, chit)
            try

                sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
                if (j == anneal_steps)

                    result = is_physical(sol.zero) ? sol.zero : nothing
                end
                prev_sol = sol.zero

            catch e
                print("THERE WAS AN ERROR")
                showerror(stdout, e, catch_backtrace())
            end
        end
    else 
        f2!(F, x) = res_func!(F,x,chi)
        result = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
    end 


    return result
end
# -------------------------
# Residual function
# -------------------------
# Given a guess x = [lH1, lJ1, lH3, lJ3], returns the residuals of the 4 equations
function residuals(x, P, chi, d, kappa, delta, epsilon, n, b)
    lJ1, lJ3, lH1, lH3 = x  # current variables

    a = 1.0 / 6
    lWP =  1.0 / d
    lV1 = ( lH1 / lJ1^2 - 1 / lJ1)
    lV3 = ( (lH3 / lJ3^2 - 1 / lJ3) )

    C = d + (1/2)*delta * b * lV1 / n

    K0 = C

    lWT = 1 / (K0)

    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 +
        (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) * (d - 1) +
        (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)

    gammaYh2 = (4 / π) / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3

    lT1 = -chi^2 / (kappa / P + lK1)^2 * delta
    lT3 = ( -chi^2 / (kappa / P + lK3)^2 * delta )

    # Residuals
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma)) * lWT)
    rh1 = lH1 - 1 / (1 / lJ1 + gammaYh2 * lT1 / (n * chi))
    rh3 = lH3 - ( 1 / (1 / lJ3 +  gammaYh2 * lT3 * epsilon^2 / (n * chi)) )
    rj3 = lJ3 - ( (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)) )

    return [rj1, rj3, rh1, rh3]
end

# -------------------------
# Gradient Descent Solver
# -------------------------
function gradient_descent_solver(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n=1.0, b=1.0,
    P=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    x = copy(initial_guess)
    if P === nothing
        P = d^1.2  # default
    end
    loss = nothing

    for iter in 1:max_iter
        # Compute residuals
        res = residuals(x, P, chi, d, kappa, delta, epsilon, n, b)

        # Loss = sum of squares
        loss = sum(res .^ 2) + sum(x .^ 2)

        # Gradient of loss wrt x
        grad = ForwardDiff.gradient(x -> sum(residuals(x, P, chi, d, kappa, delta, epsilon, n, b) .^ 2), x)

        # Update step
        x -= lr * grad

        # Convergence check
        if loss < tol
            if verbose
                println("Converged in $iter iterations, loss = $loss")
            end
            return x
        end

        # if verbose && iter % 500 == 0
        #     println("Iter $iter: loss = $loss")
        # end
    end

    if verbose
        println("Reached max iterations, final loss = $loss")
    end
    return x
end


function compute_lK_ratio(sol, P, n, chi, d, delta, kappa, epsilon, b)
    if sol === nothing
        return (NaN, NaN)
    end
    lJ1, lJ3, lH1, lH3 = sol
    lV1 = (lH1 / lJ1^2 - 1 / lJ1)
    lV3 = (lH3 / lJ3^2 - 1 / lJ3)
    lWT = 1 / (d + delta * b * (lV1) / n)
    lWP = 1 / d

    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 +
        (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) * (d - 1) +
        (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)
    gammaYh2 = (4 / π) / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3
    return (lK1 / (lK1 + kappa / P), lK3 / (lK3 + kappa / P))
end



function sweep_learnabilities(initial_guess; alphas, chi, d, kappa, delta, epsilon, n, b)
    P_vals = d .^ collect(alphas)
    learnabilities_1 = Float64[]
    learnabilities_3 = Float64[]

    x = copy(initial_guess)  # warm-start

    for P in P_vals
        sol = nlsolve_solver(
            x,
            chi=chi, d=d, kappa=kappa, delta=delta,
            epsilon=epsilon, n=n, b=b,
            P=P,lr=1e-5, max_iter=500000, anneal=true, verbose=true
        )

        l1, l3 = compute_lK_ratio(sol, P, n, chi, d, delta, kappa, epsilon, b)
        push!(learnabilities_1, l1)
        push!(learnabilities_3, l3)

        # warm start next P with the last solution (if valid)
        if !(sol === nothing)
            x = sol
        end
    end

    return P_vals, learnabilities_1, learnabilities_3
end



end 