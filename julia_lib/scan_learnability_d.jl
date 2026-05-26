using NLsolve
using JSON

# -------------------------
# Define functions (No longer need @everywhere)
# -------------------------
function is_physical(x)
    return all(isfinite.(x)) && all(x .> 0)
end

function residuals_with_learnability(x, chi, d, kappa, delta, epsilon, n1, n2, b, learnability)
    lJ1, lJ3, lH1, lH3, lWT, P = x

    lWP = 1.0 / d
    TrSigma = lWT + lWP * (d - 1)

    # Conjugate inter-layer discrepancies by the inverse kernels
    lV1 = -lJ1^(-1) * (lH1 - lJ1) * lJ1^(-1)
    lV3 = -lJ3^(-1) * (lH3 - lJ3) * lJ3^(-1)

    b_val = 4 / π * 1 / (1 + 2 * TrSigma)

    EChh = lH1 + lH3 +
           (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) * (d - 1) +
           (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)

    gammaYh2 = (4 / π) / (1 + 2 * EChh)

    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3

    lT1 = -(chi^2 / (kappa / P + lK1)^2 * delta) - chi^2 * kappa / (P * chi) * lK1 / (lK1 + kappa / P)
    lT3 = -(chi^2 / (kappa / P + lK3)^2 * delta) - chi^2 * kappa / (P * chi) * lK3 / (lK3 + kappa / P)

    # Original residuals
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma)) * lWT)
    rh1 = lH1 - 1 / (1 / lJ1 + gammaYh2 * lT1 / (n2 * chi))
    rh3 = lH3 - (1 / (1 / lJ3 + gammaYh2 * lT3 * epsilon^2 / (n2 * chi)))
    rj3 = lJ3 - ((16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)))
    rlWT = lWT - 1 / (d + delta * b_val * (n2 / n1) * lV1)

    # Learnability constraint
    if learnability >= 1.0 || learnability <= 0.0
        rP = P - d^1.2
    else
        rP = P - (learnability * kappa) / (lK1 * (1 - learnability))
    end

    return [rj1, rj3, rh1, rh3, rlWT, rP]
end

function nlsolve_solver_with_learnability(initial_guess;
    anneal=false,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, n2=1.0, b=1.0,
    learnability=0.5,
    anneal_steps=30000,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    x = copy(initial_guess)

    function res_func!(F, x, c)
        F[:] = residuals_with_learnability(x, c, d, kappa, delta, epsilon, n1, n2, b, learnability)
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
                if verbose
                    println("THERE WAS AN ERROR")
                    showerror(stdout, e, catch_backtrace())
                end
                return nothing
            end
        end
    else
        f2!(F, x) = res_func!(F, x, chi)
        try
            sol = nlsolve(f2!(F, x), x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
            result = sol.zero
        catch e
            if verbose
                println("THERE WAS AN ERROR")
                showerror(stdout, e, catch_backtrace())
            end
            return nothing
        end
    end

    return result
end

function solve_for_d(d, initial_guess, chi, kappa, delta, epsilon, n1, n2, b, learnability)
    solution = nlsolve_solver_with_learnability(
        initial_guess;
        anneal=true,
        chi=chi,
        d=Float64(d),
        kappa=kappa,
        delta=delta,
        epsilon=epsilon,
        n1=n1,
        n2=n2,
        b=b,
        learnability=learnability,
        anneal_steps=30000,
        max_iter=1_000_000,
        tol=1e-12,
        lr=1e-6,
        verbose=false
    )

    return (d=Float64(d), solution=solution)
end

# -------------------------
# Serial scanner function
# -------------------------
function scan_d_values_serial()
    # Parameters
    num_d_points = 10
    d_values = exp10.(range(log10(50), log10(400), length=num_d_points))
    # d_values = [63]
    kappa = 1.0
    chi = 80.0
    N = 1600
    delta = 1.0
    epsilon = 0.03
    n1 = N
    n2 = N
    b = 1.0
    learnability = 0.5

    println("Starting serial scan over d values...")
    println("d values: ", d_values)
    println("="^60)

    # Initialize results dictionary
    results = Dict(
        "lJ1" => Float64[], "lJ3" => Float64[],
        "lH1" => Float64[], "lH3" => Float64[],
        "lWT" => Float64[], "P" => Float64[],
        "d" => Float64[]
    )

    # Run solvers in series
    for d in d_values
        println("\nSolving for d = $d...")

        # Create initial guess for current d
        initial_guess = Float64[
            4/(3*π)*1/d^0.5,
            1/d^(3/2),
            4/(3*π)*1/d^0.5,
            1/d^(3/2),
            1/d,
            d
        ]

        res = solve_for_d(d, initial_guess, chi, kappa, delta, epsilon, n1, n2, b, learnability)

        d_val = res.d
        sol = res.solution

        if sol !== nothing
            push!(results["lJ1"], sol[1])
            push!(results["lJ3"], sol[2])
            push!(results["lH1"], sol[3])
            push!(results["lH3"], sol[4])
            push!(results["lWT"], sol[5])
            push!(results["P"], sol[6])
            push!(results["d"], d_val)

            println("✓ Success: lJ1=$(round(sol[1], digits=6)), P=$(round(sol[6], digits=6))")
        else
            println("✗ Failed for d = $d")
            for key in ["lJ1", "lJ3", "lH1", "lH3", "lWT", "P"]
                push!(results[key], NaN)
            end
            push!(results["d"], d_val)
        end
    end

    # Save to JSON
    output_file = "d_scan_results_serial.json"
    open(output_file, "w") do f
        JSON.print(f, results, 4)
    end

    println("\nScan complete! Results saved to: $output_file")
    return results
end

# -------------------------
# Main execution
# -------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("SERIAL D-VALUE SCANNER")
    println("="^60)

    results = scan_d_values_serial()
end