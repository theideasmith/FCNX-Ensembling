# Needs the multiwell solver (solve_multiwell_balance) that is no longer in julia_lib/FCS2_VGA.jl;
# point FCS2_MULTIWELL at a copy of FCS2_VGA.jl that still has it.
include(get(ENV, "FCS2_MULTIWELL", joinpath(@__DIR__, "..", "..", "julia_lib", "FCS2_VGA.jl")))
using .FCS2_VGA
using Printf

const OUT = "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/action_h0_activation_plots/RubinRegime"
const D, N, A0, S0, K = 150.0, 700.0, 8.0 / 700.0, 0.5, 0.07

solve(; chi, P, eps) = FCS2_VGA.solve_multiwell_balance(; d=D, n1=N, chi=chi, kappa=K, P=P,
    a0=A0, s0=S0, epsilon=eps, he1=1.0, wmax=8.0)

function sweep(name, xs, kw)
    open(joinpath(OUT, name), "w") do io
        println(io, "x,eps,L1,L3,pi,mu_in,mu_out,nwells")
        for eps in (-0.3, 0.3), x in xs
            t = @elapsed s = solve(; eps=eps, kw(x)...)
            if s === nothing
                @printf("%s eps=%+.1f x=%g  none (%.1fs)\n", name, eps, x, t)
                println(io, "$x,$eps,NaN,NaN,NaN,NaN,NaN,0")
            else
                @printf("%s eps=%+.1f x=%g L1=%.3f L3=%.3f pi=%.3f (%.1fs)\n", name, eps, x, s.L1, s.L3, s.pi, t)
                println(io, "$x,$eps,$(s.L1),$(s.L3),$(s.pi),$(s.mu_in),$(s.mu_out),$(s.nwells)")
            end
            flush(io)
        end
    end
end

if get(ARGS, 1, "") == "test"
    @time s = solve(; chi=1.0, P=3000.0, eps=-0.3)
    s === nothing ? println("none") : println((s.L1, s.L3, s.pi, s.mu_in, s.mu_out))
else
    sweep("learnability_vs_alpha_kappa0p07_chi1.csv", range(1.0, 2.2; length=25), α -> (chi=1.0, P=D^α))
    sweep("learnability_vs_chi_kappa0p07_P3000.csv", exp.(range(log(1.0), log(50.0); length=25)), χ -> (chi=χ, P=3000.0))
end
