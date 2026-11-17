include("FCS3_VGA.jl")
using .FCS3_VGA
using Printf

# Construct Problem Parameters
params = ProblemParams3(
    d=150.0f0,
    P=3000.0f0,
    χ=1600,
    n1=1600.0f0,
    n2=1600.0f0,
    κ=2.0f0,
    ϵ=0.074f0,
    δ=1.0f0
)

# Initial guess vector [lJ1, lJ3, lH1, lH3, lWT, mu_J1, mu_J3, mu_H1, mu_H3, mu_WT]
initial_guess = [0.5, 0.1, 0.5, 0.1, 0.1, 0.1, 0.01, 0.1, 0.01, 0.1]

println("Solving FCN3 system with parameters: d=150, P=600, χ=1.0, N=1600, κ=2.0...")
sol = solve_FCN3_Erf(params, initial_guess; anneal_chi=true, anneal_steps=10000, tol=1e-8)

println("\n" * "="^65)
println(" Results for FCS3_VGA Solver")
println("="^65)
@printf("  lJ1 (Linear Weight Kernel)  : %.6f\n", sol.lJ1)
@printf("  lJ3 (Cubic Weight Kernel)   : %.6f\n", sol.lJ3)
@printf("  lH1 (Linear Pre-Activation) : %.6f\n", sol.lH1)
@printf("  lH3 (Cubic Pre-Activation)  : %.6f\n", sol.lH3)
@printf("  lK1 (Linear Activation K)   : %.6f\n", sol.lK1)
@printf("  lK3 (Cubic Activation K)    : %.6f\n", sol.lK3)
@printf("  lWT (Total Weight Norm)     : %.6f\n", sol.lWT)
@printf("  muW (Weight Projection Mean): %.6f\n", sol.muW)
@printf("  Learnability (Linear Mode)  : %.6f\n", sol.learnability1)
@printf("  Learnability (Cubic Mode)   : %.6f\n", sol.learnability3)
println("="^65)