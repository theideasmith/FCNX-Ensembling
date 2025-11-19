import torch
from scipy.optimize import fsolve
from scipy.optimize import root
import traceback

import autograd.numpy as np
from autograd import jacobian
from tqdm import tqdm


# Set variables to parsed values
n = 1000.0  # Ensure float
chi = float(n)  # Ensure float
kappa = 1.0
d = 5000.0  # Ensure float
b = 4 / (3 * np.pi)
epsilon = 0.03
delta = 1.0
# Define the residual function
def residual(vars, d, delta, b, N, χ, kappa, P, epsilon):
    # Unpack variables
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
    
    # Intermediate calculations
    # lWT = 1.0 / (d + delta * b * lV1 / N), where lV1 = lV1_re + 1j * lV1_im
    denom = d + delta * b * (lV1_re + 1j * lV1_im) / N
    denom_re = d + delta * b * lV1_re / N  # Real part of denominator
    denom_im = delta * b * lV1_im / N      # Imaginary part
    denom_mag2 = denom_re**2 + denom_im**2
    lWT_re = denom_re / denom_mag2         # Real part of 1/denom
    lWT_im = -denom_im / denom_mag2        # Imaginary part of 1/denom
    
    lWP = 1.0 / d
    TrSigma = lWT_re + lWP * (d - 1)       # TrSigma is real
    EChh = lH1 + lH3 + ((8.0 / (np.pi * (1.0 + 2.0 * TrSigma)**3)) * (213.0 * (lWT_re**3 - 3 * lWT_re * lWT_im**2) + 9.0 * lWT_re)) * (d - 1) + \
           (4.0 / (np.pi * (1.0 + 2.0 * TrSigma)) * lWT_re) * (d - 1.0)
    gammaYh2 = (4.0 / np.pi) * 1.0 / (1.0 + 2.0 * EChh)
    
    # K1 and equations for first set
    lK1 = gammaYh2 * lH1
    # eq1 = lT1 - (-χ**2.0 / (kappa / P + lK1)**2.0 * delta)
    denom_k1 = kappa / P + lK1
    denom_k1_sq = denom_k1**2
    eq1_re = lT1_re - (-χ**2.0 / denom_k1_sq * delta)
    eq1_im = lT1_im  # Since the right-hand side is real
    
    # eq2 = lV1 - (1.0 / lJ1**2 * lH1 - 1.0 / lJ1)
    eq2_re = lV1_re - (1.0 / lJ1**2 * lH1 - 1.0 / lJ1)
    eq2_im = lV1_im  # Since the right-hand side is real
    
    # eq3 = lH1 - (1.0 / (1.0 / lJ1 + gammaYh2 * lT1 / (N * χ)))
    denom_eq3 = 1.0 / lJ1 + gammaYh2 * (lT1_re + 1j * lT1_im) / (N * χ)
    denom_eq3_re = 1.0 / lJ1 + gammaYh2 * lT1_re / (N * χ)
    denom_eq3_im = gammaYh2 * lT1_im / (N * χ)
    denom_eq3_mag2 = denom_eq3_re**2 + denom_eq3_im**2
    eq3 = lH1 - (denom_eq3_re / denom_eq3_mag2)
    
    # eq4 = lJ1 - (4.0 / (np.pi * (1.0 + 2.0 * TrSigma)) * lWT)
    eq4 = lJ1 - (4.0 / (np.pi * (1.0 + 2.0 * TrSigma)) * lWT_re)
    
    # K3 and equations for second set
    lK3 = gammaYh2 * lH3
    # eq5 = lT3 - (-χ**2.0 / (kappa / P + lK3)**2.0 * delta)
    denom_k3 = kappa / P + lK3
    denom_k3_sq = denom_k3**2
    eq5_re = lT3_re - (-χ**2.0 / denom_k3_sq * delta)
    eq5_im = lT3_im  # Since the right-hand side is real
    
    # eq6 = lV3 - (1.0 / lJ3**2 * lH3 - 1.0 / lJ3)
    eq6_re = lV3_re - (1.0 / lJ3**2 * lH3 - 1.0 / lJ3)
    eq6_im = lV3_im  # Since the right-hand side is real
    
    # eq7 = lH3 - (1.0 / (lJ3**(-1) + gammaYh2 * lT3 * epsilon**2.0 / (N * χ)))
    denom_eq7 = 1.0 / lJ3 + gammaYh2 * (lT3_re + 1j * lT3_im) * epsilon**2.0 / (N * χ)
    denom_eq7_re = 1.0 / lJ3 + gammaYh2 * lT3_re * epsilon**2.0 / (N * χ)
    denom_eq7_im = gammaYh2 * lT3_im * epsilon**2.0 / (N * χ)
    denom_eq7_mag2 = denom_eq7_re**2 + denom_eq7_im**2
    eq7 = lH3 - (denom_eq7_re / denom_eq7_mag2)
    
    # eq8 = lJ3 - ((8.0) / (np.pi * (1.0 + 2.0 * TrSigma)**3) * (213.0 * lWT**3 + 9.0 * lWT))
    eq8 = lJ3 - ((8.0 / (np.pi * (1.0 + 2.0 * TrSigma)**3)) * (213.0 * (lWT_re**3 - 3 * lWT_re * lWT_im**2) + 9.0 * lWT_re))
    
    # Return residuals as a real-valued vector
    return np.array([
        eq1_re, eq1_im,
        eq2_re, eq2_im,
        eq3,
        eq4,
        eq5_re, eq5_im,
        eq6_re, eq6_im,
        eq7,
        eq8
    ])

# Compute the Jacobian
jacobian_func = jacobian(residual)

# Define fsolve functions
def fsolve_func(vars, d, delta, b, N, χ, kappa, P, epsilon):
    return residual(vars, d, delta, b, N, χ, kappa, P, epsilon)

def fsolve_jacobian(vars, d, delta, b, N, χ, kappa, P, epsilon):
    return jacobian_func(vars, d, delta, b, N, χ, kappa, P, epsilon)


def compute_lK1(sol):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
    lV1 = lV1_re + 1j * lV1_im
    lWT = 1 / (d + delta * b * lV1 / n)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8.0) / (np.pi * (1 + 2 * TrSigma)**3) * (213.*lWT**3 + 9.*lWT)) * (d-1) + (4. / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4.0 / np.pi) * 1.0 / (1.0 + 2.0 * EChh)
    lK1 = gammaYh2 * lH1
    return lK1

def _solve_one_init(args):
    init, d, P, χ, N = args
    init = np.array(init)
    try:
        sol, info, ier, mesg = fsolve(
            lambda vars:fsolve_func(vars, d, delta, b, N, χ, kappa, P, epsilon), 
            init, 
            maxfev = 200000,
            # fprime=lambda vars: fsolve_jacobian(vars, d, delta, b, N, χ, kappa, P, epsilon), 
            full_output=True)

        if ier == 1:
            residuals = np.abs(fsolve_func(sol,  d, delta, b, N, χ, kappa, P, epsilon))
            converged = np.max(residuals) < 1e-2
            positive_definite = (sol[4] >= 0) & (sol[5] >= 0) & (sol[10] >= 0) & (sol[11] >= 0)
            if converged and positive_definite:
                return sol
            else:
                pass
                # print('Solver failed to converge within error\n')
                # print(f'Residual: {np.max(residuals)}\n')
                # print(f'Positivity: {positive_definite}\n')
        else:
            pass
            # print(f'Solver failed to converge: {mesg}\n')

    except Exception as e:

        print("Full error details\n")
        print(traceback.format_exc())

        return None
    return None

def rand_initial():
    return np.array([
        np.random.normal(1e5, 1e1),     # lT1_re
        np.random.normal(1e5, 1e2),     # lT1_im
        np.random.normal(1e5, 1e2),     # lV1_re
        np.random.normal(1e5, 1e2),     # lV1_im
        np.random.normal(0.025, 0.01), # lH1
        np.random.normal(0.025, 0.01), # lJ1
        np.random.normal(0, 1e2),     # lT3_re
        np.random.normal(0, 1e2),     # lT3_im
        np.random.normal(1e6, 1e3),     # lV3_re
        np.random.normal(1e6, 1e3),     # lV3_im
        np.random.normal(0.025, 0.01), # lH3
        np.random.normal(0.025, 0.01)  # lJ3
    ])

def anneal(i0, d, P, N, chi0=1.0e-8, collect=False):
    # Initialize the solution list with the initial condition
    current_solutions = [(chi0, i0)]
    beta = np.log(N / chi0) / 1000
    t=0
    with tqdm(total=1000, desc=f"Annealing d={d:.0f}, P={P:.0e}") as pbar:
        while True:
            # Compute chi_t
            chi_t = chi0 * np.exp(beta * t)
            # print(f'chi_t: {chi_t:.8f}')

            # Get the previous solution
            _, previous_solution = current_solutions[-1]

            args = (previous_solution, d, P, chi_t, N)

            # Solve for the current chi_t
            sol = _solve_one_init(args)
            if sol is None:
                pbar.close()
                return None

            # Append the new solution
            current_solutions.append((chi_t, sol))

            # Check termination condition
            if chi_t >= N:
                pbar.close()
                if collect:
                    return current_solutions
                return current_solutions[-1][1]

            # Increment t
            t += 1
            pbar.update(1)

def get_lKs(sol, d):

    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
    lV1 = lV1_re + 1j * lV1_im
    lWT = 1 / (d + delta * b * lV1 / n)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3
    return np.real(lK1), np.real(lK3)

import numpy as np
import multiprocessing as mp
from collections import defaultdict

# Assuming d, kappa, initial_conditions, anneal, and get_lKs are defined elsewhere
# N=1000
def solv(args):
    i, d, P, N, kappa = args
    sl = anneal(i, d, P, N, chi0 = 1e-8, collect=False)
    return sl, args
def sol_to_learnability(sl, args):
    i, d, P, N, kappa = args
    lK1, lK3 = get_lKs(sl,d)
    l1 = lK1 / (lK1 + kappa / P)
    l3 = lK3 / (lK3 + kappa / P)
    return l1, l3, sl

def gen_initial_conditions(num_samples = 10000):
    initial_conditions = [rand_initial() for _ in range(num_samples)]
    return initial_conditions


initial_conditions = gen_initial_conditions(num_samples=150)


alphas = np.linspace(0.1, 2.0, num=80)

N_values = [600]#sorted([2*1e5, 3*1e5,8*1e5, 1e4,2*1e4,5*1e4,8*1e4,1e6,2*1e6])
d_values = np.logspace(np.log10(25), np.log10(1000), 15)
P_values = {d: d ** alphas for d in d_values}
# Prepare all tasks as (i, P, kappa) pairs
tasks = [(i, d, P, N, kappa) for d in d_values for N in N_values for P in P_values[d] for i in initial_conditions ]

# Parallelize using 22 cores
with mp.Pool(24) as pool:
    results = [sol_to_learnability(sl,args) for sl, args in pool.imap(solv, tasks) if sl is not None]
# Organize results by P


idx = 0
res_dict_by_d = []
for d in d_values:
    res_dict_by_N = []
    for N in N_values:
        res_dict = defaultdict(list)

        for P in P_values[d]:
            # print('Hi')
            for i in initial_conditions:
                try:
                    res_dict[P].append(results[idx])

                    idx += 1
                except Exception as e:

                    pass

        # Compute lists and stddevs per P
        l1s_per_P = {}
        l3s_per_P = {}
        stddevs_l1 = {}
        stddevs_l3 = {}
        for P in [P for P in P_values]: 
            if len(res_dict[P]) > 0:

                try:
                    l1s = np.array([r[0] for r in res_dict[P]])
                    l3s = np.array([r[1] for r in res_dict[P]])
                    l1s_per_P[P] = l1s
                    l3s_per_P[P] = l3s
                    stddevs_l1[P] = np.std(l1s)
                    stddevs_l3[P] = np.std(l3s)
                except Exception as e:
                    pass
            res_dict_by_N.append([l1s_per_P,l3s_per_P,stddevs_l1,stddevs_l3])
    res_dict_by_d.append(res_dict_by_N)

# Save res_dict_by_d with correct parameters
import pickle
filename = f'N_{N_values[0]}_d_{d_values[0]}_chi0_1e-8_num_samples{len(initial_conditions)}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(res_dict_by_d, f)

# Now plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.rcdefaults()  # Resets all settings to their default values
# Set plot style to arXiv quality, avoiding specific font if unavailable
rcParams['font.family'] = 'serif'  # Use default serif font available on system
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1
rcParams['lines.linewidth'] = 2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['xtick.minor.size'] = 3
rcParams['ytick.minor.size'] = 3
rcParams['legend.fontsize'] = 10
rcParams['axes.labelsize'] = 12
rcParams['figure.figsize'] = (10, 8)

# Define colormaps for each d value
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
# Define markers for each alpha value (but since alpha is continuous, we'll use different markers for different d)
markers_d = ['o', 's', '^', 'D', 'v']  # Different markers for each d
def nanrm(input_dict):
    
    return {key: value for key, value in input_dict.items() if not value.shape == 0}
# Create the plot
fig, ax = plt.subplots()

# Loop through d values
for k in range(len(res_dict_by_d)):
    d = d_values[k]
    P_values_for_d = P_values[d]
    cmap_name = cmaps[k % len(cmaps)]
    marker = markers_d[k % len(markers_d)]
    try:
        res_dict_by_N = res_dict_by_d[k]
        selected_N_indices = sorted(N_values)
    except Exception as e:
        print(e)
    for idx in range(len(selected_N_indices)):
        try:
            N = sorted(N_values)[idx]
            i = sorted(N_values).index(N)

            # print([g.shape for g in res_dict_by_N])
            # Extract data for this N
            l1s_per_P, l3s_per_P, stddevs_l1, stddevs_l3 = res_dict_by_N[idx]
            mean_l1s = np.array([np.mean(l1s_per_P[P]) for P in P_values_for_d])
            mean_l3s = np.array([np.mean(l3s_per_P[P]) for P in P_values_for_d])
            std_l1s = np.array([stddevs_l1[P] for P in P_values_for_d])
            std_l3s = np.array([stddevs_l3[P] for P in P_values_for_d])
            alphas = np.log(P_values_for_d) / np.log(d)
            # print(alphas.shape)

            l1s = [l1s_per_P[P] for P in P_values_for_d]
            
                # raise(Exception)
            # print(l1s.shape)
            l3s = np.array([l3s_per_P[P] for P in l3s_per_P.keys()])
            # Get colors for this N using the d-specific colormap, different for l1 and l3
            colors_l1 = plt.cm.get_cmap(cmap_name)(np.linspace(0, 0.5, len(selected_N_indices)))
            colors_l3 = plt.cm.get_cmap(cmap_name)(np.linspace(0.5, 1, len(selected_N_indices)))
            color_l1 = colors_l1[selected_N_indices.index(N)]
            color_l3 = colors_l3[selected_N_indices.index(N)]

            # Plot mean l1 with error bars
            ax.errorbar(alphas[:mean_l1s.shape[0]], mean_l1s, yerr=std_l1s, fmt='x-', color=color_l1, ecolor=color_l1, capsize=3, markersize=5, label=fr'$\lambda_1$, N={N:.0e}, d={d:.0f}')

            ax.errorbar(alphas[:mean_l3s.shape[0]], mean_l3s, yerr=std_l3s, fmt='o--', color=color_l3, ecolor=color_l3, capsize=3, markersize=5, label=fr'$\lambda_3$, N={N:.0e}, d={d:.0f}')
        except Exception as e:
            print(e)

# Customize the plot
ax.set_xlabel(r'$\alpha = \log P / \log d$', fontsize=14)
ax.set_ylabel(r'Mean $l_1$ and $l_3$', fontsize=14)
ax.set_title('Mean $l_1$ and $l_3$ vs. $\\alpha$', fontsize=16, pad=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.minorticks_on()

# Add legend to the right
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)

# Tight layout to prevent label clipping
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Adjust to make space for legend

# Save the plot in high quality
plot_filename = f'l1_l3_vs_alpha_N_{N_values[0]}_d_{d_values[0]}_legend.pdf'
plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches='tight')
plt.show()