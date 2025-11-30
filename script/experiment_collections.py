"""
Central registry of experiment name lists.

Place groups of hardcoded experiment folder names here so analysis scripts
can import them from one place.

Edit this file to add/remove experiments; other scripts should import
`erf_cubic_P_SWEEP` or create additional named groups.
"""

# Primary default list used by `experiment_analyzer`.
erf_cubic_P_SWEEP = [
    'erf_cubic_eps_0.03_P_400_D_40_N_250_epochs_20000000_lrA_2.50e-09_time_20251125_140822',
    'erf_cubic_eps_0.03_P_200_D_40_N_250_epochs_20000000_lrA_5.00e-09_time_20251125_140822',
    'erf_cubic_eps_0.03_P_40_D_40_N_250_epochs_20000000_lrA_2.50e-08_time_20251125_140822',
    'erf_cubic_eps_0.03_P_1000_D_40_N_250_epochs_20000000_lrA_1.00e-09_time_20251125_140822',
]

# Example: you can add more named groups for different sweeps.
EXPERIMENT_GROUPS = {
    'erf_cubic_P_SWEEP': erf_cubic_P_SWEEP,
}
