import torch
import numpy as np
from pathlib import Path
import sys
import json
import pickle
import subprocess
import tempfile
import re
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
from kappa_eff_solver import compute_kappa_eff
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from scipy.special import erf
import argparse

@dataclass(frozen=True)
class ExperimentGroup:
    name: str
    model_dirs: list[str]


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "experiment"


def collect_model_dirs(base_dir: Path, pattern: re.Pattern[str]) -> list[str]:
    if not base_dir.is_dir():
        return []
    return sorted(
        str(path)
        for path in base_dir.iterdir()
        if path.is_dir() and pattern.match(path.name)
    )

model_dirs = [
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_2'
]
# model_dirs = [
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_1e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_2'
# ]

import os
MODELDIR = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults'
model_dirs = [os.path.join(MODELDIR, d) for d in os.listdir(MODELDIR)]
# Only take directories (not files)
model_dirs = [d for d in model_dirs if os.path.isdir(d)]

model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_80.0_lr_0.0003_T_2.0_seed_42']

model_dirs = [
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_2',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_3']
model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.03', 
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_3_eps_0.03']

model_dirs = [
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_1'
]

# model_dirs = [
#     "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
#     "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
#     "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
#     "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
# ]

# model_dirs = [
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N300_chi_300.0_lr_0.0001_T_0.1_seed_0_eps_0.03',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_0.1_seed_0_eps_0.03'
# ]

model_dirs = [

'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_2.0_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_0.0001_T_2.0_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_0.0001_T_2.0_seed_0_eps_0.03'
    
]

model_dirs = [
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_1_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_2_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N300_chi_300.0_lr_0.0001_T_0.1_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_0.1_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_5e-06_T_0.1_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_5e-06_T_0.1_seed_42_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_5e-06_T_0.1_seed_0_eps_0.03",

    ]
# Existing completed runs from the old beta/alpha sweep
# (red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16).
# Hardcoded so a re-run under new scaling does not change this group's file list.
_BETA_ALPHA_EXISTING_MODEL_DIRS = [
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta1.000_alpha1.00_d50_P160_N250_sa01.0000_kappa0.3000_seed0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta1.741_alpha1.63_d66_P231_N330_sa00.7579_kappa0.2718_seed0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta3.031_alpha2.66_d87_P333_N435_sa00.5743_kappa0.2462_seed0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta5.278_alpha4.33_d115_P480_N574_sa00.4353_kappa0.2231_seed0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta9.190_alpha7.06_d152_P693_N758_sa00.3299_kappa0.2021_seed0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/red_robin_sample_complexity_sweep_cubic_beta_alpha_P0160_Pmax1000_betamax16/models/beta16.000_alpha11.51_d200_P1000_N1000_sa00.2500_kappa0.1831_seed0',
]

# Learnable beta/alpha sweep (new scaling under Pmax22500_betamax9).
_LEARNABLE_BETA_ALPHA_MODELS_DIR = (
    Path(__file__).parent
    / "red_robin_sample_complexity_sweep_cubic_learnable_beta_alpha_P0160_Pmax22500_betamax9"
    / "models"
)
_learnable_beta_alpha_pattern = re.compile(
    r"^learnable_beta[\d.]+_alpha[\d.]+_d\d+_P\d+_N\d+_sa0[\d.]+_kappa[\d.]+_seed\d+$"
)

# Invariant alpha/beta sweep (eps=1/2, nu=1/4 so kappa ~ beta^{-1/2} at fixed alpha).
_INVARIANT_BETA_ALPHA_MODELS_DIR = (
    Path(__file__).parent
    / "red_robin_alpha_beta_invariant_P0160_Pmax3674_betamax9"
    / "models"
)
_invariant_beta_alpha_pattern = re.compile(
    r"^invariant_beta[\d.]+_alpha[\d.]+_d\d+_P\d+_N\d+_sa0[\d.]+_kappa[\d.]+_seed\d+$"
)

# Fixed-P=1500 invariant beta sweep (N0=1000, Asnap, eps=0.03, schedule).
_INVARIANT_PFIXED1500_MODELS_DIR = (
    Path(__file__).parent
    / "red_robin_alpha_beta_invariant_Pfixed1500_betamax9_nu0.25_N01000_lam_eq_omega_alpha_fixed_ens1_Asnap_eps0.03_schedule"
    / "models"
)

from n_chi_eq_N_linear_schedule import (  # noqa: E402
    MODELS_DIR as _N_CHI_EQ_N_MODELS_DIR,
)

# Restarted P-sweep run dirs (new train lrs in P_LR, not the old P_LR_SOURCE checkpoints).
from red_robin_launcher_P_sweep import (  # noqa: E402
    P_VALUES as _P_SWEEP_P_VALUES,
    SEEDS as _P_SWEEP_SEEDS,
    classic_run_dir as _p_sweep_classic_run_dir,
)
_P_SWEEP_MODEL_DIRS = [
    str(_p_sweep_classic_run_dir(p_val, seed))
    for p_val in _P_SWEEP_P_VALUES
    for seed in _P_SWEEP_SEEDS
]

# Journal Langevin cubic (d, P) sweep: full MF chi=N, T=1, eps=1/2, He1+He3 target.
# Run dirs are created by link_langevin_cubic_runs.py, which symlinks
# model_final.pt back to journal/LearningCubic_models checkpoints.
_LANGEVIN_CUBIC_EP50K_DIR = Path(__file__).parent / "LangevinCubic_ep50000"
_LANGEVIN_CUBIC_EP1M_DIR = Path(__file__).parent / "LangevinCubic_ep1000000"
_langevin_cubic_pattern = re.compile(
    r"^d(?P<d>\d+)_P(?P<P>\d+)_N\d+_chi_[\d.]+_lr_[\d.eE+-]+_T_[\d.]+_seed_\d+_eps_[\d.]+$"
)


def langevin_cubic_model_dirs(base_dir: Path, dims=None) -> list[str]:
    """Langevin cubic run dirs ordered by (d, P) rather than lexicographically."""
    if not base_dir.is_dir():
        return []
    found = []
    for path in base_dir.iterdir():
        if not path.is_dir():
            continue
        match = _langevin_cubic_pattern.match(path.name)
        if match is None:
            continue
        d_val = int(match.group("d"))
        if dims is not None and d_val not in dims:
            continue
        found.append((d_val, int(match.group("P")), str(path)))
    return [path for _d, _p, path in sorted(found)]


EXPERIMENT_GROUPS = [
    ExperimentGroup(
        name="ScaledDownW0",
        model_dirs=[
            '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P750_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074_s0_0.1111111111111111_sigmaW0_0.0011111111111111111',
            '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P500_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074_s0_0.25_sigmaW0_0.0025',
            '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P250_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074_s0_1.0_sigmaW0_0.01',
            '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1000_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074_s0_0.0625_sigmaW0_0.000625',
        ],
    ),
    ExperimentGroup(
        name="SampleComplexityTestI",
        model_dirs=list(_P_SWEEP_MODEL_DIRS),
    ),
    ExperimentGroup(
        name="SampleComplexityBetaAlpha",
        model_dirs=list(_BETA_ALPHA_EXISTING_MODEL_DIRS),
    ),
    ExperimentGroup(
        name="SampleComplexityLearnableBetaAlpha",
        model_dirs=collect_model_dirs(_LEARNABLE_BETA_ALPHA_MODELS_DIR, _learnable_beta_alpha_pattern),
    ),
    ExperimentGroup(
        name="AlphaBetaInvariant",
        model_dirs=collect_model_dirs(_INVARIANT_BETA_ALPHA_MODELS_DIR, _invariant_beta_alpha_pattern),
    ),
    ExperimentGroup(
        name="InvariantPfixed1500Asnap",
        model_dirs=collect_model_dirs(
            _INVARIANT_PFIXED1500_MODELS_DIR, _invariant_beta_alpha_pattern
        ),
    ),
    ExperimentGroup(
        name="NChiEqNLinearSchedule",
        model_dirs=collect_model_dirs(
            _N_CHI_EQ_N_MODELS_DIR,
            re.compile(
                r"^d\d+_P\d+_N\d+_chi_[\d.]+_lr_[\d.eE+-]+_T_[\d.]+_seed_\d+_eps_[\d.]+_schedule$"
            ),
        ),
    ),
    ExperimentGroup(
        name="ChiN5Schedule",
        model_dirs=[
            str(
                Path(__file__).parent
                / "d150_P600_N1400_chi_280.0_lr_0.0001_T_2.0_seed_0_eps_0.03_schedule"
            ),
        ],
    ),
    ExperimentGroup(
        name="P400Schedule",
        model_dirs=[
            str(
                Path(__file__).parent
                / "d100_P400_N700_chi_700.0_lr_1e-05_T_0.1_seed_0_eps_0.074_schedule"
            ),
        ],
    ),
    ExperimentGroup(
        name="LargeN",
        model_dirs=[
            str(
                Path(__file__).parent
                / "d150_P600_N3800_chi_3800.0_lr_0.0001_T_2.0_seed_0_eps_0.03"
            ),
        ],
    ),
    ExperimentGroup(
        name="LangevinCubicEp50k",
        model_dirs=langevin_cubic_model_dirs(_LANGEVIN_CUBIC_EP50K_DIR),
    ),
    *(
        ExperimentGroup(
            name=f"LangevinCubicEp50kD{d_val}",
            model_dirs=langevin_cubic_model_dirs(
                _LANGEVIN_CUBIC_EP50K_DIR, dims={d_val}
            ),
        )
        for d_val in (5, 10, 20, 30)
    ),
    ExperimentGroup(
        name="LangevinCubicEp1M",
        model_dirs=langevin_cubic_model_dirs(_LANGEVIN_CUBIC_EP1M_DIR),
    ),
    *(
        ExperimentGroup(
            name=f"LangevinCubicEp1MD{d_val}",
            model_dirs=langevin_cubic_model_dirs(
                _LANGEVIN_CUBIC_EP1M_DIR, dims={d_val}
            ),
        )
        for d_val in (20,)
    ),
]
EXPERIMENT_GROUP_BY_NAME = {group.name: group for group in EXPERIMENT_GROUPS}

OUTPUT_BASE_DIR = Path(__file__).parent / "action_h0_activation_plots"

# Experiment groups that use (alpha, beta) MF scaling. Values are imported from
# the launcher that generated the runs so the plot footer matches the experiment.
_SCALING_EXPONENT_MODULES = {
    "AlphaBetaInvariant": "red_robin_alpha_beta_invariant",
    "InvariantPfixed1500Asnap": "red_robin_alpha_beta_invariant",
    "SampleComplexityBetaAlpha": "red_robin_sample_complexity_sweep_cubic_beta_alpha",
    "SampleComplexityLearnableBetaAlpha": "red_robin_sample_complexity_sweep_cubic_beta_alpha",
}
_INVARIANT_ACTION_GROUPS = {"AlphaBetaInvariant", "InvariantPfixed1500Asnap"}


def format_math_fraction(value: float) -> str:
    frac = Fraction(value).limit_denominator(16)
    if frac.denominator == 1:
        return str(frac.numerator)
    sign = "-" if frac < 0 else ""
    return rf"{sign}\frac{{{abs(frac.numerator)}}}{{{frac.denominator}}}"


def scaling_exponents_footnote(group_name: str) -> str | None:
    module_name = _SCALING_EXPONENT_MODULES.get(group_name)
    if module_name is None:
        return None
    try:
        module = import_module(module_name)
    except ImportError:
        return None

    epsilon = format_math_fraction(module.EPSILON)
    nu = format_math_fraction(module.NU)
    rho = format_math_fraction(module.RHO)
    lam = format_math_fraction(module.LAMBDA)
    omega = format_math_fraction(module.OMEGA)
    lines = [
        (
            r"$d \sim \beta^{\epsilon},\quad "
            r"N \sim \beta^{\nu},\quad "
            r"\sigma_a^{2} \sim \beta^{\rho},\quad "
            r"P \sim \alpha^{\lambda},\quad "
            r"\kappa \sim (\alpha/\beta)^{2\omega}$"
        ),
        (
            rf"$\epsilon={epsilon},\quad "
            rf"\nu={nu},\quad "
            rf"\rho={rho},\quad "
            rf"\lambda={lam},\quad "
            rf"\omega={omega}$"
        ),
    ]
    if group_name in _INVARIANT_ACTION_GROUPS:
        lines.insert(
            0,
            r"Action $F = 2P^{2}\sigma_a^{2}/(\pi\kappa^{2}\,d\,N)$ held invariant",
        )
    return "\n".join(lines)


def annotate_theory_empirical_gaps(ax, p_values, empirical, theory, empirical_std=None) -> None:
    emp = np.asarray(empirical, dtype=np.float64)
    th = np.asarray(theory, dtype=np.float64)
    if empirical_std is None:
        std = np.zeros_like(emp)
    else:
        std = np.asarray(empirical_std, dtype=np.float64)
        if std.shape != emp.shape:
            std = np.zeros_like(emp)

    y_tops = []
    for x, y_emp, y_th, y_err in zip(p_values, emp, th, std):
        if not (np.isfinite(y_emp) and np.isfinite(y_th) and y_th != 0.0):
            continue
        gap_pct = 100.0 * (y_emp - y_th) / y_th
        err = y_err if np.isfinite(y_err) else 0.0
        y_top = max(y_emp + abs(err), y_th)
        y_tops.append(y_top)
        sign = "+" if gap_pct >= 0 else "\u2212"
        ax.annotate(
            f"{sign}{abs(gap_pct):.1f}%",
            xy=(x, y_top),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.15",
            clip_on=False,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    if y_tops:
        ymin, ymax = ax.get_ylim()
        data_max = max(y_tops)
        if ymax < data_max * 1.18:
            ax.set_ylim(ymin, max(ymax, data_max) * 1.18)


def annotate_scaling_exponents(fig, group_name: str) -> None:
    footnote = scaling_exponents_footnote(group_name)
    if not footnote:
        fig.tight_layout()
        return
    n_lines = footnote.count("\n") + 1
    fig.tight_layout(rect=[0, 0.04 + 0.045 * n_lines, 1, 1])
    fig.text(
        0.5,
        0.01,
        footnote,
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=1.45,
        color="0.2",
    )


def parse_config_from_dirname(dirname):
    dir_path = Path(dirname)
    config_path = dir_path / "config.json"

    def seed_from_dirname(name: str):
        seed_match = re.search(r"seed_?(\d+)", name)
        return int(seed_match.group(1)) if seed_match else None

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        seed = cfg.get("dataset_seed", cfg.get("seed"))
        if seed is None:
            seed = seed_from_dirname(dir_path.name)
        else:
            seed = int(seed)
        return (
            int(cfg["d"]),
            int(cfg["P"]),
            int(cfg["N"]),
            float(cfg["chi"]),
            seed,
            float(cfg["temperature"]),
            cfg.get("eps"),
            cfg.get("s0"),
        )

    name = dir_path.name
    beta_alpha_match = re.match(
        r"(?:learnable_)?beta(?P<beta>[\d.]+)_alpha(?P<alpha>[\d.]+)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)"
        r"_sa0(?P<sa0>[\d.]+)_kappa(?P<kappa>[\d.]+)_seed(?P<seed>\d+)",
        name,
    )
    if beta_alpha_match:
        d = int(beta_alpha_match.group("d"))
        p_val = int(beta_alpha_match.group("P"))
        n = int(beta_alpha_match.group("N"))
        chi = float(n)
        seed = int(beta_alpha_match.group("seed"))
        kappa = float(beta_alpha_match.group("kappa"))
        temperature = 2.0 * kappa
        # Learnable / beta-alpha sweeps use task_eps=0.0 and s0=1.0.
        return d, p_val, n, chi, seed, temperature, 0.0, 1.0

    parts = name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4])

    T = None
    if 'T' in parts:
        T = float(parts[parts.index('T') + 1])

    seed = seed_from_dirname(name)

    epsilon = None
    if 'eps' in parts:
        epsilon = float(parts[parts.index('eps') + 1])

    s0 = None
    if 's0' in parts:
        s0 = float(parts[parts.index('s0') + 1])

    return d, P, N, chi, seed, T, epsilon, s0


def reconstruct_training_inputs(P, d, seed, device):
    """Replay the training-script RNG: torch.manual_seed(seed); randn(P, d) on CUDA.

    CPU and CUDA generators are independent, so reconstructing on CPU after a
    CUDA training run yields a different dataset (effectively a test set).
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    gen_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    X = torch.randn(int(P), int(d), device=gen_device)
    return X.to(device)


def make_training_dataset(P, d, seed, eps, device):
    X = reconstruct_training_inputs(P, d, seed, device)
    z = X[:, 0]
    he3 = (z ** 3 - 3.0 * z) / (6.0 ** 0.5)
    y = z + float(eps) * he3
    return X, y, z, he3

def load_a_snapshots(model_dir, device=None):
    """Load late Langevin (A, W0) snapshots if present (default: CPU).

    Returns
    -------
    A : (T, ens0, N)
    W0 : (T, ens0, N, d)
    epochs : list[int]
    or None if no snapshots exist.
    """
    snap_dir = Path(model_dir) / "A_snapshots"
    paths = sorted(snap_dir.glob("epoch_*.pt"))
    if not paths:
        return None
    map_location = "cpu" if device is None else device
    As, W0s, epochs = [], [], []
    for path in paths:
        snap = torch.load(path, map_location=map_location, weights_only=False)
        As.append(snap["A"])
        W0s.append(snap["W0"])
        epochs.append(int(snap["epoch"]))
    A = torch.stack(As, dim=0)
    W0 = torch.stack(W0s, dim=0)
    if device is not None:
        A = A.to(device)
        W0 = W0.to(device)
    return A, W0, epochs


def load_model(model_dir, device, use_a_snapshots=False):
    """Load trained FCN2 from the final checkpoint.

    Snapshot averaging is handled explicitly by callers (see
    ``copy_snapshot_into_model_`` / ``compute_one_model``), not by expanding
    ``ens``, which OOMs for large T.
    """
    del use_a_snapshots  # kept for call-site compatibility
    d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
    model_dir_path = Path(model_dir)
    for candidate in ("model_final.pt", "model.pt", "checkpoint.pt"):
        model_path = model_dir_path / candidate
        if model_path.exists():
            break
    else:
        print(f"Model not found in {model_dir}")
        return None, None, None
    if candidate == "checkpoint.pt":
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    ens = state_dict["W0"].shape[0]
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1 / d, 1 / (N * chi)), device=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, d, P, seed


def copy_snapshot_into_model_(model, A_snaps, W0_snaps, t: int, device):
    """Load snapshot t into model in-place (A_snaps/W0_snaps may live on CPU)."""
    with torch.no_grad():
        model.A.copy_(A_snaps[t].to(device=device, dtype=model.A.dtype))
        model.W0.copy_(W0_snaps[t].to(device=device, dtype=model.W0.dtype))


def output_learnability_streaming_snapshot_avg(
    model,
    A_snaps,
    W0_snaps,
    d,
    total_samples=20_000_000,
    batch_size=20_000,
    device="cuda",
    dtype=torch.float32,
    y_He1=1.0,
    y_He3=1.0,
):
    """Population learnability of the predictive mean f_bar = mean_t f_t(x)."""
    t_snaps = int(A_snaps.shape[0])
    sum_yx0 = 0.0
    sum_yh3 = 0.0
    sum_x0h3 = 0.0
    n_seen = 0
    model.eval()
    with torch.no_grad():
        while n_seen < total_samples:
            cur_bs = min(batch_size, total_samples - n_seen)
            x = torch.randn(cur_bs, d, dtype=dtype, device=device)
            x0 = x[:, 0]
            h3 = (x0 ** 3 - 3.0 * x0) / (6.0 ** 0.5)
            y_pred = None
            for t in range(t_snaps):
                copy_snapshot_into_model_(model, A_snaps, W0_snaps, t, device)
                f_t = collapse_model_prediction(model(x))
                y_pred = f_t if y_pred is None else y_pred + f_t
            y_pred = y_pred / float(t_snaps)
            sum_yx0 += (y_pred * x0).sum().double().item()
            sum_yh3 += (y_pred * h3).sum().double().item()
            sum_x0h3 += (x0 * h3).sum().double().item()
            n_seen += cur_bs
            del x, x0, h3, y_pred
    mean_yx0 = sum_yx0 / n_seen
    mean_yh3 = sum_yh3 / n_seen
    mean_x0h3 = sum_x0h3 / n_seen
    linear_proj = mean_yx0 / y_He1 if abs(float(y_He1)) > 1e-12 else float("nan")
    if abs(float(y_He3)) < 1e-12:
        cubic_proj = float("nan")
    else:
        cubic_proj = (mean_yh3 - mean_yx0 * mean_x0h3) / y_He3
    return {
        "linear": float(linear_proj),
        "cubic": float(cubic_proj),
        "n_samples": n_seen,
        "n_snapshots": t_snaps,
    }


def H_eig_snapshot_average(model, A_snaps, W0_snaps, X, device):
    """Mean_t of H_eig under snapshot weights."""
    t_snaps = int(A_snaps.shape[0])
    acc = None
    with torch.no_grad():
        for t in range(t_snaps):
            copy_snapshot_into_model_(model, A_snaps, W0_snaps, t, device)
            eigs = model.H_eig(X, X)
            acc = eigs if acc is None else acc + eigs
    return acc / float(t_snaps)


def snapshot_weight_targets(W0_snaps):
    """Flatten teacher-coord W0 over all snapshots."""
    return W0_snaps[:, :, :, 0].reshape(-1).detach().cpu().numpy()


def snapshot_A_and_W_target(A_snaps, W0_snaps):
    """Paired A and W_{i,0} after averaging over the snapshot index.

    A_snaps : (T, ens, N), W0_snaps : (T, ens, N, d)
    Returns flattened (ens * N,) arrays of mean_t A and mean_t W_{:,:,0}.
    """
    a = A_snaps.mean(dim=0).reshape(-1).detach().cpu().numpy()
    w = W0_snaps.mean(dim=0)[:, :, 0].reshape(-1).detach().cpu().numpy()
    return w, a


def predictive_mean_on_X(model, A_snaps, W0_snaps, X, device):
    """f_bar(X) = mean_t f_t(X), shape (P,)."""
    t_snaps = int(A_snaps.shape[0])
    with torch.no_grad():
        acc = None
        for t in range(t_snaps):
            copy_snapshot_into_model_(model, A_snaps, W0_snaps, t, device)
            f_t = collapse_model_prediction(model(X))
            acc = f_t if acc is None else acc + f_t
        return acc / float(t_snaps)


def model_param_title(model_dir: str) -> str:
    d, P, N, chi, seed, T, eps, s0 = parse_config_from_dirname(model_dir)
    name = Path(model_dir).name
    lines = [rf"$d={d},\ P={P},\ N={N}$"]
    extras = []
    ba = re.search(r"beta([\d.]+)_alpha([\d.]+)", name)
    if ba:
        extras.append(rf"$\beta={float(ba.group(1)):.3g}$")
        extras.append(rf"$\alpha={float(ba.group(2)):.3g}$")
    kappa = re.search(r"kappa([\d.]+)", name)
    if kappa:
        extras.append(rf"$\kappa={float(kappa.group(1)):.3g}$")
    sa0 = re.search(r"sa0([\d.]+)", name)
    if sa0:
        extras.append(rf"$\sigma_a^{{2}}={float(sa0.group(1)):.3g}$")
    if not extras and T is not None:
        extras.append(rf"$T={T:g}$")
    if seed is not None:
        extras.append(rf"seed={seed}")
    if extras:
        lines.append(", ".join(extras))
    return "\n".join(lines)


def _scatter_readout_vs_hidden(ax, w_vals, a_vals, xlabel, color):
    w_vals = np.asarray(w_vals, dtype=np.float64).reshape(-1)
    a_vals = np.asarray(a_vals, dtype=np.float64).reshape(-1)
    ax.scatter(
        w_vals,
        a_vals,
        s=8,
        alpha=0.32,
        color=color,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.axhline(0.0, color="0.65", lw=0.8, zorder=1)
    ax.axvline(0.0, color="0.65", lw=0.8, zorder=1)
    finite = np.isfinite(w_vals) & np.isfinite(a_vals)
    w = w_vals[finite]
    a = a_vals[finite]
    r = np.nan
    if w.size >= 2 and float(np.std(w)) > 0 and float(np.std(a)) > 0:
        r = float(np.corrcoef(w, a)[0, 1])
        slope, intercept = np.polyfit(w, a, 1)
        xs = np.linspace(float(w.min()), float(w.max()), 200)
        ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.35, zorder=3, label=rf"$r={r:.3f}$")
        ax.legend(frameon=True, fontsize=8, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$A_i$")
    ax.grid(True, alpha=0.3)
    return r


def plot_readout_vs_hidden_weights(model_dirs, output_dir, device, group_name=None):
    """Per-model scatter of readout A_i vs the matching hidden weight W_i.

    When ``A_snapshots/`` exist, both A and W are averaged over the snapshot
    index before pairing units: scatter is ``mean_t A_{t,i}`` vs ``mean_t W_{t,i,0}``.
    """
    n_models = len(model_dirs)
    if n_models == 0:
        return

    n_panels = n_models + 1
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.6 * nrows),
        squeeze=False,
        dpi=200,
    )

    r_records = []
    plotted = -1
    used_snapshots = False
    for model_dir in model_dirs:
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        plotted += 1
        ax = axes[plotted // ncols, plotted % ncols]
        snap_pack = load_a_snapshots(model_dir, device=None)
        if snap_pack is not None:
            A_snaps, W0_snaps, _epochs = snap_pack
            w_target, a_vals = snapshot_A_and_W_target(A_snaps, W0_snaps)
            used_snapshots = True
            del A_snaps, W0_snaps
        else:
            a_vals = model.A.detach().cpu().numpy().reshape(-1)
            w_target = model.W0[:, :, 0].detach().cpu().numpy().reshape(-1)
        xlabel = (
            r"$\bar W_{i,0}$ (hidden, teacher coord.)"
            if snap_pack is not None
            else r"$W_{i,0}$ (hidden, teacher coord.)"
        )
        r = _scatter_readout_vs_hidden(
            ax,
            w_target,
            a_vals,
            xlabel,
            "royalblue",
        )
        if snap_pack is not None:
            ax.set_ylabel(r"$\bar A_i$")
        ax.set_title(model_param_title(model_dir), fontsize=10)
        d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
        beta_m = re.search(r"beta([\d.]+)", Path(model_dir).name, re.IGNORECASE)
        r_records.append(
            {
                "P": P,
                "r": r,
                "d": d,
                "N": N,
                "beta": float(beta_m.group(1)) if beta_m else float("nan"),
            }
        )
        del model

    r_records = [rec for rec in r_records if np.isfinite(rec["r"])]
    if r_records:
        plotted += 1
        ax_scale = axes[plotted // ncols, plotted % ncols]
        p_vals = np.asarray([rec["P"] for rec in r_records], dtype=np.float64)
        r_vals = np.asarray([rec["r"] for rec in r_records], dtype=np.float64)
        beta_vals = np.asarray([rec["beta"] for rec in r_records], dtype=np.float64)
        use_beta = np.all(np.isfinite(beta_vals)) and len(np.unique(p_vals)) == 1
        x_vals = beta_vals if use_beta else p_vals
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        r_vals = r_vals[order]
        r_label = r"$r(\bar A_i, \bar W_{i,0})$" if used_snapshots else r"$r(A_i, W_{i,0})$"
        ax_scale.plot(
            x_vals,
            r_vals,
            linestyle="--",
            marker="o",
            color="crimson",
            markersize=7,
            linewidth=1.6,
            label=r_label,
        )
        ax_scale.set_xlabel(r"$\beta$" if use_beta else r"$P$")
        ax_scale.set_ylabel(r"$r$")
        ax_scale.set_title(r"$r$ vs $\beta$" if use_beta else r"$r$ vs $P$")
        ax_scale.grid(True, alpha=0.3)
        ax_scale.legend(loc="best", fontsize=8)

    for j in range(plotted + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(
        r"Readout $\bar A_i$ vs hidden $\bar W_{i,0}$ (snapshot-averaged, paired by unit)"
        if used_snapshots
        else r"Readout $A_i$ vs hidden $W_{i,0}$ (paired by unit, all ensemble members)",
        fontsize=13,
        y=1.01,
    )
    if group_name:
        annotate_scaling_exponents(fig, group_name)
    else:
        fig.tight_layout()
    out_path = Path(output_dir) / "readout_A_vs_hidden_W.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved readout vs hidden-weight scatter to {out_path}")


def collapse_model_prediction(y_pred_raw):
    if y_pred_raw.ndim == 1:
        return y_pred_raw
    if y_pred_raw.ndim == 2:
        return y_pred_raw.mean(dim=1)
    return y_pred_raw.reshape(y_pred_raw.shape[0], -1).mean(dim=1)


def population_teacher_hermite_coeffs(eps: float) -> tuple[float, float]:
    """Population Hermite coeffs of y = x0 + eps He3 for standard Gaussian x0."""
    return 1.0, float(eps)


def output_learnability_streaming(
    model,
    d,
    total_samples=20_000_000,
    batch_size=200_000,
    device="cuda",
    dtype=torch.float32,
    y_He1=1.0,
    y_He3=1.0,
):
    """Streaming test-set / population linear and cubic learnability.

    Accumulates E[f He1] and E[f He3] on fresh Gaussian samples, then
    residualizes the cubic coefficient. Denominators are the teacher
    Hermite coeffs (population values, not the P-point train estimates).
    """
    sum_yx0 = 0.0
    sum_yh3 = 0.0
    sum_x0h3 = 0.0
    n_seen = 0

    model.eval()
    with torch.no_grad():
        while n_seen < total_samples:
            cur_bs = min(batch_size, total_samples - n_seen)
            x = torch.randn(cur_bs, d, dtype=dtype, device=device)
            x0 = x[:, 0]
            h3 = (x0 ** 3 - 3.0 * x0) / (6.0 ** 0.5)
            y_pred = collapse_model_prediction(model(x))
            sum_yx0 += (y_pred * x0).sum().double().item()
            sum_yh3 += (y_pred * h3).sum().double().item()
            sum_x0h3 += (x0 * h3).sum().double().item()
            n_seen += cur_bs
            del x, x0, h3, y_pred

    mean_yx0 = sum_yx0 / n_seen
    mean_yh3 = sum_yh3 / n_seen
    mean_x0h3 = sum_x0h3 / n_seen
    linear_proj = mean_yx0 / y_He1 if abs(float(y_He1)) > 1e-12 else float("nan")
    if abs(float(y_He3)) < 1e-12:
        cubic_proj = float("nan")
    else:
        cubic_proj = (mean_yh3 - mean_yx0 * mean_x0h3) / y_He3
    return {
        "linear": float(linear_proj),
        "cubic": float(cubic_proj),
        "n_samples": n_seen,
    }


def learnability_from_eigenvalue(eigenvalue, ridge, P):
    if eigenvalue is None or ridge is None or P is None:
        return np.nan
    if not np.isfinite(eigenvalue) or not np.isfinite(ridge) or P <= 0:
        return np.nan
    return float(eigenvalue / (eigenvalue + ridge / P))


def test_learnability_for_model(model, d, eps, device):
    y_He1, y_He3 = population_teacher_hermite_coeffs(eps)
    return output_learnability_streaming(
        model,
        d,
        total_samples=20_000_000,
        batch_size=20_000,
        device=device,
        y_He1=y_He1,
        y_He3=y_He3,
    )


def plot_experiment_group(experiment_group, device, recompute=False, vga_advanced=False):
    model_dirs = experiment_group.model_dirs
    output_dir = OUTPUT_BASE_DIR / slugify(experiment_group.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    computation_cache_path = output_dir / 'computation_cache.pkl'

    print("=" * 42)
    print(f"Experiment group: {experiment_group.name}")
    print(f"  Output dir: {output_dir}")
    print(f"  VGA entropy: {'exact GMM (--vga-advanced)' if vga_advanced else 'Hershey-Olsen (default)'}")
    print("=" * 42)

    if not model_dirs:
        print(f"  No model directories found; skipping.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt
    # Group models by parameter set (d, P, N, chi)
    from collections import defaultdict
    grouped = defaultdict(list)
    param_labels = {}
    for model_dir in model_dirs:
        d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
        key = (d, P, N, chi)
        grouped[key].append((model_dir, seed))
        param_labels[key] = f"d={d}, P={P}, N={N}, chi={chi}"

    param_keys = sorted(grouped.keys(), key=lambda x: (x[1], x[3]))  # sort by P, chi
    ncols = 4
    nrows = (len(param_keys) + ncols - 1) // ncols
    fig_scatter, axes_scatter = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    empirical_curves = []

    for idx, key in enumerate(param_keys):
        ax = axes_scatter[idx//ncols, idx%ncols]
        for i, (model_dir, seed) in enumerate(
            sorted(grouped[key], key=lambda x: (x[1] is None, x[1] if x[1] is not None else -1, x[0]))
        ):
            print(f"Loading model from {model_dir}")
            d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
            model, *_ = load_model(model_dir, device)
            if model is None:
                continue
            # Replay the actual training set (P points, CUDA RNG), not a test draw.
            X, Y_t, *_ = make_training_dataset(P, d, seed, float(epsilon) if epsilon is not None else 0.0, device)
            snap_pack = load_a_snapshots(model_dir, device=None)
            if snap_pack is not None:
                A_snaps, W0_snaps, _epochs = snap_pack
                model_output = predictive_mean_on_X(
                    model, A_snaps, W0_snaps, X, device
                ).detach().cpu().numpy()
                del A_snaps, W0_snaps
            else:
                f_full = model.forward(X).detach().cpu().numpy()  # (P, ens)
                if f_full.ndim == 2:
                    model_output = f_full.mean(axis=1)
                else:
                    model_output = f_full
            Y = Y_t.detach().cpu().numpy()
            color = color_cycle[i % len(color_cycle)]
            ax.scatter(Y, model_output, s=2, alpha=0.7, label=f'seed={seed}', color=color)
        # Add y = x reference line
        min_val = ax.get_xlim()[0]
        max_val = ax.get_xlim()[1]
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='y = x')
        ax.set_xlabel('Y(x) (True Target)')
        ax.set_ylabel('Model Output')
        ax.set_title(param_labels[key])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- New: Per-model h0_activation projection log-action plots ---
    n_models = len(model_dirs)
    ncols_hist = 4
    nrows_hist = (n_models + ncols_hist - 1) // ncols_hist
    fig_hist, axes_hist = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)
    p_summary_entries = []

    def log_density_curve(values, bins=40, min_samples=3, eps=1e-9):
        values_np = np.asarray(values, dtype=np.float64)
        counts_raw, bin_edges = np.histogram(values_np, bins=bins)
        density, _ = np.histogram(values_np, bins=bin_edges, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = counts_raw >= min_samples
        log_density = np.where(mask, -np.log(density + eps), np.nan)
        return centers, log_density

    def symmetric_bimodal_action(x, mu, sigma, eps=1e-12):
        x_np = np.asarray(x, dtype=np.float64)
        mu = float(abs(mu))
        sigma = float(max(sigma, 1e-12))
        density = 0.5 * np.exp(-0.5 * ((x_np - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        density += 0.5 * np.exp(-0.5 * ((x_np + mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        return -np.log(density + eps)

    def fit_bimodal_gaussian_action(values, eps=1e-12):
        values_np = np.asarray(values, dtype=np.float64).reshape(-1)
        if values_np.size < 10:
            return None

        mean = float(values_np.mean())
        std = float(values_np.std())
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0

        median = float(np.median(values_np))
        lower = values_np[values_np <= median]
        upper = values_np[values_np > median]
        mu1_init = float(lower.mean()) if lower.size else mean - 0.5 * std
        mu2_init = float(upper.mean()) if upper.size else mean + 0.5 * std
        sigma_init = max(std / 2.0, 1e-3)
        counts_raw, bin_edges = np.histogram(values_np, bins=40)
        empirical = counts_raw.astype(np.float64)
        empirical_sum = float(empirical.sum())
        if empirical_sum <= 0.0:
            return None
        empirical = empirical / empirical_sum

        def normal_cdf(x, mu, sigma):
            sigma = np.maximum(sigma, 1e-12)
            return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

        def model_bin_probs(pi, mu1, mu2, sigma1, sigma2):
            left = bin_edges[:-1]
            right = bin_edges[1:]
            probs = (
                pi * (normal_cdf(right, mu1, sigma1) - normal_cdf(left, mu1, sigma1))
                + (1.0 - pi) * (normal_cdf(right, mu2, sigma2) - normal_cdf(left, mu2, sigma2))
            )
            probs = np.clip(probs, eps, None)
            return probs / probs.sum()

        def unpack(params):
            logit_pi, mu1, mu2, log_sigma1, log_sigma2 = params
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            sigma1 = np.exp(log_sigma1)
            sigma2 = np.exp(log_sigma2)
            return pi, mu1, mu2, sigma1, sigma2

        def objective(params):
            pi, mu1, mu2, sigma1, sigma2 = unpack(params)
            model_probs = model_bin_probs(pi, mu1, mu2, sigma1, sigma2)
            return float(np.sum(empirical * (np.log(empirical + eps) - np.log(model_probs + eps))))

        initial_params = np.array([
            0.0,
            mu1_init,
            mu2_init,
            np.log(sigma_init),
            np.log(sigma_init),
        ], dtype=np.float64)

        result = minimize(objective, initial_params, method='L-BFGS-B')
        if not result.success:
            return None

        pi, mu1, mu2, sigma1, sigma2 = unpack(result.x)
        if mu1 > mu2:
            pi = 1.0 - pi
            mu1, mu2 = mu2, mu1
            sigma1, sigma2 = sigma2, sigma1

        return {
            'pi': float(pi),
            'mu1': float(mu1),
            'sigma1': float(sigma1),
            'mu2': float(mu2),
            'sigma2': float(sigma2),
            'success': bool(result.success),
            'kl': float(result.fun),
        }

    def bimodal_gaussian_action(x, fit_params):
        x_np = np.asarray(x, dtype=np.float64)
        pi = fit_params['pi']
        mu1 = fit_params['mu1']
        sigma1 = fit_params['sigma1']
        mu2 = fit_params['mu2']
        sigma2 = fit_params['sigma2']
        density = (
            pi * np.exp(-0.5 * ((x_np - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
            + (1.0 - pi) * np.exp(-0.5 * ((x_np - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2.0 * np.pi))
        )
        return -np.log(density + 1e-12)

    def output_learnability_on_train(y_pred, x0, h3, y_He1, y_He3):
        """Linear / cubic learnability of model outputs on the training sample.

        Same residualization as the streaming estimator,
            (<f, He3> - <f, He1> <He1, He3>) / <y, He3>
        but with empirical inner products on the P training points only.
        """
        mean_yx0 = (y_pred * x0).mean()
        mean_yh3 = (y_pred * h3).mean()
        mean_x0h3 = (x0 * h3).mean()
        linear_proj = mean_yx0 / y_He1 if abs(float(y_He1)) > 1e-12 else float("nan")
        if abs(float(y_He3)) < 1e-12:
            cubic_proj = float("nan")
        else:
            cubic_proj = (mean_yh3 - mean_yx0 * mean_x0h3) / y_He3
        return {
            "linear": float(linear_proj),
            "cubic": float(cubic_proj),
            "n_samples": int(y_pred.numel()),
        }

    def h0_activation_projections_streaming(
        model,
        d,
        total_samples=10_000_000,
        batch_size=20_000,
        device='cuda',
        dtype=torch.float32,
    ):
        """
        Streaming estimate of first-layer activation projections onto the
        linear (x0) and cubic (x0^3 - 3 x0) directions, without materializing
        more than one batch at a time.

        Accumulates
            proj = E_x[ h0(x) * phi(x0) ]
        as a running sum over Gaussian samples, returning tensors of shape
        (ens, n1) suitable for action histograms / variance estimates.
        """
        ens = model.ens
        n1 = model.n1
        sum_h3 = torch.zeros(ens, n1, dtype=torch.float64, device=device)
        sum_lin = torch.zeros(ens, n1, dtype=torch.float64, device=device)
        n_seen = 0

        model.eval()
        with torch.no_grad():
            while n_seen < total_samples:
                cur_bs = min(batch_size, total_samples - n_seen)

                x = torch.randn(cur_bs, d, dtype=dtype, device=device)
                h0 = model.h0_activation(x)
                x0 = x[:, 0]
                h3 = (x0**3 - 3.0 * x0) / (6.0**0.5)
                lin = x0

                sum_h3 += torch.einsum('pqn,p->qn', h0, h3).double()
                sum_lin += torch.einsum('pqn,p->qn', h0, lin).double()
                n_seen += cur_bs

                del x, h0, x0, h3, lin

        proj_h3 = (sum_h3 / n_seen).to(dtype=dtype)
        proj_lin = (sum_lin / n_seen).to(dtype=dtype)
        return {
            "hermite3": proj_h3,
            "linear": proj_lin,
            "n_samples": n_seen,
        }

    vga_cache = {}
    kappa_eff_cache = {}

    def bare_kappa_from_config(T, chi):
        if T is not None:
            return float(T / 2.0)
        if chi not in (None, 0):
            return float(1.0 / chi)
        return np.nan

    def get_kappa_eff_for_params(d, P, kappa_bare, N, chi):
        if not np.isfinite(kappa_bare):
            return kappa_bare
        key = (int(d), int(P), float(kappa_bare), int(N), float(chi))
        if key in kappa_eff_cache:
            return kappa_eff_cache[key]
        try:
            kappa_eff = float(
                compute_kappa_eff(
                    d=int(d),
                    P=int(P),
                    kappa_bare=float(kappa_bare),
                    n1=int(N),
                    chi=float(chi),
                    device=device,
                    verbose=False,
                )
            )
            print(f"kappa_bare={kappa_bare:.6g} -> kappa_eff={kappa_eff:.6g} (d={d}, P={P})")
        except Exception as exc:
            print(f"kappa_eff computation failed ({exc}); using bare kappa={kappa_bare}")
            kappa_eff = float(kappa_bare)
        kappa_eff_cache[key] = kappa_eff
        return kappa_eff

    def get_vga_for_model(model_dir, d, P, N, chi, T, epsilon, s0):
        if model_dir in vga_cache:
            return vga_cache[model_dir]
        kappa_bare = bare_kappa_from_config(T, chi)
        kappa = get_kappa_eff_for_params(d, P, kappa_bare, N, chi)
        print(f"Using kappa_eff for VGA: {kappa} (bare={kappa_bare}, advanced={vga_advanced})")
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "fcn2_vga_erf.jl"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cmd = [
                "julia",
                str(julia_script),
                "--d", str(d),
                "--n1", str(N),
                "--P", str(P),
                "--chi", str(chi),
                "--kappa", str(kappa),
                "--epsilon", str(epsilon if epsilon is not None else 0.03),
                "--s0", str(s0 if s0 is not None else 1.0),
                "--to", str(tmp_path),
                "--quiet",
            ]
            if vga_advanced:
                cmd.append("--advanced")
            subprocess.run(cmd, check=True, capture_output=True)
            with open(tmp_path, "r") as f:
                vga_cache[model_dir] = json.load(f)
        except Exception as exc:
            print(f"Could not compute VGA theory for {model_dir}: {exc}")
            vga_cache[model_dir] = None
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return vga_cache[model_dir]

    cached = None
    cache_version = 9  # empirical quantities averaged over A_snapshots when present
    refresh_learnability_only = False
    if not recompute and computation_cache_path.exists():
        try:
            with open(computation_cache_path, 'rb') as f:
                cached = pickle.load(f)
            cached_version = cached.get('version')
            if cached_version == cache_version:
                if bool(cached.get('vga_advanced', False)) != bool(vga_advanced):
                    print("Computation cache VGA entropy mode mismatch; recomputing projections.")
                    cached = None
                else:
                    print(f"Loaded computation cache from {computation_cache_path}")
            elif cached_version in (5, 6):
                print(
                    f"Cache v{cached_version}: reusing h0 projections; "
                    "refreshing output learnability on a Gaussian test set."
                )
                refresh_learnability_only = True
            else:
                print("Computation cache version mismatch; recomputing projections.")
                cached = None
        except Exception as exc:
            print(f"Could not load computation cache ({exc}); recomputing projections.")
            cached = None

    def compute_one_model(model_dir):
        d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            return None

        snap_pack = load_a_snapshots(model_dir, device=None)
        if snap_pack is not None:
            A_snaps, W0_snaps, snap_epochs = snap_pack
            t_snaps = int(A_snaps.shape[0])
            print(
                f"  A_snapshots: averaging T={t_snaps} Langevin snapshots "
                f"(epochs {snap_epochs[0]}..{snap_epochs[-1]}) for empirical quantities"
            )
        else:
            A_snaps = W0_snaps = None
            t_snaps = 0

        eps = float(epsilon) if epsilon is not None else 0.0
        y_He1, y_He3 = population_teacher_hermite_coeffs(eps)
        if A_snaps is not None:
            learnability_outputs = output_learnability_streaming_snapshot_avg(
                model,
                A_snaps,
                W0_snaps,
                d,
                total_samples=20_000_000,
                batch_size=20_000,
                device=device,
                y_He1=y_He1,
                y_He3=y_He3,
            )
            learnability_source = "test_streaming_asnap_avg"
            n_snap_ens = t_snaps
        else:
            learnability_outputs = test_learnability_for_model(model, d, eps, device)
            learnability_source = "test_streaming"
            n_snap_ens = int(model.ens)
        print(
            "Test-set learnability  linear=",
            learnability_outputs["linear"],
            " cubic=",
            learnability_outputs["cubic"],
            " n=",
            learnability_outputs["n_samples"],
            f" snapshots={n_snap_ens}",
        )

        if A_snaps is not None:
            samples_per = max(20_000, int(np.ceil(10_000_000 / t_snaps)))
            proj_h3_parts = []
            proj_lin_parts = []
            for t in range(t_snaps):
                copy_snapshot_into_model_(model, A_snaps, W0_snaps, t, device)
                h0_projections = h0_activation_projections_streaming(
                    model,
                    d,
                    total_samples=samples_per,
                    batch_size=20_000,
                    device=device,
                    dtype=torch.float32,
                )
                proj_h3_parts.append(h0_projections["hermite3"].detach().flatten().cpu().numpy())
                proj_lin_parts.append(h0_projections["linear"].detach().flatten().cpu().numpy())
            proj_h3 = np.concatenate(proj_h3_parts, axis=0)
            proj_lin = np.concatenate(proj_lin_parts, axis=0)
        else:
            h0_projections = h0_activation_projections_streaming(
                model, d,
                total_samples=10_000_000,
                batch_size=20_000,
                device=device,
                dtype=torch.float32,
            )
            proj_h3 = h0_projections["hermite3"].detach().flatten().cpu().numpy()
            proj_lin = h0_projections["linear"].detach().flatten().cpu().numpy()
        var_h3 = float(np.var(proj_h3))
        var_lin = float(np.var(proj_lin))

        n_rayleigh = 5000
        X_ray = torch.randn(n_rayleigh, d, dtype=torch.float32, device=device)
        if A_snaps is not None:
            linear_eigs = H_eig_snapshot_average(model, A_snaps, W0_snaps, X_ray, device)
        else:
            with torch.no_grad():
                linear_eigs = model.H_eig(X_ray, X_ray)
        linear_eig_target = float(linear_eigs[0].item())
        linear_eig_perp = float(linear_eigs[1:].mean().item()) if d > 1 else np.nan
        print(
            f"H_eig Rayleigh (n={n_rayleigh}): "
            f"lJ1_target={linear_eig_target:.6g}, lJ1_perp_mean={linear_eig_perp:.6g}"
        )
        del X_ray, linear_eigs

        vga_theory = get_vga_for_model(model_dir, d, P, N, chi, T, epsilon, s0)
        target_vga = vga_theory.get("vga", {}).get("target", {}) if vga_theory is not None else {}
        lJ1T = target_vga.get("lJ1")
        lJ3T = target_vga.get("lJ3")

        hermite3_curve = log_density_curve(proj_h3, bins=40, min_samples=3)
        linear_curve = log_density_curve(proj_lin, bins=40, min_samples=3)
        h0_entry = {
            "model_dir": model_dir,
            "model_name": Path(model_dir).name,
            "P": P,
            "proj_h3": proj_h3,
            "proj_lin": proj_lin,
            "var_h3": var_h3,
            "var_lin": var_lin,
            "linear_eig_target": linear_eig_target,
            "linear_eig_perp": linear_eig_perp,
            "lJ1T": lJ1T,
            "lJ3T": lJ3T,
            "hermite3_curve": hermite3_curve,
            "linear_curve": linear_curve,
            "n_snapshots": n_snap_ens if A_snaps is not None else 0,
        }
        empirical_entry = {
            "P": P,
            "model_dir": model_dir,
            "model_name": Path(model_dir).name,
            "hermite3": hermite3_curve,
            "linear": linear_curve,
            "lJ1T": lJ1T,
            "lJ3T": lJ3T,
            "n_snapshots": n_snap_ens if A_snaps is not None else 0,
        }
        kappa_bare = bare_kappa_from_config(T, chi)
        kappa_eff = get_kappa_eff_for_params(d, P, kappa_bare, N, chi)
        p_summary_entry = {
            "P": P,
            "model_dir": model_dir,
            "model_name": Path(model_dir).name,
            "kappa_bare": kappa_bare,
            "kappa": kappa_eff,
            "kappa_eff": kappa_eff,
            "linear_empirical": linear_eig_target,
            "linear_empirical_perp": linear_eig_perp,
            "linear_projection_var": var_lin,
            "linear_theory": float(lJ1T) if lJ1T is not None and lJ1T > 0 else np.nan,
            "cubic_empirical": var_h3,
            "cubic_theory": float(lJ3T) if lJ3T is not None and lJ3T > 0 else np.nan,
            "linear_learnability_empirical": learnability_outputs["linear"],
            "cubic_learnability_empirical": learnability_outputs["cubic"],
            "learnability_n_samples": learnability_outputs["n_samples"],
            "learnability_source": learnability_source,
            "ensemble_size": int(n_snap_ens),
            "n_snapshots": int(n_snap_ens) if A_snaps is not None else 0,
        }
        del model
        if snap_pack is not None:
            del A_snaps, W0_snaps
        return h0_entry, empirical_entry, p_summary_entry

    # Index any reusable cached entries by model_dir.
    cached_h0_by_dir = {}
    cached_emp_by_dir = {}
    cached_summary_by_dir = {}
    if cached is not None and not recompute:
        for entry in cached.get('h0_plot_data', []):
            key = entry.get('model_dir')
            if key:
                cached_h0_by_dir[key] = entry
        for entry in cached.get('empirical_curves', []):
            key = entry.get('model_dir') or next(
                (
                    h0['model_dir']
                    for h0 in cached.get('h0_plot_data', [])
                    if h0.get('model_name') == entry.get('model_name')
                ),
                None,
            )
            if key:
                entry = dict(entry)
                entry['model_dir'] = key
                cached_emp_by_dir[key] = entry
        for entry in cached.get('p_summary_entries', []):
            key = entry.get('model_dir')
            if key:
                cached_summary_by_dir[key] = entry

    empirical_curves = []
    p_summary_entries = []
    h0_plot_data = []
    cache_dirty = recompute or cached is None or refresh_learnability_only
    for model_dir in model_dirs:
        reuse = (
            not recompute
            and model_dir in cached_h0_by_dir
            and model_dir in cached_emp_by_dir
            and model_dir in cached_summary_by_dir
        )
        if reuse:
            print(f"Reusing cached projections for {model_dir}")
            h0_plot_data.append(cached_h0_by_dir[model_dir])
            empirical_curves.append(cached_emp_by_dir[model_dir])
            summary_entry = dict(cached_summary_by_dir[model_dir])
            if refresh_learnability_only:
                d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
                model, *_ = load_model(model_dir, device)
                if model is not None:
                    eps = float(epsilon) if epsilon is not None else 0.0
                    y_He1, y_He3 = population_teacher_hermite_coeffs(eps)
                    snap_pack = load_a_snapshots(model_dir, device=None)
                    if snap_pack is not None:
                        A_snaps, W0_snaps, _epochs = snap_pack
                        learnability_outputs = output_learnability_streaming_snapshot_avg(
                            model, A_snaps, W0_snaps, d,
                            total_samples=20_000_000, batch_size=20_000, device=device,
                            y_He1=y_He1, y_He3=y_He3,
                        )
                        summary_entry["learnability_source"] = "test_streaming_asnap_avg"
                        summary_entry["ensemble_size"] = int(A_snaps.shape[0])
                        summary_entry["n_snapshots"] = int(A_snaps.shape[0])
                        del A_snaps, W0_snaps
                    else:
                        learnability_outputs = test_learnability_for_model(model, d, eps, device)
                        summary_entry["learnability_source"] = "test_streaming"
                    summary_entry["linear_learnability_empirical"] = learnability_outputs["linear"]
                    summary_entry["cubic_learnability_empirical"] = learnability_outputs["cubic"]
                    summary_entry["learnability_n_samples"] = learnability_outputs["n_samples"]
                    print(
                        f"  Test-set learnability P={P}: linear={learnability_outputs['linear']:.4f} "
                        f"cubic={learnability_outputs['cubic']:.4f} n={learnability_outputs['n_samples']}"
                    )
                    del model
                cache_dirty = True
            p_summary_entries.append(summary_entry)
            continue

        print(f"Computing projections for {model_dir}")
        result = compute_one_model(model_dir)
        if result is None:
            continue
        h0_entry, empirical_entry, p_summary_entry = result
        h0_plot_data.append(h0_entry)
        empirical_curves.append(empirical_entry)
        p_summary_entries.append(p_summary_entry)
        cache_dirty = True

    if cache_dirty:
        with open(computation_cache_path, 'wb') as f:
            pickle.dump(
                {
                    "version": cache_version,
                    "vga_advanced": bool(vga_advanced),
                    "model_dirs": list(model_dirs),
                    "h0_plot_data": h0_plot_data,
                    "empirical_curves": empirical_curves,
                    "p_summary_entries": p_summary_entries,
                },
                f,
            )
        print(f"Saved computation cache to {computation_cache_path}")
    else:
        print(f"Computation cache up to date at {computation_cache_path}")

    # Always regenerate h0 projection histogram plots from cached/computed data
    for idx, entry in enumerate(h0_plot_data):
        axh = axes_hist[idx // ncols_hist, idx % ncols_hist]
        for v, label, color, var in zip(
            [entry["proj_h3"], entry["proj_lin"]],
            ['Hermite3', 'Linear'],
            ['royalblue', 'orange'],
            [entry["var_h3"], entry["var_lin"]],
        ):
            centers, log_density = log_density_curve(v, bins=40, min_samples=3)
            mask = np.isfinite(log_density)
            axh.plot(centers[mask], log_density[mask], label=rf'{label} ($\sigma^2={var:.3g}$)', color=color, linewidth=1.2, marker='x', ms=4)

        lJ1T = entry.get("lJ1T")
        lJ3T = entry.get("lJ3T")
        lin_centers, lin_action = entry["linear_curve"]
        lin_mask = np.isfinite(lin_action)
        if lJ1T is not None and lJ1T > 0 and np.any(lin_mask):
            x_min = float(lin_centers[lin_mask].min())
            x_max = float(lin_centers[lin_mask].max())
            x_theory = np.linspace(x_min, x_max, 1000)
            gaussian_action = 0.5 * x_theory**2 / float(lJ1T) + 0.5 * np.log(2.0 * np.pi * float(lJ1T))
            axh.plot(x_theory, gaussian_action, '--', color='black', linewidth=1.4, label=rf'VGA theory ($lJ1T={float(lJ1T):.3g}$)')

        h3_centers, h3_action = entry["hermite3_curve"]
        h3_mask = np.isfinite(h3_action)
        if lJ3T is not None and lJ3T > 0 and np.any(h3_mask):
            x_min = float(h3_centers[h3_mask].min())
            x_max = float(h3_centers[h3_mask].max())
            x_theory = np.linspace(x_min, x_max, 1000)
            gaussian_action = 0.5 * x_theory**2 / float(lJ3T) + 0.5 * np.log(2.0 * np.pi * float(lJ3T))
            axh.plot(x_theory, gaussian_action, '--', color='forestgreen', linewidth=1.4, label=rf'VGA theory ($lJ3T={float(lJ3T):.3g}$)')

        axh.set_title(entry["model_name"])
        axh.set_xlabel('Projection value')
        axh.set_ylabel('Action: -log P')
        axh.legend()
        axh.grid(True, alpha=0.3)

    idx = max(len(h0_plot_data) - 1, 0)
    for j in range(len(h0_plot_data), nrows_hist * ncols_hist):
        axes_hist[j // ncols_hist, j % ncols_hist].axis('off')
    fig_hist.tight_layout()
    fig_hist.savefig(output_dir / 'h0_activation_projection_histograms.png', dpi=150)
    plt.close(fig_hist)
    print(f'Saved h0_activation projection histograms to {output_dir / "h0_activation_projection_histograms.png"}')

    if empirical_curves:
        fig_empirical = plt.figure(figsize=(11, 6), dpi=300)
        gs = fig_empirical.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)
        ax_hermite = fig_empirical.add_subplot(gs[0, 0])
        ax_linear = fig_empirical.add_subplot(gs[0, 1], sharey=ax_hermite)
        cax = fig_empirical.add_subplot(gs[0, 2])
        p_values = [entry["P"] for entry in empirical_curves]
        p_min = min(p_values)
        p_max = max(p_values)
        norm = Normalize(vmin=p_min, vmax=p_max)
        cmap = plt.cm.viridis

        for entry in empirical_curves:
            p_val = entry["P"]
            color = cmap(norm(p_val))
            centers_h, log_density_h = entry["hermite3"]
            mask_h = np.isfinite(log_density_h)
            ax_hermite.plot(
                centers_h[mask_h],
                log_density_h[mask_h],
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )

            centers_l, log_density_l = entry["linear"]
            mask_l = np.isfinite(log_density_l)
            ax_linear.plot(
                centers_l[mask_l],
                log_density_l[mask_l],
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )

        for axis, title in ((ax_hermite, "Hermite3"), (ax_linear, "Linear")):
            axis.set_title(title, fontsize=15, pad=10)
            axis.set_xlabel("Projection value", fontsize=14)
            axis.tick_params(axis='both', labelsize=12)
            axis.grid(True, which='major', color='0.85', alpha=0.45, linewidth=0.6)
            axis.grid(True, which='minor', color='0.92', alpha=0.30, linewidth=0.4)
            axis.minorticks_on()
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

        ax_hermite.set_ylabel(r"Action: $-\log \mathcal{P}$", fontsize=14)
        ax_linear.set_ylabel("")
        ax_linear.tick_params(labelleft=False)

        fig_empirical.suptitle("Empirical first-layer activation actions", fontsize=16, y=0.97)

        p_scale_handle = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        p_scale_handle.set_array([])
        p_scale_colorbar = fig_empirical.colorbar(p_scale_handle, cax=cax)
        p_scale_colorbar.set_label(r"$P_{\mathrm{scale}}$", fontsize=13)
        p_scale_colorbar.ax.tick_params(labelsize=12)

        fig_empirical.savefig(output_dir / 'empirical_h0_activation_actions_hermite3_linear.pdf', bbox_inches='tight')
        plt.close(fig_empirical)
        print(f'Saved empirical combined action plot to {output_dir / "empirical_h0_activation_actions_hermite3_linear.pdf"}')

        fig_empirical_theory = plt.figure(figsize=(11, 6), dpi=300)
        gs_theory = fig_empirical_theory.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)
        ax_hermite_theory = fig_empirical_theory.add_subplot(gs_theory[0, 0])
        ax_linear_theory = fig_empirical_theory.add_subplot(gs_theory[0, 1], sharey=ax_hermite_theory)
        cax_theory = fig_empirical_theory.add_subplot(gs_theory[0, 2])

        for entry in empirical_curves:
            p_val = entry["P"]
            color = cmap(norm(p_val))

            centers_h, log_density_h = entry["hermite3"]
            mask_h = np.isfinite(log_density_h)
            ax_hermite_theory.plot(
                centers_h[mask_h],
                log_density_h[mask_h],
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )
            lJ3T_entry = entry.get("lJ3T")
            if lJ3T_entry is not None and lJ3T_entry > 0 and np.any(mask_h):
                x_min = float(centers_h[mask_h].min())
                x_max = float(centers_h[mask_h].max())
                x_theory = np.linspace(x_min, x_max, 1000)
                gaussian_action = 0.5 * x_theory**2 / float(lJ3T_entry) + 0.5 * np.log(2.0 * np.pi * float(lJ3T_entry))
                ax_hermite_theory.plot(
                    x_theory,
                    gaussian_action,
                    linestyle='--',
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                )

            centers_l, log_density_l = entry["linear"]
            mask_l = np.isfinite(log_density_l)
            ax_linear_theory.plot(
                centers_l[mask_l],
                log_density_l[mask_l],
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )
            lJ1T_entry = entry.get("lJ1T")
            if lJ1T_entry is not None and lJ1T_entry > 0 and np.any(mask_l):
                x_min = float(centers_l[mask_l].min())
                x_max = float(centers_l[mask_l].max())
                x_theory = np.linspace(x_min, x_max, 1000)
                gaussian_action = 0.5 * x_theory**2 / float(lJ1T_entry) + 0.5 * np.log(2.0 * np.pi * float(lJ1T_entry))
                ax_linear_theory.plot(
                    x_theory,
                    gaussian_action,
                    linestyle='--',
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                )

        ax_hermite_theory.set_title(f"Empirical + theory H0 actions | {experiment_group.name}")
        ax_hermite_theory.set_xlabel("Projection value")
        ax_hermite_theory.set_ylabel("Action: -log P")
        ax_hermite_theory.grid(True, alpha=0.3)
        ax_linear_theory.set_xlabel("Projection value")
        ax_linear_theory.grid(True, alpha=0.3)

        p_scale_handle_theory = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        p_scale_handle_theory.set_array([])
        p_scale_colorbar_theory = fig_empirical_theory.colorbar(p_scale_handle_theory, cax=cax_theory)
        p_scale_colorbar_theory.set_label(r"$P_{\mathrm{scale}}$", fontsize=13)
        p_scale_colorbar_theory.ax.tick_params(labelsize=12)

        fig_empirical_theory.savefig(output_dir / 'empirical_h0_activation_actions_hermite3_linear_theory_overlay.pdf', bbox_inches='tight')
        plt.close(fig_empirical_theory)
        print(f'Saved empirical H0 theory overlay plot to {output_dir / "empirical_h0_activation_actions_hermite3_linear_theory_overlay.pdf"}')

    # --- New: Per-model first-layer target weight action plots ---
    fig_weights, axes_weights = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)
    fig_wsq, axes_wsq = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)
    empirical_weight_curves = []

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue

        snap_pack = load_a_snapshots(model_dir, device=None)
        if snap_pack is not None:
            _A_snaps, W0_snaps, _epochs = snap_pack
            weights_target = snapshot_weight_targets(W0_snaps)
            weights_perp = W0_snaps[:, :, :, 1:].reshape(-1).detach().cpu().numpy()
            del _A_snaps, W0_snaps
        else:
            weights_target = model.W0[:, :, 0].detach().cpu().numpy().reshape(-1)
            weights_perp = model.W0[:, :, 1:].detach().cpu().numpy().reshape(-1)
        counts_raw, bin_edges = np.histogram(weights_target, bins=40)
        density, _ = np.histogram(weights_target, bins=bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = counts_raw >= 3
        action = np.where(mask, -np.log(density + 1e-9), np.nan)
        var_w0t = float(np.var(weights_target))

        vga_theory = get_vga_for_model(model_dir, d, P, N, chi, T, epsilon, s0)
        vga_target = vga_theory.get("vga", {}).get("target") if vga_theory is not None else None
        lWT = vga_target.get("lWT", None) if vga_target is not None else None

        axw = axes_weights[idx//ncols_hist, idx%ncols_hist]
        axw.plot(bin_centers[mask], action[mask], color='royalblue', linewidth=1.2, marker='x', ms=4, label=rf'$W_{{:,:,0}}$ action ($\sigma^2={var_w0t:.3g}$)')

        print("lWT:", lWT)
        if lWT is not None and lWT > 0:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_theory = np.linspace(x_min, x_max, 1000)
            gaussian_action = 0.5 * x_theory**2 / float(lWT) + 0.5 * np.log(2.0 * np.pi * float(lWT))
            axw.plot(x_theory, gaussian_action, '--', color='black', linewidth=1.4, label=rf'Gaussian theory ($lWT={float(lWT):.3g}$)')

        if vga_target is not None:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_fit = np.linspace(x_min, x_max, 1000)
            action_fit = symmetric_bimodal_action(x_fit, vga_target.get("muW", 0.0), vga_target.get("sigS", 1.0))
            axw.plot(
                x_fit,
                action_fit,
                color='crimson',
                linewidth=1.8,
                label=(
                    rf"VGA bimodal $\mu={float(abs(vga_target.get('muW', 0.0))):.3g}$, "
                    rf"$\sigma={float(vga_target.get('sigS', 1.0)):.3g}$"
                ),
            )
            print(
                f"VGA bimodal parameters for {Path(model_dir).name}: "
                f"muW={float(vga_target.get('muW', 0.0)):.6g}, sigS={float(vga_target.get('sigS', 1.0)):.6g}, "
                f"lWT={float(vga_target.get('lWT', float('nan'))):.6g}"
            )

        bimodal_fit = fit_bimodal_gaussian_action(weights_target)
        if bimodal_fit is not None:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_fit = np.linspace(x_min, x_max, 1000)
            action_fit = bimodal_gaussian_action(x_fit, bimodal_fit)
            axw.plot(
                x_fit,
                action_fit,
                color='magenta',
                linewidth=1.2,
                linestyle='--',
                label=(
                    rf"Empirical fit $\pi={bimodal_fit['pi']:.3g}$, "
                    rf"$\mu_1={bimodal_fit['mu1']:.3g}$, $\sigma_1={bimodal_fit['sigma1']:.3g}$, "
                    rf"$\mu_2={bimodal_fit['mu2']:.3g}$, $\sigma_2={bimodal_fit['sigma2']:.3g}$"
                ),
            )
            print(
                f"Bimodal W1 fit for {Path(model_dir).name}: "
                f"pi={bimodal_fit['pi']:.6g}, mu1={bimodal_fit['mu1']:.6g}, sigma1={bimodal_fit['sigma1']:.6g}, "
                f"mu2={bimodal_fit['mu2']:.6g}, sigma2={bimodal_fit['sigma2']:.6g}, kl={bimodal_fit['kl']:.6g}"
            )

        axw.set_title(Path(model_dir).name, fontsize=26)
        axw.set_xlabel('Target weight value', fontsize=22)
        axw.set_ylabel('Action: -log P', fontsize=22)
        axw.tick_params(axis='both', labelsize=18)
        axw.legend(fontsize=18)
        axw.grid(True, alpha=0.3)

        empirical_weight_curves.append(
            {
                "P": P,
                "model_name": Path(model_dir).name,
                "target": (bin_centers, action),
                "lWT": lWT,
                "vga_target": vga_target,
            }
        )
        for entry in p_summary_entries:
            if entry.get("model_dir") == model_dir:
                entry["w_empirical"] = var_w0t
                entry["w_theory"] = float(lWT) if lWT is not None and lWT > 0 else np.nan
                break

        axsq = axes_wsq[idx//ncols_hist, idx%ncols_hist]
        weights_target_sq = weights_target**2
        weights_perp_sq = weights_perp**2
        axsq.hist(weights_target_sq, bins=40, density=True, color='royalblue', alpha=0.55, label=r'Target $W_{:, :, 0}^2$')
        axsq.hist(weights_perp_sq, bins=40, density=True, color='seagreen', alpha=0.45, label=r'Perp $W_{:, :, 1:}^2$')
        axsq.set_title(Path(model_dir).name, fontsize=26)
        axsq.set_xlabel('Squared weight value', fontsize=22)
        axsq.set_ylabel('Density', fontsize=22)
        axsq.tick_params(axis='both', labelsize=18)
        axsq.legend(fontsize=18)
        axsq.grid(True, alpha=0.3)
        del model

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_weights[j//ncols_hist, j%ncols_hist].axis('off')
    fig_weights.tight_layout()
    fig_weights.savefig(output_dir / 'weight_action_target_histograms.png', dpi=150)
    plt.close(fig_weights)
    print(f'Saved target weight action plots to {output_dir / "weight_action_target_histograms.png"}')

    if empirical_weight_curves:
        fig_weight_empirical, ax_weight_empirical = plt.subplots(figsize=(10, 7))
        p_values = [entry["P"] for entry in empirical_weight_curves]
        p_min = min(p_values)
        p_max = max(p_values)
        norm = Normalize(vmin=p_min, vmax=p_max)
        cmap = plt.cm.viridis

        for entry in empirical_weight_curves:
            p_val = entry["P"]
            centers, action = entry["target"]
            mask = np.isfinite(action)
            ax_weight_empirical.plot(
                centers[mask],
                action[mask],
                color=cmap(norm(p_val)),
                linewidth=1.6,
                alpha=0.95,
            )

        ax_weight_empirical.set_title(f"Empirical target weight actions | {experiment_group.name}")
        ax_weight_empirical.set_xlabel("Target weight value")
        ax_weight_empirical.set_ylabel("Action: -log P")
        ax_weight_empirical.grid(True, alpha=0.3)

        p_color_handle = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        p_color_handle.set_array([])
        colorbar = fig_weight_empirical.colorbar(p_color_handle, ax=ax_weight_empirical)
        colorbar.set_label("P")

        fig_weight_empirical.tight_layout()
        fig_weight_empirical.savefig(output_dir / 'empirical_weight_action_target_histograms.png', dpi=150)
        plt.close(fig_weight_empirical)
        print(f'Saved empirical target weight action plot to {output_dir / "empirical_weight_action_target_histograms.png"}')

        fig_weight_empirical_theory, ax_weight_empirical_theory = plt.subplots(figsize=(10, 7))
        for entry in empirical_weight_curves:
            p_val = entry["P"]
            centers, action = entry["target"]
            mask = np.isfinite(action)
            color = cmap(norm(p_val))
            ax_weight_empirical_theory.plot(
                centers[mask],
                action[mask],
                color=color,
                linewidth=1.6,
                alpha=0.95,
            )

            lWT_entry = entry.get("lWT")
            if lWT_entry is not None and lWT_entry > 0 and np.any(mask):
                x_min = float(centers[mask].min())
                x_max = float(centers[mask].max())
                x_theory = np.linspace(x_min, x_max, 1000)
                gaussian_action = 0.5 * x_theory**2 / float(lWT_entry) + 0.5 * np.log(2.0 * np.pi * float(lWT_entry))
                ax_weight_empirical_theory.plot(
                    x_theory,
                    gaussian_action,
                    linestyle='--',
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                )

            vga_target_entry = entry.get("vga_target")
            if vga_target_entry is not None:
                x_min = float(centers[mask].min()) if np.any(mask) else float(centers[0])
                x_max = float(centers[mask].max()) if np.any(mask) else float(centers[-1])
                x_fit = np.linspace(x_min, x_max, 1000)
                action_fit = symmetric_bimodal_action(x_fit, vga_target_entry.get("muW", 0.0), vga_target_entry.get("sigS", 1.0))
                ax_weight_empirical_theory.plot(
                    x_fit,
                    action_fit,
                    color=color,
                    linewidth=1.1,
                    linestyle=':',
                    alpha=0.95,
                )

        ax_weight_empirical_theory.set_title(f"Empirical + theory target weight actions | {experiment_group.name}")
        ax_weight_empirical_theory.set_xlabel("Target weight value")
        ax_weight_empirical_theory.set_ylabel("Action: -log P")
        ax_weight_empirical_theory.grid(True, alpha=0.3)

        p_color_handle_theory = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        p_color_handle_theory.set_array([])
        colorbar_theory = fig_weight_empirical_theory.colorbar(p_color_handle_theory, ax=ax_weight_empirical_theory)
        colorbar_theory.set_label("P")

        fig_weight_empirical_theory.tight_layout()
        fig_weight_empirical_theory.savefig(output_dir / 'empirical_weight_action_target_histograms_theory_overlay.png', dpi=150)
        plt.close(fig_weight_empirical_theory)
        print(f'Saved empirical target weight theory overlay plot to {output_dir / "empirical_weight_action_target_histograms_theory_overlay.png"}')

    if p_summary_entries:
        from collections import defaultdict

        grouped_summary = defaultdict(list)
        for entry in p_summary_entries:
            grouped_summary[entry["P"]].append(entry)

        p_values = sorted(grouped_summary.keys())
        linear_empirical_mean = []
        linear_empirical_std = []
        linear_theory = []
        cubic_empirical_mean = []
        cubic_empirical_std = []
        cubic_theory = []
        w_empirical_mean = []
        w_empirical_std = []
        w_theory = []
        linear_learnability_empirical_mean = []
        linear_learnability_empirical_std = []
        linear_learnability_theory = []
        cubic_learnability_empirical_mean = []
        cubic_learnability_empirical_std = []
        cubic_learnability_theory = []

        def _sem(values):
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return 0.0
            return float(arr.std(ddof=1) / np.sqrt(arr.size))

        for p_val in p_values:
            entries = grouped_summary[p_val]

            def _mean_sem(key: str):
                values = np.asarray([entry[key] for entry in entries if np.isfinite(entry.get(key, np.nan))], dtype=np.float64)
                if values.size == 0:
                    return np.nan, np.nan
                mean = float(values.mean()) if values.size else np.nan
                return mean, _sem(values)

            lin_mean, lin_std = _mean_sem("linear_empirical")
            cub_mean, cub_std = _mean_sem("cubic_empirical")
            w_mean, w_std = _mean_sem("w_empirical")

            linear_empirical_mean.append(lin_mean)
            linear_empirical_std.append(lin_std)
            cubic_empirical_mean.append(cub_mean)
            cubic_empirical_std.append(cub_std)
            w_empirical_mean.append(w_mean)
            w_empirical_std.append(w_std)

            linear_theory_vals = [entry["linear_theory"] for entry in entries if np.isfinite(entry.get("linear_theory", np.nan))]
            cubic_theory_vals = [entry["cubic_theory"] for entry in entries if np.isfinite(entry.get("cubic_theory", np.nan))]
            w_theory_vals = [entry["w_theory"] for entry in entries if np.isfinite(entry.get("w_theory", np.nan))]
            linear_theory.append(float(np.mean(linear_theory_vals)) if linear_theory_vals else np.nan)
            cubic_theory.append(float(np.mean(cubic_theory_vals)) if cubic_theory_vals else np.nan)
            w_theory.append(float(np.mean(w_theory_vals)) if w_theory_vals else np.nan)

            linear_learnability_emp = []
            linear_learnability_theory_vals = []
            cubic_learnability_emp = []
            cubic_learnability_theory_vals = []
            for entry in entries:
                kappa_entry = entry.get("kappa_eff", entry.get("kappa"))
                if not np.isfinite(kappa_entry):
                    continue
                linear_emp_val = entry.get("linear_learnability_empirical")
                linear_theory_val = learnability_from_eigenvalue(entry.get("linear_theory"), kappa_entry, p_val)
                cubic_emp_val = entry.get("cubic_learnability_empirical")
                cubic_theory_val = learnability_from_eigenvalue(entry.get("cubic_theory"), kappa_entry, p_val)

                if np.isfinite(linear_emp_val):
                    linear_learnability_emp.append(linear_emp_val)
                if np.isfinite(linear_theory_val):
                    linear_learnability_theory_vals.append(linear_theory_val)
                if np.isfinite(cubic_emp_val):
                    cubic_learnability_emp.append(cubic_emp_val)
                if np.isfinite(cubic_theory_val):
                    cubic_learnability_theory_vals.append(cubic_theory_val)

            linear_learnability_empirical_mean.append(
                float(np.nanmean(linear_learnability_emp)) if linear_learnability_emp else np.nan
            )
            linear_learnability_empirical_std.append(_sem(linear_learnability_emp))
            linear_learnability_theory.append(
                float(np.nanmean(linear_learnability_theory_vals)) if linear_learnability_theory_vals else np.nan
            )
            cubic_learnability_empirical_mean.append(
                float(np.nanmean(cubic_learnability_emp)) if cubic_learnability_emp else np.nan
            )
            cubic_learnability_empirical_std.append(_sem(cubic_learnability_emp))
            cubic_learnability_theory.append(
                float(np.nanmean(cubic_learnability_theory_vals)) if cubic_learnability_theory_vals else np.nan
            )

        fig_summary, axes_summary = plt.subplots(1, 3, figsize=(16, 6), dpi=300, sharex=True)
        summary_specs = [
            (
                axes_summary[0],
                "Linear eigenvalue vs P",
                r"$P$",
                r"Eigenvalue / variance",
                linear_empirical_mean,
                linear_empirical_std,
                linear_theory,
                "royalblue",
            ),
            (
                axes_summary[1],
                "Cubic eigenvalue vs P",
                r"$P$",
                r"Eigenvalue / variance",
                cubic_empirical_mean,
                cubic_empirical_std,
                cubic_theory,
                "forestgreen",
            ),
            (
                axes_summary[2],
                "Readin weight variance vs P",
                r"$P$",
                r"Variance",
                w_empirical_mean,
                w_empirical_std,
                w_theory,
                "crimson",
            ),
        ]

        for ax, title, xlabel, ylabel, empirical_mean, empirical_std, theory_vals, color in summary_specs:
            empirical_mean_arr = np.asarray(empirical_mean, dtype=np.float64)
            empirical_std_arr = np.asarray(empirical_std, dtype=np.float64)
            theory_arr = np.asarray(theory_vals, dtype=np.float64)

            ax.errorbar(
                p_values,
                empirical_mean_arr,
                yerr=empirical_std_arr,
                fmt='o',
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label='Empirical',
            )
            ax.plot(
                p_values,
                theory_arr,
                linestyle='--',
                marker='s',
                markersize=5,
                color='black',
                linewidth=1.6,
                label='Theory',
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            annotate_theory_empirical_gaps(
                ax, p_values, empirical_mean_arr, theory_arr, empirical_std_arr
            )

        annotate_scaling_exponents(fig_summary, experiment_group.name)
        fig_summary.savefig(output_dir / 'eigenvalue_variance_vs_P.png', bbox_inches='tight')
        plt.close(fig_summary)
        print(f'Saved eigenvalue/variance summary plot to {output_dir / "eigenvalue_variance_vs_P.png"}')

        fig_learnability, axes_learnability = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
        learnability_specs = [
            (
                axes_learnability[0],
                "Linear test learnability vs P",
                linear_learnability_empirical_mean,
                linear_learnability_empirical_std,
                linear_learnability_theory,
                "royalblue",
                0,
            ),
            (
                axes_learnability[1],
                "Cubic test learnability vs P",
                cubic_learnability_empirical_mean,
                cubic_learnability_empirical_std,
                cubic_learnability_theory,
                "forestgreen",
                None,
            ),
        ]

        for ax, title, empirical_mean, empirical_std, theory_vals, color, ylim_bottom in learnability_specs:
            empirical_mean_arr = np.asarray(empirical_mean, dtype=np.float64)
            empirical_std_arr = np.asarray(empirical_std, dtype=np.float64)
            theory_arr = np.asarray(theory_vals, dtype=np.float64)

            ax.errorbar(
                p_values,
                empirical_mean_arr,
                yerr=empirical_std_arr,
                fmt='o',
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label='Empirical (test)',
            )
            ax.plot(
                p_values,
                theory_arr,
                linestyle='--',
                marker='s',
                markersize=5,
                color='black',
                linewidth=1.6,
                label='Theory',
            )
            ax.set_title(title)
            ax.set_xlabel(r"$P$")
            ax.set_ylabel("Learnability")
            if ylim_bottom is not None:
                ax.set_ylim(bottom=ylim_bottom)
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig_learnability.tight_layout()
        fig_learnability.savefig(output_dir / 'learnability_vs_P.png', bbox_inches='tight')
        plt.close(fig_learnability)
        print(f'Saved learnability summary plot to {output_dir / "learnability_vs_P.png"}')

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_wsq[j//ncols_hist, j%ncols_hist].axis('off')
    fig_wsq.tight_layout()
    fig_wsq.savefig(output_dir / 'weight_sq_target_histograms.png', dpi=150)
    plt.close(fig_wsq)
    print(f'Saved squared target weight plots to {output_dir / "weight_sq_target_histograms.png"}')

    plot_readout_vs_hidden_weights(model_dirs, output_dir, device, experiment_group.name)

    # # --- Output projection histograms (onto target x[:,0] and perp x[:,3]) ---
    # fig_output, axes_output = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)
    # from compute_h3_projections import compute_theory_with_julia
    # for idx, model_dir in enumerate(model_dirs):
    #     d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
    #     model, *_ = load_model(model_dir, device)
    #     if model is None:
    #         continue
    #     # Generate x from seed
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #     # Streaming batch computation for projections
    #     total_samples = 50000000
    #     batch_size = 5000
    #     num_batches = total_samples // batch_size
    #     remainder = total_samples % batch_size
    #     dtype = torch.float32
    #     ens = model.ens
    #     n1 = model.n1
    #     # Accumulators
    #     proj_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    #     proj_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    #     for i in range(num_batches + (1 if remainder > 0 else 0)):
    #         bs = batch_size if i < num_batches else remainder
    #         if bs == 0:
    #             break
    #         X_batch = torch.randn(bs, d, dtype=dtype, device=device)

    #         x0 = X_batch[:, 0]
    #         phi3_target = x0**3 - 3.0 * x0
    #         # Perpendicular projections: average over all x[:,1:]
    #         phi3_perp_sum = torch.zeros(bs, dtype=dtype, device=device)
    #         for j in [1]:
    #             xj = X_batch[:, j]
    #             phi3_perp_sum += xj**3 - 3.0 * xj
    #         phi3_perp = phi3_perp_sum
    #         with torch.no_grad():
    #             a0 = model.h0_activation(X_batch)
    #         proj_target_sum += torch.einsum('pqn,p->qn', a0, phi3_target)
    #         proj_perp_sum += torch.einsum('pqn,p->qn', a0, phi3_perp)

    #         del X_batch, a0, phi3_target, phi3_perp, phi3_perp_sum, x0
    #     del model
    #     torch.cuda.empty_cache()
    #     # Normalize
    #     proj_target = proj_target_sum / total_samples
    #     proj_perp = proj_perp_sum / total_samples
    #     # Compute variances
    #     var_target = proj_target.var().item()
    #     var_perp = proj_perp.var().item()
    #     axo = axes_output[idx//ncols_hist, idx%ncols_hist]
    #     # Histogram and action plot for target

    #     # --- Overlay theoretical Gaussian action curves ---
    #     # Parse config for chi, kappa, epsilon if present
    #     d_cfg, P_cfg, N_cfg, chi, seed_cfg, T_cfg, epsilon_cfg = parse_config_from_dirname(model_dir)
    #     # Estimate kappa as 1/chi if not present
    #     kappa = T_cfg
    #     # Use N as n1
    #     epsilon = epsilon_cfg if epsilon_cfg is not None else 0.0
    #     theory = compute_theory_with_julia(d_cfg, N_cfg, P_cfg, chi, kappa, epsilon)
    #     lJ3T = theory["target"]["lJ3T"]
    #     lJ3P = theory["perpendicular"]["lJ3P"]

    #     # Compute histogram and mask before using for theory overlay x-range
    #     v_flat = proj_target.flatten().cpu()
    #     hist_range = (v_flat.min().item(), v_flat.max().item())
    #     bins = 200
    #     hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
    #     bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     bin_width = (hist_range[1] - hist_range[0]) / bins
    #     probs_density = (hist / hist.sum()) / bin_width
    #     mask = probs_density > 0
    #     bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_cpu = mask.cpu() if mask.is_cuda else mask

    #     # Plot theoretical Gaussian action for target and perp, using a smooth line over the same x-range as experiment (after mask)
    #     import numpy as np
    #     bin_centers_np = bin_centers_cpu[mask_cpu].numpy() if hasattr(bin_centers_cpu, 'numpy') else np.array(bin_centers_cpu)[mask_cpu]
    #     if len(bin_centers_np) > 1:
    #         x_min = float(bin_centers_np.min())
    #         x_max = float(bin_centers_np.max())
    #     else:
    #         x_min = float(bin_edges[0])
    #         x_max = float(bin_edges[-1])
    #     x_theory = np.linspace(x_min, x_max, 1000)
    #     for lJ3, color, label in [
    #         (lJ3T, 'royalblue', 'Theory Target'),
    #         # (lJ3P, 'orange', 'Theory Perp')
    #     ]:
    #         var = lJ3
    #         action = 0.5 * x_theory**2 / var + 0.5 * np.log(2 * np.pi * var)
    #         axo.plot(x_theory, action, '--', color=color, label=f'{label} $\\sigma^2={var:.2e}$')
    #     v_flat = proj_target.flatten().cpu()
    #     hist_range = v_flat.min().item(), v_flat.max().item()
    #     bins = 200
    #     hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
    #     bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     bin_width = (hist_range[1] - hist_range[0]) / bins
    #     probs_density = (hist / hist.sum()) / bin_width
    #     mask = probs_density > 0
    #     bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_cpu = mask.cpu() if mask.is_cuda else mask
    #     axo.plot(
    #         bin_centers_cpu[mask_cpu].numpy(),
    #         (-probs_density[mask].log()).cpu().numpy(),
    #         label=f'Target $x_0$ ($\\sigma^2={var_target:.2e}$)',
    #         color='royalblue'
    #     )
    #     # Histogram and action plot for averaged perp
    #     v_flat_perp = proj_perp.flatten().cpu()
    #     hist_perp = torch.histc(v_flat_perp, bins=bins, min=hist_range[0], max=hist_range[1])
    #     probs_perp_density = (hist_perp / hist_perp.sum()) / bin_width
    #     mask_perp = probs_perp_density > 0
    #     bin_centers_perp_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_perp_cpu = mask_perp.cpu() if mask_perp.is_cuda else mask_perp
    #     d_val = d if d is not None else 0
    #     axo.plot(
    #         bin_centers_perp_cpu[mask_perp_cpu].numpy(),
    #         (-probs_perp_density[mask_perp].log()).cpu().numpy(),
    #         label=f'Perp avg $x_{{1...{d_val-1}}}$ ($\\sigma^2={var_perp:.2e}$)',
    #         color='orange'
    #     )
        
    #     # Parse config parameters for title
    #     d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
    #     axo.set_title(f"Action: $-\\log P$ | d={d}, P={P}, N={N}, $\\chi$={chi}", fontsize=11)
    #     axo.set_xlabel('Output projection value')
    #     axo.set_ylabel('Action: -log P')
    #     axo.legend()
    #     axo.grid(True, alpha=0.3)
    
    # for j in range(idx+1, nrows_hist*ncols_hist):
    #     axes_output[j//ncols_hist, j%ncols_hist].axis('off')
    # fig_output.tight_layout()
    # fig_output.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'output_projection_histograms.png'), dpi=150)
    # plt.close(fig_output)
    # print('Saved output projection histograms to output_projection_histograms.png')

    # Hide unused subplots
    for j in range(len(param_keys), nrows*ncols):
        axes_scatter[j//ncols, j%ncols].axis('off')
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / 'grid_X0_vs_model_output_grouped.png', dpi=150)
    print(f'Saved grouped grid scatter plot to {output_dir / "grid_X0_vs_model_output_grouped.png"}')


def main():
    group_names = [group.name for group in EXPERIMENT_GROUPS]
    parser = argparse.ArgumentParser(description='Plot h0 activation and target weight actions')
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Recompute all cached projections/learnability/VGA data for selected groups (plots always regenerate). Without this flag, only missing runs are computed.',
    )
    parser.add_argument(
        '--group',
        action='append',
        dest='groups',
        choices=group_names,
        metavar='NAME',
        help=f'Experiment group to plot (repeatable). Choices: {", ".join(group_names)}. Default: all groups.',
    )
    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List available experiment groups and exit',
    )
    parser.add_argument(
        '--vga-advanced',
        action='store_true',
        help='Use exact Gauss-Hermite GMM entropy in VGA (instead of Hershey-Olsen bound). Affects mixture coeffs muW/sigS.',
    )
    args = parser.parse_args()

    if args.list_groups:
        for group in EXPERIMENT_GROUPS:
            print(f"{group.name}: {len(group.model_dirs)} model dirs")
        return

    selected_names = args.groups if args.groups else group_names
    selected_groups = [EXPERIMENT_GROUP_BY_NAME[name] for name in selected_names]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for experiment_group in selected_groups:
        plot_experiment_group(
            experiment_group,
            device,
            recompute=args.recompute,
            vga_advanced=args.vga_advanced,
        )

if __name__ == "__main__":
    main()
