#!/bin/bash

python fcn2_ensemble_mse.py --models \
/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_0_eps_0.03 \
/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_1_eps_0.03 \
/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_2_eps_0.03 \
/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_3_eps_0.03
