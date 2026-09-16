#!/bin/bash
# Resume the completed d150/P600/N1400/chi280 schedule run and hold the
# final lr (lr0/9) for 50 million additional wall epochs.
set -euo pipefail
cd /home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel
exec /home/akiva/miniconda3/envs/ml_env/bin/python -u train_fcn2_erf.py \
  --d 150 --P 600 --N 1400 --chi 280 \
  --temperature 2.0 --lr 0.0001 --eps 0.03 --ens 10 \
  --dataset-seed 0 --epochs 20000000 --extra-epochs 50000000 \
  --log-interval 500000 --device cuda:0 --classic --schedule \
  2>&1 | tee -a launcher_logs_chi_N_over_5_schedule_d150_P600_N1400.log
