# Erf D-Sweep Mean-Field Scaling Convergence

This milestone replicates the d-sweep structure from `linear_mf_scaling_convergence` but uses **erf activation** instead of linear activation.

## Overview

**Goal**: Study eigenvalue scaling with dimension `d` for FCN3 networks with erf activation under mean-field scaling.

**Network**: `FCN3NetworkEnsembleErf` with 3 layers
- Input dimension: `d`
- Hidden widths: `N = 50`, `chi = N = 50`
- Training samples: `P = 3*d`
- Ensembles: 50 networks trained in parallel

**Training**:
- Algorithm: Stochastic Langevin dynamics
- Learning rate: `lr = 1e-5 / P`
- Temperature: `T = 2 * kappa` where `kappa = 1/chi`
- Weight decay: Layer-specific with mean-field scaling
- Epochs: 300M (checkpointed every 10K)

## Files

### Training Scripts

- **`d_sweep.py`**: Main training script
  - Trains all d values in parallel (d = 2, 6, 8, 10)
  - Single d mode: `python d_sweep.py --d 2 --device cuda:0`
  - Parallel mode: `python d_sweep.py` (launches all d values)
  - Saves: model checkpoints, eigenvalues over time, losses, config

### Monitoring Scripts

- **`poll_and_plot_lH_vs_d.py`**: Real-time monitoring
  - Polls model checkpoints every 5 seconds
  - Computes empirical eigenvalues using `H_eig(std=True)`
  - Computes theoretical predictions from `ExperimentErf`
  - Plots log-log: empirical vs theory lH vs d with slopes
  - Tracks slope convergence over time

## Usage

### 1. Start Training (Parallel)
```bash
cd /home/akiva/FCNX-Ensembling/milestones/erf_d_sweep
python d_sweep.py
```

This launches 4 parallel processes:
- d=2, d=8 on cuda:0
- d=6, d=10 on cuda:1

### 2. Start Training (Single d)
```bash
python d_sweep.py --d 2 --device cuda:0
```

### 3. Monitor in Real-Time
```bash
# In a separate terminal
python poll_and_plot_lH_vs_d.py
```

Outputs:
- `plots/poll_lH_vs_d.png`: Empirical vs theory lH vs d
- `plots/slopes_over_time.png`: Convergence of log-log slopes
- `plots/slope_history.json`: Slope history data

## Output Structure

```
erf_d_sweep/
├── d2_P6_N50_chi50/
│   ├── config.json
│   ├── model.pt (latest checkpoint)
│   ├── model_final.pt
│   ├── eigenvalues_over_time.json
│   ├── eigenvalues_over_epochs.png
│   ├── eigenvalues_vs_predictions.png
│   ├── predictions.json
│   └── losses.json
├── d6_P18_N50_chi50/
├── d8_P24_N50_chi50/
├── d10_P30_N50_chi50/
├── plots/
│   ├── poll_lH_vs_d.png
│   ├── slopes_over_time.png
│   └── slope_history.json
├── runs/ (TensorBoard logs)
└── d_sweep.py, poll_and_plot_lH_vs_d.py
```

## Key Differences from Linear Version

1. **Activation**: Uses `FCN3NetworkEnsembleErf` with `torch.erf()` activation
2. **Theory**: Uses `ExperimentErf.eig_predictions()` for theoretical eigenvalues
3. **Eigenvalues**: Erf networks have different spectral properties than linear
4. **Convergence**: May show different scaling behavior with d

## Expected Results

- **Empirical eigenvalues**: Top eigenvalue and mean of rest computed from ensemble
- **Theoretical predictions**: `lHT` (training regime) from EK theory
- **Scaling**: Log-log slope analysis shows how eigenvalues scale with d
- **Comparison**: Erf vs linear scaling differences

## TensorBoard

View training metrics:
```bash
tensorboard --logdir=runs
```

Tracks:
- Per-ensemble losses
- All eigenvalues over time
- Convergence to theoretical predictions

## Notes

- Training 300M epochs takes significant time (hours to days per d)
- Polling script is non-invasive and can run during training
- Uses same mean-field scaling conventions as linear version
- Weight initialization: `(1/d, 1/N, 1/(N*chi))`
