# Two-Layer ERF Network Hidden Layer Kernel Eigenvalue Analysis

## Overview

This milestone implements and analyzes a **two-layer fully connected neural network** with one hidden layer using `erf` activation. The focus is on computing and tracking the eigenvalues of the **pre-activation kernel H** of the hidden layer during training.

## Architecture

```
Input(d) → Hidden(n1) with erf activation → Output(1) linear readout
```

- **Input dimension**: d
- **Hidden layer width**: n1 (denoted as N in scripts)
- **Output**: Single linear readout
- **Ensemble size**: 50 networks trained in parallel
- **Training**: Langevin dynamics with weight decay

## Pre-Activation Kernel H

The kernel H captures correlations in the pre-activations of the hidden layer:

```
h0[u, q, k] = (W0[q] @ X[u])[k]  (pre-activation for sample u, ensemble q, neuron k)

K_q[u, v] = (1/(n1·P)) Σ_k h0[u,q,k] · h0[v,q,k]  (per-ensemble kernel)

K[u, v] = mean_q(K_q[u, v])  (ensemble-averaged kernel)
```

Eigenvalues are computed via Rayleigh quotients with target eigenfunctions Y.

## Implementation

### Core Classes

**`FCN2NetworkActivationGeneric`** ([lib/FCN2Network.py](../../lib/FCN2Network.py)):
- Two-layer network with ensemble support
- Supports `erf` and `linear` activations
- `H_eig(X, Y, std=False)` method computes kernel eigenvalues
  - Returns eigenvalues for eigenfunctions Y
  - Optional: returns (mean, std) over ensemble members
- Efficient einsum-based implementation with path optimization

**Convenience classes**:
- `FCN2NetworkEnsembleErf`: Erf activation (default)
- `FCN2NetworkEnsembleLinear`: Linear activation

### Training Script

**`train_fcn2_erf.py`**: Main training script with eigenvalue tracking

```bash
python train_fcn2_erf.py --d 10 --P 30 --N 100 --epochs 10000000 --device cuda:0
```

**Key features**:
- Langevin dynamics training with temperature-based weight decay
- Tracks eigenvalues every 10K epochs
- Saves checkpoints and supports resuming
- TensorBoard logging for loss and eigenvalues
- Plots eigenvalue evolution over training

**Hyperparameters**:
- Learning rate: `lr = 1e-5` (default)
- Temperature: `T = 0.02` (default)
- Weight decay: `λ_W0 = d·T`, `λ_A = N·T`
- Langevin noise: `σ = √(2·lr·T)`
- Weight init: `σ²_W0 = 1/d`, `σ²_A = 1/N`

### Test Suite

**`test_fcn2.py`**: Comprehensive tests for H_eig computation

```bash
python test_fcn2.py
```

Tests include:
1. Basic H_eig computation
2. Standard deviation computation across ensembles
3. Multiple eigenfunction columns
4. Linear vs erf activation comparison
5. Forward pass shape verification

## Usage Examples

### Quick Test

```bash
cd /home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel
python test_fcn2.py
```

### Train Small Network

```bash
python train_fcn2_erf.py --d 5 --P 15 --N 50 --epochs 1000000 --device cpu
```

### Train Larger Network on GPU

```bash
python train_fcn2_erf.py --d 20 --P 60 --N 200 --epochs 50000000 --lr 5e-6 --temperature 0.01 --device cuda:0
```

### Monitor with TensorBoard

```bash
tensorboard --logdir=runs/
```

## Output Structure

Training creates a directory: `d{d}_P{P}_N{N}/`

Contents:
- `model.pt`: Latest checkpoint
- `model_final.pt`: Final model after training
- `config.json`: Training configuration
- `eigenvalues_over_time.json`: Eigenvalue history
- `losses.json`: Loss history (mean and std over ensembles)
- `eigenvalues_over_time.png`: Plot of eigenvalue evolution

TensorBoard logs saved to: `runs/d{d}_P{P}_N{N}/`

## Comparison with 3-Layer Networks

This milestone provides a simpler baseline to compare with the 3-layer networks:

| Aspect | 2-Layer (this milestone) | 3-Layer (existing) |
|--------|-------------------------|-------------------|
| Hidden layers | 1 | 2 |
| Kernel analyzed | H (hidden layer pre-activation) | H (second hidden layer pre-activation) |
| Parameters | W0: (ens, n1, d), A: (ens, n1) | W0, W1, A |
| Training complexity | Lower | Higher |
| Theory available | Simpler analytical solutions | More complex NTK analysis |

## Future Directions

1. **Theoretical comparison**: Compare empirical eigenvalues with GP kernel theory
2. **Scaling studies**: Sweep d with P=3d to study eigenvalue scaling
3. **Activation comparison**: Compare erf vs linear hidden layer kernels
4. **Finite-width effects**: Study convergence as N → ∞
5. **Mean-field limit**: Compare with chi scaling (χ = N)

## Files

```
milestones/fcn2_erf_hidden_kernel/
├── README.md                    # This file
├── train_fcn2_erf.py           # Training script
└── test_fcn2.py                # Test suite

lib/
└── FCN2Network.py              # Network implementations
```

## Dependencies

- PyTorch
- opt_einsum
- numpy
- matplotlib
- tensorboard

All dependencies are in the main project's `requirements.txt`.
