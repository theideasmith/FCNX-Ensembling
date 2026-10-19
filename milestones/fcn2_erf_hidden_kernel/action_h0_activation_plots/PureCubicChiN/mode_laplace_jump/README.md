# Mode-Laplace f* jump: Φ-min root selection

Among multiple roots of `g(f)=0`, pick the mixture free-energy minimum

```
Φ(f) = -(1-f) log Z_b - f log Z_s + f log f + (1-f) log(1-f)
```

Artifacts:
- `gscan_rho*.csv` — fixed-f `g(f)` tables
- `roots_Phi_vs_rho.csv` — each root re-solved with Φ
- `phimin_vs_bisect.csv` — Φ-min vs naive bisect pick
- `L3_phimin_vs_experiment.png` — L3 with experiment overlay

Reproduce:
```bash
ITERS=30 LAPLACE_DEV=cuda:1 python -u plot_mode_laplace_jump_phimin.py
```
