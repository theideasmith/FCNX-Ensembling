# Equilibration test (report Section 8)

Same diagnostic as `snapshot_history_crossings.py`: follow each neuron's `w0` through
`A_snapshots`, count bulk↔outer crossings with hysteresis `|w0|<0.5` / `|w0|>1.0`,
and track outer fraction over time.

Script: `pure_cubic_equilibration_crossings.py`

| ρ | i→o | o→i | fo first/last 10 | verdict |
|---|---|---|---|---|
| 0.15 | 28 | 28 | 0.095 / 0.093 | EQUILIBRATED |
| 0.08 | 1 | 0 | 0.156 / 0.162 | ONE-WAY / STALLED |
| 0.03 | 0 | 0 | 0.099 / 0.100 | NO CROSSINGS |
| 0.015 | 0 | 0 | 0.324 / 0.325 | NO CROSSINGS |

Flat outer fraction ≠ equilibrium: at low ρ the chains are frozen (inner w0 autocorr
lag-1 ≈ 0.98). Same failure mode as steepwell d≥16 before preconditioned Langevin.
