#!/usr/bin/env python3
"""
Run a short profile using `script/diagnostics_langevin_runtime.run_benchmark`
for P=5*d, N=4*d, with d log-spaced from 10..80 (10 points). Prints a table
of diagnostics.

This runner uses repeats=1 to keep runtime low. Adjust `repeats` or d-range
if you want more accurate timings.
"""
import os
import sys
from pprint import pprint

# import the run_benchmark function
from script.diagnostics_langevin_runtime import run_benchmark
import numpy as np


def main():
    # 10 log-spaced points from 10 to 80
    d_min = 10
    d_max = 80
    d_points = 10

    # Define P-scaling: P = 5 * d
    p_scalings = [("5d", lambda d: int(5 * d))]

    # Run short benchmark: repeats=1, ens=1
    results = run_benchmark(d_min=d_min, d_max=d_max, d_points=d_points,
                            ens=1, repeats=1, use_cuda=True, out_dir=None,
                            batch_size=4096, p_scalings=p_scalings)

    # Print concise diagnostics
    print('\nProfile results (P=5*d, N=4*d):')
    for scale_label, res in results.items():
        print(f"\nScaling: {scale_label}")
        ds = res['d']
        Ps = res['P']
        Ns = res['N']
        means = res['times_mean']
        stds = res['times_std']
        learns1 = res.get('learn1', [float('nan')] * len(ds))
        learns3 = res.get('learn3', [float('nan')] * len(ds))

        def human_time(sec: float) -> str:
            if not np.isfinite(sec) or sec < 0:
                return 'N/A'
            sec = int(round(sec))
            days, rem = divmod(sec, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, seconds = divmod(rem, 60)
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            if minutes:
                parts.append(f"{minutes}m")
            parts.append(f"{seconds}s")
            return ' '.join(parts)

        print(f"{'d':>6} {'P':>8} {'N':>6} {'time_mean(s)':>14} {'time_std(s)':>12} {'ep/sec':>10} {'ETA_25M':>14} {'learn1':>8} {'learn3':>8}")
        for i, d in enumerate(ds):
            mean_t = float(means[i])
            std_t = float(stds[i])
            ep_per_sec = 1.0 / mean_t if mean_t > 0 and np.isfinite(mean_t) else float('nan')
            total_epochs = 25_000_000
            est_seconds = mean_t * total_epochs if np.isfinite(mean_t) else float('nan')
            est_hr = human_time(est_seconds)
            print(f"{int(d):6d} {int(Ps[i]):8d} {int(Ns[i]):6d} {mean_t:14.6f} {std_t:12.6f} {ep_per_sec:10.4g} {est_hr:>14} {float(learns1[i]):8.4g} {float(learns3[i]):8.4g}")


if __name__ == '__main__':
    main()
