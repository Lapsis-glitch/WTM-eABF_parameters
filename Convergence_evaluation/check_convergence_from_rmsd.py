#!/usr/bin/env python3
"""Recompute the per-replica convergence frames from the saved RMSD time series.

The convergence criterion of PMFAnalyzer (analyze_ND.py) only needs the raw RMSD of each
history PMF to the reference. Data/<sweep>/*rmsd*.csv stores that series for every replica,
and Data/<sweep>/*convergence*per_seed*.csv the convergence frame found when the full PMF
histories were analysed with folder_parser.py. This script feeds each saved RMSD series
through the same smoothing, exponential fit and detection code, with the settings of
folder_parser.py, and reports how many replicas give the same frame.

Usage (from the repository root):
    python Convergence_evaluation/check_convergence_from_rmsd.py Data
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ND import PMFAnalyzer  # noqa: E402

# Settings used by folder_parser.py / folder_parser_2D.py
RMSD_THRESH = 0.592186869182
SLOPE_THRESH = 0.01


def convergence_frame(rmsd):
    """Run the PMFAnalyzer criterion on one RMSD series, return the frame index or None."""
    a = PMFAnalyzer.__new__(PMFAnalyzer)
    a.rmsd_thresh = RMSD_THRESH
    a.slope_thresh = SLOPE_THRESH
    a.use_ref_and_slope = True
    a.count_std_thresh = None
    a.reference_pmf = True          # only tested for "is not None"
    a.rmsd_raw = np.asarray(rmsd, dtype=float)
    a.t = np.arange(len(a.rmsd_raw))
    a.rmsd_smooth = a._smooth_rmsd(a.rmsd_raw)
    a.params, a.rmsd_fit = a._fit_exp_decay()
    idx = a._detect_convergence()
    return None if idx is None else int(idx)


def check(folder):
    conv = glob.glob(os.path.join(folder, '*convergence*per_seed*.csv'))
    rmsd = glob.glob(os.path.join(folder, '*rmsd*.csv'))
    if len(conv) != 1 or len(rmsd) != 1:
        return None
    c = pd.read_csv(conv[0], dtype=str)
    r = pd.read_csv(rmsd[0], dtype={k: str for k in c.columns[:3]})
    keys = list(c.columns[:3])
    expected = {tuple(row[keys]): (None if pd.isna(row['frame']) else int(row['frame']))
                for _, row in c.iterrows()}
    n_same, diffs = 0, []
    for key, g in r.groupby(keys, sort=False):
        got = convergence_frame(g.sort_values('frame')['rmsd'].to_numpy())
        want = expected.get(tuple(key), 'missing')
        if got == want:
            n_same += 1
        else:
            diffs.append((key, want, got))
    return n_same, len(expected), diffs


def main():
    warnings.filterwarnings('ignore')   # curve_fit overflow warnings, as in folder_parser.py runs
    root = sys.argv[1] if len(sys.argv) > 1 else 'Data'
    for folder in sorted(glob.glob(os.path.join(root, '*'))):
        res = check(folder)
        if res is None:
            continue
        n_same, n, diffs = res
        print(f'{os.path.basename(folder):24s} {n_same:5d} / {n:5d} replicas identical')
        for key, want, got in diffs[:10]:
            print(f'    {key}: saved {want}, recomputed {got}')
        if len(diffs) > 10:
            print(f'    ... {len(diffs) - 10} more')


if __name__ == '__main__':
    main()
