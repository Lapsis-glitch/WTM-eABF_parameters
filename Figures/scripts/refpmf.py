"""Reference PMFs (median, average over all runs, average without outliers).

Re-implements reference_builder.compute_reference_pmf_with_outliers from
github.com/Lapsis-glitch/WTM-eABF_parameters (Convergence_evaluation @ 2f5bbe7)
on the per-run final PMFs in Data/. The original uses kB in kJ/mol/K on PMFs
in kcal/mol. Pass kB=KB_KJ to reproduce it, KB_KCAL for the corrected values.
"""

import csv
import collections

import numpy as np
from scipy.interpolate import RegularGridInterpolator

KB_KJ = 0.008314462618      # kJ/mol/K, as in reference_builder.py
KB_KCAL = 0.0019872043      # kcal/mol/K


def read_final_pmfs(path):
    """Per-run final PMFs from a param,value,seed,xi,G CSV.
    Returns {(param, value, seed): (xi, G)}."""
    runs = collections.defaultdict(lambda: ([], []))
    with open(path) as f:
        for r in csv.DictReader(f):
            x, g = runs[(r['param'], r['value'], r['seed'])]
            x.append(float(r['xi']))
            g.append(float(r['G']))
    return {k: (np.array(x), np.array(g)) for k, (x, g) in runs.items()}


def interpolate(xi, G, n_points=100):
    """Linear interpolation onto n_points between the run grid ends (pmf_io)."""
    grid = np.linspace(xi.min(), xi.max(), n_points)
    f = RegularGridInterpolator((xi,), G, bounds_error=False, fill_value=np.nan)
    return grid, f(grid[:, None])


def references(G_stack, T, kB, mad_cut=3.5):
    """G_stack: (n_runs, n_points) in kcal/mol. Returns dict of PMFs (min 0),
    errors and the boolean mask of the runs kept by the MAD filter."""
    beta = 1.0 / (kB * T)
    P = np.exp(-beta * G_stack)
    P /= P.sum(axis=1, keepdims=True)

    def to_pmf(p):
        F = -np.log(p) / beta
        return F - np.min(F)

    P_med = np.median(P, axis=0)
    P_all = P.mean(axis=0)
    dev = np.sqrt(np.mean((P - P_med) ** 2, axis=1))
    mad = np.median(np.abs(dev - np.median(dev))) + 1e-12
    keep = dev < np.median(dev) + mad_cut * mad
    P_filt = P[keep].mean(axis=0)
    return dict(
        median=to_pmf(P_med),
        average_all=to_pmf(P_all),
        average_all_err=(P.std(axis=0) / (P_all + 1e-12)) / beta,
        average_filtered=to_pmf(P_filt),
        average_filtered_err=(P[keep].std(axis=0) / (P_filt + 1e-12)) / beta,
        keep=keep,
    )


def rmsd(a, b):
    return np.sqrt(np.nanmean(((a - np.nanmin(a)) - (b - np.nanmin(b))) ** 2))
