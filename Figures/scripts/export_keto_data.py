"""Export the ketoprofen data used by the figures from the raw NAMD output into Data/ketoprofen/.

Run from the repository root, after ./unpack_keto.sh has decompressed the PMF histories:
    python Figures/scripts/export_keto_data.py

Uses the I/O and PMF code of Ketoprofen/analysis (keto_pmf, pspace,
convergence). Every PMF is symmetrized in G space and anchored to bulk (|z| >= 35 A = 0). RMSDs
are taken over |z| <= 38 A after removing the mean difference, against the symmetrized
long-time_4000_seed40 reference (~3.4 us). kT = kB * 310 K = 0.616 kcal/mol.

Writes
  keto_per_seed.csv          one row per run: final RMSD to the reference, first crossing of kT
  keto_cells.csv             one row per cell: accuracy mean/SD over seeds, mean pairwise RMSD
  keto_rmsd_ref.csv          running RMSD to the reference, every 1 ns (every fifth history block)
  keto_xi_minus_lambda_hist.csv / _stats.csv   xi - lambda from the colvars trajectories
                             (every 0.1 ns, step 0 dropped; restarted runs only after the restart)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KETO = os.path.join(ROOT, 'Ketoprofen', 'runs')
sys.path.insert(0, os.path.join(ROOT, 'Ketoprofen', 'analysis'))
import keto_pmf as k        # noqa: E402
import pspace as ps         # noqa: E402
import convergence as cv    # noqa: E402

OUT = os.path.join(ROOT, 'Data', 'ketoprofen')
ZMAX = 38.0
EVERY = 5                    # history blocks are 0.2 ns apart, keep 1 ns for the time series
HIST_EDGES = np.linspace(-6.0, 6.0, 1201)    # 0.01 A bins, only non-empty bins are written


def anchor(G, z):
    G = ps.symmetrize_free(G)
    return G - G[..., np.abs(z) >= 35].mean(axis=-1, keepdims=True)


def rmsd_rows(G, ref, mask):
    d = G[:, mask] - ref[mask]
    d = d - d.mean(axis=1, keepdims=True)
    return np.sqrt((d ** 2).mean(axis=1))


def xi_minus_lambda(path):
    a = np.loadtxt(path, comments='#')
    a = a[a[:, 0] > 0]                       # drop step 0 (xi = lambda by construction)
    return a[:, 1] - a[:, 2]


def ids(r):
    if r.family == 'biastemp':
        return dict(stage=1, extFluc=0.1, fullSamples=5000, biasT=r.params['biastemp'])
    return dict(stage=2, extFluc=r.params['extFluc'], fullSamples=r.params['fullSamp'],
                biasT=4000)


def main():
    zr, gr = k.read_pmf(os.path.join(KETO, 'long-time_4000_seed40', 'output', k.CZAR_NAME))
    ref = anchor(gr, zr)
    mask = np.abs(zr) <= ZMAX

    seeds, series, hists, stats, final = [], [], [], [], {}
    for r in k.discover_runs(KETO):
        base = dict(run=r.dirname, cell=r.key, **ids(r), seed=r.seed)
        # final PMF (live czar file, full 500 ns of sampling also for restarted runs)
        z, g = k.read_pmf(r.czar)
        assert np.allclose(z, zr)
        gs = anchor(g, z)
        final[r.dirname] = gs
        acc = ps.rmsd(gs, ref, mask)
        # running RMSD from the stitched history
        t, G, z2, rec = cv.running_history(r)
        assert rec and np.allclose(z2, zr)
        rm = rmsd_rows(anchor(G, z2), ref, mask)
        below = np.flatnonzero(rm < k.KT)
        seeds.append(dict(base, final_rmsd=acc, last_block_rmsd=rm[-1],
                          first_crossing_ns=t[below[0]] if len(below) else np.nan,
                          restart_first_step=r.restart_first_step()))
        sel = slice(EVERY - 1, None, EVERY)
        series.append(pd.DataFrame(dict(run=r.dirname, time_ns=np.round(t[sel], 3),
                                        rmsd=np.round(rm[sel], 5))))
        x = xi_minus_lambda(r.traj)
        c, _ = np.histogram(x, HIST_EDGES)
        h = pd.DataFrame(dict(run=r.dirname,
                              bin_center=np.round(0.5 * (HIST_EDGES[1:] + HIST_EDGES[:-1]), 4),
                              count=c))
        hists.append(h[h['count'] > 0])
        stats.append(dict(run=r.dirname, n_samples=len(x), n_outside=int(len(x) - c.sum()),
                          mean=x.mean(), std=x.std(), min=x.min(), max=x.max()))
        print(f'{r.dirname:38s} final={acc:.3f} last_block={rm[-1]:.3f} '
              f'cross={seeds[-1]["first_crossing_ns"]:.1f}')

    per_seed = pd.DataFrame(seeds)
    rows = []
    for cell, g in per_seed.groupby('cell', sort=False):
        P = np.array([final[x] for x in g.run])
        pair = [ps.rmsd(P[i], P[j], mask) for i in range(len(P)) for j in range(i + 1, len(P))]
        fc = g.first_crossing_ns
        rows.append(dict(cell=cell, **{c: g[c].iloc[0] for c in
                                       ('stage', 'extFluc', 'fullSamples', 'biasT')},
                         n=len(g), acc_mean=g.final_rmsd.mean(), acc_sd=g.final_rmsd.std(ddof=1),
                         repro_mean_pairwise=np.mean(pair), n_crossed=fc.notna().sum(),
                         first_crossing_mean=fc.mean(), first_crossing_sd=fc.std(ddof=1)))

    os.makedirs(OUT, exist_ok=True)
    meta = ['run', 'cell', 'stage', 'extFluc', 'fullSamples', 'biasT', 'seed']
    per_seed.to_csv(os.path.join(OUT, 'keto_per_seed.csv'), index=False, float_format='%.5g')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'keto_cells.csv'), index=False,
                              float_format='%.5g')
    m = per_seed[meta]
    pd.concat(series).merge(m, on='run')[meta + ['time_ns', 'rmsd']].to_csv(
        os.path.join(OUT, 'keto_rmsd_ref.csv'), index=False)
    pd.concat(hists).merge(m, on='run')[meta + ['bin_center', 'count']].to_csv(
        os.path.join(OUT, 'keto_xi_minus_lambda_hist.csv'), index=False)
    pd.DataFrame(stats).merge(m, on='run')[meta + ['n_samples', 'n_outside', 'mean', 'std',
                                                   'min', 'max']].to_csv(
        os.path.join(OUT, 'keto_xi_minus_lambda_stats.csv'), index=False, float_format='%.6g')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
