"""SI figures for ketoprofen: seed-resolved RMSD_ref and coupling to the extended variable.

Run from the repository root:
    python Figures/scripts/si_keto.py
Writes Figures/SI/FigS_keto_rmsd.pdf and Figures/SI/FigS_keto_coupling.pdf (+ .png previews
in build/).

Data: Data/ketoprofen/ (export_keto_data.py).
RMSD: running RMSD of the symmetrized PMF to the 3.4 us reference, every 1 ns, as in
si_sweeps.py (faint replicas, median over replicas solid, log y). (A) stage one, three replicas
per biasTemperature. (B, C) stage two, grouped by extendedFluctuation (B) or fullSamples (C),
20 replicas per value. Dashed line: kT at 310 K.
Coupling: as si_coupling.py for the stage-two runs, grouped by extendedFluctuation. (A) uses the
counts pooled over the 20 replicas of each value, (B) the per-run SD (mean, min-max) and |mean|.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import si_sweeps  # noqa: F401  (shared style)
from si_sweeps import letters, save, C_MED, C_FEW

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data', 'ketoprofen')
KT = 0.0019872041 * 310.0
cmap = plt.get_cmap('viridis')
TGRID = np.arange(1.0, 500.5, 1.0)


def rmsd_figure():
    d = pd.read_csv(os.path.join(DATA, 'keto_rmsd_ref.csv'))
    panels_cfg = [
        ('biasTemperature (K)', d[d.stage == 1], 'biasT', '{:g}'),
        ('extendedFluctuation (Å)', d[d.stage == 2], 'extFluc', '{:g}'),
        ('fullSamples', d[d.stage == 2], 'fullSamples', '{:.0f}'),
    ]
    PW, PH = 1.45, 1.30
    X0, DX = 0.55, 2.15
    Y0 = 0.45
    W, H = X0 + 2 * DX + PW + 0.6, Y0 + PH + 0.25
    fig = plt.figure(figsize=(W, H))
    panels = []
    for col, (title, sel, key, fmt) in enumerate(panels_cfg):
        ax = fig.add_axes([(X0 + col * DX) / W, Y0 / H, PW / W, PH / H])
        vals = sorted(sel[key].unique())
        for i, v in enumerate(vals):
            c = cmap(0.92 * i / max(len(vals) - 1, 1))
            # Restarted runs have slightly shifted block times: put every run on a 1 ns grid.
            R = np.array([np.interp(TGRID, g.time_ns, g.rmsd, left=np.nan)
                          for _, g in sel[sel[key] == v].groupby('run')])
            for r in R:
                ax.plot(TGRID, r, color=c, lw=0.3, alpha=0.25)
            ax.plot(TGRID, np.nanmedian(R, axis=0), color=c, lw=1.1, label=fmt.format(v))
        ax.axhline(KT, color='k', lw=0.7, ls='--', zorder=5)
        ax.set_xlim(0, 500)
        ax.set_xticks(np.arange(0, 501, 100))
        ax.set_yscale('log')
        ax.set_ylim(0.1, 10)
        ax.set_yticks([0.1, 1, 10])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(title, fontsize=8, pad=2)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02), fontsize=6, ncol=1,
                  handlelength=1.0, handletextpad=0.3, labelspacing=0.15, borderaxespad=0.2)
        ax.set_xlabel('Time (ns)')
        if col == 0:
            ax.set_ylabel(r'RMSD$_\mathrm{ref}$ (kcal/mol)')
        panels.append(ax)
    letters(fig, panels, W, H)
    save(fig, 'FigS_keto_rmsd')


def coupling_figure():
    hist = pd.read_csv(os.path.join(DATA, 'keto_xi_minus_lambda_hist.csv'))
    stats = pd.read_csv(os.path.join(DATA, 'keto_xi_minus_lambda_stats.csv'))
    hist, stats = hist[hist.stage == 2], stats[stats.stage == 2]

    W, H = 3.33, 4.6
    fig = plt.figure(figsize=(W, H))
    axA = fig.add_axes([0.62 / W, 2.75 / H, 2.55 / W, 1.55 / H])
    axB = fig.add_axes([0.62 / W, 0.50 / H, 2.55 / W, 1.55 / H])

    # A: distributions of (xi - lambda) / extendedFluctuation, averaged over the replicas
    grid = np.linspace(-6, 6, 600)
    vals = sorted(hist.extFluc.unique())
    edges = np.linspace(-6.0, 6.0, 1201)
    centers = np.round(0.5 * (edges[1:] + edges[:-1]), 4)
    for i, v in enumerate(vals):
        c = cmap(0.92 * i / (len(vals) - 1))
        # Counts pooled over the replicas (runs restarted mid-way have fewer samples), then
        # rebinned to 0.1 extendedFluctuation so that all values have the same resolution.
        g = hist[hist.extFluc == v].groupby('bin_center')['count'].sum()
        cnt = pd.Series(0, index=centers).add(g, fill_value=0).to_numpy()
        x = centers / v
        p = cnt / (cnt.sum() * 0.01 / v)
        step = max(int(round(0.1 * v / 0.01)), 1)
        n = len(x) // step * step
        dens = np.interp(grid, x[:n].reshape(-1, step).mean(1),
                         p[:n].reshape(-1, step).mean(1), left=0, right=0)
        axA.plot(grid, dens, color=c, lw=1.0, label=f'{v:g}')
    axA.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color='k', lw=0.8, ls='--',
             label='Normal')
    axA.set_xlim(-4, 4)
    axA.set_ylim(0, 0.45)
    axA.set_xlabel(r'($\xi-\lambda$) / extendedFluctuation')
    axA.set_ylabel('Probability density')
    axA.legend(loc='upper left', fontsize=6, ncol=1, handlelength=1.0, handletextpad=0.3,
               labelspacing=0.15, title='extended\nFluctuation (Å)', title_fontsize=6)

    # B: SD and |mean| of xi - lambda against the target
    s = stats.groupby('extFluc').agg(sd=('std', 'mean'), sd_lo=('std', 'min'),
                                     sd_hi=('std', 'max'), mu=('mean', lambda x: np.abs(x).mean()))
    v = s.index.to_numpy()
    axB.plot([0.005, 10], [0.005, 10], color='0.6', lw=0.6, zorder=0)
    axB.errorbar(v, s.sd, yerr=[s.sd - s.sd_lo, s.sd_hi - s.sd], fmt='o-', color=C_MED, ms=3,
                 lw=1.0, capsize=1.5, elinewidth=0.6, label='SD')
    axB.plot(v, s.mu, 's-', color=C_FEW, ms=3, lw=1.0, label='|Mean|')
    axB.set_xscale('log')
    axB.set_yscale('log')
    axB.set_xlim(0.03, 1)
    axB.set_ylim(1e-4, 1)
    fmt = mpl.ticker.FuncFormatter(lambda t, _: f'{t:g}')
    axB.xaxis.set_major_formatter(fmt)
    axB.yaxis.set_major_formatter(fmt)
    axB.set_xticks([0.05, 0.1, 0.2, 0.5])
    axB.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axB.set_xlabel('extendedFluctuation (Å)')
    axB.set_ylabel(r'$\xi-\lambda$ (Å)')
    axB.legend(loc='upper left', fontsize=7, handlelength=1.5)

    for t, ax in zip('AB', (axA, axB)):
        p = ax.get_position()
        fig.text(0.02, p.y1 + 0.03 / H, t, fontsize=12, fontweight='bold', ha='left',
                 va='bottom')
    save(fig, 'FigS_keto_coupling')


if __name__ == '__main__':
    rmsd_figure()
    coupling_figure()
