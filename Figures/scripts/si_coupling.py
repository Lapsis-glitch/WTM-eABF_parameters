"""SI figure: coupling between the CV and the extended variable for deca-alanine.

Run from the repository root:
    python Figures/scripts/si_coupling.py
Writes Figures/SI/FigS_deca_coupling.pdf (+ .png preview in build/).

Data: Data/deca_ala_1D/deca_ala_1D_xi_minus_lambda_{hist,stats}.csv (xi - lambda from the
colvars trajectories of all 870 runs of the main deca-alanine sweep, every 1 ps). Only the
extendedFluctuation sweep is drawn. The other sweeps keep extendedFluctuation at 0.1 A.
(A) Distributions of (xi - lambda) / extendedFluctuation, averaged over the ten replicas,
against the standard normal expected for a harmonic coupling at thermal equilibrium.
(B) Standard deviation and mean of xi - lambda over the replicas against the target.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import si_sweeps  # noqa: F401  (shared style)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data', 'deca_ala_1D')

hist = pd.read_csv(os.path.join(DATA, 'deca_ala_1D_xi_minus_lambda_hist.csv'))
stats = pd.read_csv(os.path.join(DATA, 'deca_ala_1D_xi_minus_lambda_stats.csv'))
hist = hist[hist.param == 'extFluc']
stats = stats[stats.param == 'extFluc']

W, H = 3.33, 4.6
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([0.62 / W, 2.75 / H, 2.55 / W, 1.55 / H])
axB = fig.add_axes([0.62 / W, 0.50 / H, 2.55 / W, 1.55 / H])

# ---------------------------------------------------------------- A: scaled distributions
grid = np.linspace(-6, 3, 400)
vals = sorted(hist.value.unique())
cmap = plt.get_cmap('viridis')
for i, v in enumerate(vals):
    c = cmap(0.92 * i / (len(vals) - 1))
    dens = []
    for _, g in hist[hist.value == v].groupby('seed'):
        x = g.bin_center.to_numpy() / v
        dx = np.diff(x).mean()
        p = g['count'].to_numpy() / (g['count'].sum() * dx)
        dens.append(np.interp(grid, x, p, left=0, right=0))
    axA.plot(grid, np.mean(dens, axis=0), color=c, lw=1.0, label=f'{v:g}')
axA.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color='k', lw=0.8, ls='--',
         label='Normal')
axA.set_xlim(-6, 3)
axA.set_ylim(0, 0.45)
axA.set_xlabel(r'($\xi-\lambda$) / extendedFluctuation')
axA.set_ylabel('Probability density')
axA.legend(loc='upper left', fontsize=6, ncol=2, handlelength=1.0, handletextpad=0.3,
           columnspacing=0.6, labelspacing=0.15, title='extendedFluctuation (Å)',
           title_fontsize=6)

# ---------------------------------------------------------------- B: SD and mean vs target
s = stats.groupby('value').agg(sd=('std', 'mean'), sd_lo=('std', 'min'), sd_hi=('std', 'max'),
                               mu=('mean', 'mean'))
v = s.index.to_numpy()
axB.plot([0.005, 10], [0.005, 10], color='0.6', lw=0.6, zorder=0)
axB.plot(v, s.sd, 'o-', color=si_sweeps.C_MED, ms=3, lw=1.0, label='SD')
axB.plot(v, -s.mu, 's-', color=si_sweeps.C_FEW, ms=3, lw=1.0, label='−Mean')
axB.set_xscale('log')
axB.set_yscale('log')
axB.set_xlim(0.007, 3)
axB.set_ylim(1e-4, 10)
fmt = mpl.ticker.FuncFormatter(lambda t, _: f'{t:g}')
axB.xaxis.set_major_formatter(fmt)
axB.yaxis.set_major_formatter(fmt)
axB.set_xlabel('extendedFluctuation (Å)')
axB.set_ylabel(r'$\xi-\lambda$ (Å)')
axB.legend(loc='upper left', fontsize=7, handlelength=1.5)

for t, ax in zip('AB', (axA, axB)):
    p = ax.get_position()
    fig.text(0.02, p.y1 + 0.03 / H, t, fontsize=12, fontweight='bold', ha='left',
             va='bottom')

out = os.path.join(ROOT, 'Figures', 'SI', 'FigS_deca_coupling.pdf')
fig.savefig(out)
fig.savefig(os.path.join(ROOT, 'build', 'FigS_deca_coupling.png'), dpi=200)
print('wrote', out)
