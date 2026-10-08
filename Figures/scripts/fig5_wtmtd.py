"""Figure 5: standalone WT-MtD compared with WTM-eABF for deca-alanine.

Run from the repository root:
    python Figures/scripts/fig5_wtmtd.py
Writes Figures/Main/Fig5_WT-MtD.pdf (+ .png preview in build/).

WT-MtD: Data/deca_ala_WT-MtD/wtmtd_combined_convergence_per_seed.csv (5 replicas per value;
biasTemperature from the pure WT-MtD rerun). WTM-eABF: the deca-alanine sweeps of Figure 2,
Data/deca_ala_1D (10 replicas per value). Both: kT criterion against the WTM-eABF median
reference, time = (frame + 1) * 0.05 ns, statistics over converged replicas. Mean and SD are
shown where at least 3 replicas converged, the mean alone (orange) where 1 or 2 converged, and
a cross at the top of the panel where none converged.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')

# ---------------------------------------------------------------- style (as Figure 1)
for f in font_manager.findSystemFonts():
    if os.path.basename(f).lower().startswith('arial'):
        font_manager.fontManager.addfont(f)
mpl.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
})

C_EABF = '#1f5fa8'         # WTM-eABF (as Figure 2)
C_MTD = '#444444'          # WT-MtD
C_FEW = '#e08214'          # fewer than 3 converged replicas
SPACING = 0.05
YMAX = 10.0
MIN_N = 3

PARAMS = [
    ('MTDheight', 'hill\nWeight (kcal/mol)', [0, 0.5, 1]),
    ('MTDwidth', 'hill\nWidth (bins)', [0, 2, 4]),
    ('MTDnewhill', 'newHill\nFrequency (steps)', [1000, 2000, 3000]),
    ('MTDtemp', 'bias\nTemperature (10³ K)', [0, 10000, 20000, 30000]),
]


def stats(d, key):
    out = []
    for v, g in d[d.param == key].groupby('value'):
        t = ((g.frame.dropna() + 1) * SPACING).to_numpy()
        n = len(t)
        out.append((v, n, t.mean() if n else np.nan, t.std(ddof=1) if n >= MIN_N else np.nan))
    return pd.DataFrame(out, columns=['v', 'n', 'mean', 'sd'])


def curve(ax, s, color, ytop):
    ok = s[s.n >= MIN_N]
    ax.fill_between(ok.v, np.clip(ok['mean'] - ok.sd, 0, None), ok['mean'] + ok.sd,
                    color=color, alpha=0.20, lw=0)
    any_ = s[s.n > 0]
    ax.plot(any_.v, any_['mean'], color=color, lw=1.0, zorder=3)
    ax.plot(ok.v, ok['mean'], 'o', ms=2.6, color=color, zorder=4)
    few = any_[any_.n < MIN_N]
    ax.plot(few.v, few['mean'], 'o', ms=3.0, color=C_FEW, zorder=5)
    for _, r in s[s.n == 0].iterrows():
        ax.plot(r.v, ytop, 'x', ms=4, mew=1.0, color=color, clip_on=False, zorder=5)


eabf = pd.read_csv(os.path.join(DATA, 'deca_ala_1D', 'deca_ala_1D_convergence_per_seed.csv'))
mtd = pd.read_csv(os.path.join(DATA, 'deca_ala_WT-MtD', 'wtmtd_combined_convergence_per_seed.csv'))

# ---------------------------------------------------------------- layout
# Single column (JPCB, 3.33 in), 2 x 2, shared y axis, legend in a row across the top.
PW, PH = 1.28, 1.15
X0, DX = 0.42, 1.52
Y0, DY = 0.50, 1.72
W, H = 3.33, Y0 + DY + PH + 0.50
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


panels = []
for k, (key, label, ticks) in enumerate(PARAMS):
    row, col = divmod(k, 2)
    ax = axes(X0 + col * DX, Y0 + (1 - row) * DY, PW, PH)
    se, sm = stats(eabf, key), stats(mtd, key)
    curve(ax, se, C_EABF, YMAX * 0.97)
    curve(ax, sm, C_MTD, YMAX * 0.90)
    v = pd.concat([se.v, sm.v])
    pad = 0.05 * (v.max() - v.min())
    ax.set_xlim(min(v.min(), ticks[0]) - pad, v.max() + pad)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t / 1000:g}' if max(ticks) >= 5000 else f'{t:g}' for t in ticks])
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel(label, labelpad=1.5, fontsize=8, linespacing=1.05)
    if col:
        ax.set_yticklabels([])
    panels.append(ax)
fig.text(0.0, (Y0 + (DY + PH) / 2) / H, 'Convergence time (ns)', rotation=90, ha='left',
         va='center')

handles = [Line2D([], [], color=C_EABF, lw=1.0, marker='o', ms=2.6, label='WTM-eABF'),
           Line2D([], [], color=C_MTD, lw=1.0, marker='o', ms=2.6, label='WT-MtD'),
           Patch(color='#888888', alpha=0.35, lw=0, label=r'$\pm$1 SD'),
           Line2D([], [], color=C_FEW, lw=0, marker='o', ms=3.0, label='< 3 conv.'),
           Line2D([], [], color='black', lw=0, marker='x', ms=4, mew=1.0, label='None conv.')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
           handlelength=1.4, handletextpad=0.4, columnspacing=0.9, fontsize=7)

for t, ax in zip('ABCD', panels):
    p = ax.get_position()
    fig.text(p.x0 - (0.21 if p.x0 * W < 1 else 0.10) / W, (p.y1 * H + 0.03) / H, t,
             fontsize=12, fontweight='bold', ha='left', va='bottom')

out = os.path.join(ROOT, 'Figures', 'Main', 'Fig5_WT-MtD.pdf')
fig.savefig(out)
fig.savefig(os.path.join(ROOT, 'build', 'Fig5_WT-MtD.png'), dpi=200)
print('wrote', out)
