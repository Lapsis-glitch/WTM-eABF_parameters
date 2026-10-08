"""Figure 2: one-dimensional parameter sweeps for deca-alanine.

Run from the repository root:
    python Figures/scripts/fig2_deca_1d.py
Writes Figures/Main/Fig2_deca_1D.pdf (+ .png preview in build/).

Convergence frames are Data/deca_ala_1D/deca_ala_1D_convergence_per_seed.csv
(WTM-eABF_parameters @ 2f5bbe7, kT criterion against reference_median.pmf), 10 replicas per
value. Time = (frame + 1) * 0.05 ns. Statistics over the converged replicas only. Values with
fewer than 3 converged replicas show the mean only (orange), without bands, values at which no replica converged are marked
at the top of the panel.
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

C_MED = '#1f5fa8'          # mean (as the median line of Figure 1)
C_FEW = '#e08214'          # fewer than 3 converged replicas (as Figure 1 orange)
SPACING = 0.05             # ns per history frame
NREP = 10
YMAX = 10.0

# key in the CSV, axis label (camelCase split at the word boundary), ticks. Order: extended
# variable (A-C), ABF and grid (D, E), WT-MtD (F-I).
PARAMS = [
    ('extFluc', 'extended\nFluctuation (Å)', [0, 1, 2]),
    ('extTime', 'extendedTime\nConstant (fs)', [0, 200, 400]),
    ('extDamp', 'extendedLangevin\nDamping (ps$^{-1}$)', [0, 2, 4]),
    ('fullSamp', 'full\nSamples (10³)', [0, 5000, 10000]),
    ('colvarWidth', 'colvar\nWidth (Å)', [0, 1, 2]),
    ('MTDheight', 'hill\nWeight (kcal/mol)', [0, 0.5, 1]),
    ('MTDwidth', 'hill\nWidth (bins)', [0, 2, 4]),
    ('MTDnewhill', 'newHill\nFrequency (steps)', [1000, 2000, 3000]),
    ('MTDtemp', 'bias\nTemperature (10³ K)', [0, 5000, 10000, 15000]),
]


def tick_label(v):
    return f'{v:g}'


MIN_N = 3                  # fewer converged replicas: shown individually, no statistics


def stats(d, key):
    out = []
    for v, g in d[d.param == key].groupby('value'):
        assert len(g) == NREP
        t = ((g.frame.dropna() + 1) * SPACING).to_numpy()
        n = len(t)
        out.append((v, n, t.mean() if n else np.nan,
                    *(np.r_[t.std(ddof=1), t.min(), t.max()] if n >= MIN_N else [np.nan] * 3)))
    return pd.DataFrame(out, columns=['v', 'n', 'mean', 'sd', 'min', 'max'])


def panel(ax, s, label, ticks):
    # Bands only where at least MIN_N replicas converged, the mean line through every value
    # with at least one converged replica.
    ok = s[s.n >= MIN_N]
    ax.fill_between(ok.v, ok['min'], ok['max'], color=C_MED, alpha=0.10, lw=0)
    ax.fill_between(ok.v, np.clip(ok['mean'] - ok.sd, 0, None), ok['mean'] + ok.sd,
                    color=C_MED, alpha=0.22, lw=0)
    any_ = s[s.n > 0]
    ax.plot(any_.v, any_['mean'], color=C_MED, lw=1.0, zorder=3)
    ax.plot(ok.v, ok['mean'], 'o', ms=2.6, color=C_MED, zorder=4)
    few = any_[any_.n < MIN_N]
    ax.plot(few.v, few['mean'], 'o', ms=3.0, color=C_FEW, zorder=5)
    for _, r in s[s.n == 0].iterrows():
        ax.plot(r.v, YMAX * 0.96, 'x', ms=4, mew=1.0, color=C_FEW, clip_on=False, zorder=5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_label(t / 1000 if max(ticks) >= 5000 else t) for t in ticks])
    lo, hi = s.v.min(), s.v.max()
    pad = 0.06 * (hi - lo)
    ax.set_xlim(min(lo - pad, ticks[0] - pad * 0.3), hi + pad)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel(label, labelpad=1.5, fontsize=8, linespacing=1.05)


# ---------------------------------------------------------------- layout
# Single column (JPCB, 3.33 in), 3 x 3, shared y axis, legend in a row across the top.
d = pd.read_csv(os.path.join(DATA, 'deca_ala_1D', 'deca_ala_1D_convergence_per_seed.csv'))
PW, PH = 0.86, 0.86
X0, DX = 0.40, 0.98
Y0, DY = 0.50, 1.50
W, H = 3.33, Y0 + 2 * DY + PH + 0.52
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


panels = []
for k, (key, label, ticks) in enumerate(PARAMS):
    row, col = divmod(k, 3)
    ax = axes(X0 + col * DX, Y0 + (2 - row) * DY, PW, PH)
    panel(ax, stats(d, key), label, ticks)
    if col:
        ax.set_yticklabels([])
    panels.append(ax)
fig.text(0.0, (Y0 + DY + PH / 2) / H, 'Convergence time (ns)', rotation=90, ha='left',
         va='center')

handles = [Line2D([], [], color=C_MED, lw=1.0, marker='o', ms=2.6, label='Mean'),
           Patch(color=C_MED, alpha=0.32, lw=0, label=r'$\pm$1 SD'),
           Patch(color=C_MED, alpha=0.10, lw=0, label='Min–max'),
           Line2D([], [], color=C_FEW, lw=0, marker='o', ms=3.0, label='< 3 conv.'),
           Line2D([], [], color=C_FEW, lw=0, marker='x', ms=4, mew=1.0, label='None conv.')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
           handlelength=1.2, handletextpad=0.4, columnspacing=0.8, fontsize=7)

# ---------------------------------------------------------------- panel letters
for t, ax in zip('ABCDEFGHI', panels):
    p = ax.get_position()
    fig.text(p.x0 - (0.21 if p.x0 * W < 1 else 0.10) / W, (p.y1 * H + 0.03) / H, t,
             fontsize=12, fontweight='bold', ha='left', va='bottom')

out = os.path.join(ROOT, 'Figures', 'Main', 'Fig2_deca_1D.pdf')
fig.savefig(out)
fig.savefig(os.path.join(ROOT, 'build', 'Fig2_deca_1D.png'), dpi=200)
print('wrote', out)
