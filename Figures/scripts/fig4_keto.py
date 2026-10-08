"""Figure 4: final-state accuracy and reproducibility of ketoprofen over the second-stage grid.

Run from the repository root:
    python Figures/scripts/fig4_keto.py
Writes Figures/Main/Fig4_keto.pdf (+ .png preview in build/).

Data: Data/ketoprofen/keto_cells.csv (export_keto_data.py). Accuracy = RMSD of the final PMF of
each replica to the 3.4 us reference, mean (and SD) over the five replicas of a cell.
Reproducibility = mean pairwise RMSD between the final PMFs of the five replicas. Both panels
are RMSDs in kcal/mol and share one colour scale. Layout and orientation as Figure 3
(rows = fullSamples, columns = extendedFluctuation).
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data', 'ketoprofen')

# ---------------------------------------------------------------- style (as Figure 3)
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
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
})

FS = [500, 2000, 5000, 10000]
EF = [0.05, 0.1, 0.2, 0.5]
VMIN, VMAX = 0.0, 0.5
CMAP = 'viridis'

d = pd.read_csv(os.path.join(DATA, 'keto_cells.csv'))
d = d[d.stage == 2]


def grid(col):
    Z = np.full((len(FS), len(EF)), np.nan)
    for i, fs in enumerate(FS):
        for j, ef in enumerate(EF):
            r = d[(d.fullSamples == fs) & np.isclose(d.extFluc, ef)]
            assert len(r) == 1 and r.n.iloc[0] == 5
            Z[i, j] = r[col].iloc[0]
    return Z


acc, acc_sd, rep = grid('acc_mean'), grid('acc_sd'), grid('repro_mean_pairwise')
cm = mpl.colormaps[CMAP]


def heatmap(ax, Z, sub=None):
    m = ax.pcolormesh(Z, cmap=cm, vmin=VMIN, vmax=VMAX, edgecolors='white', linewidth=0.6)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            r, g, b, _ = cm(m.norm(Z[i, j]))
            col = 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else 'white'
            if sub is None:
                ax.text(j + 0.5, i + 0.5, f'{Z[i, j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color=col)
            else:
                ax.text(j + 0.5, i + 0.62, f'{Z[i, j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color=col)
                ax.text(j + 0.5, i + 0.30, f'±{sub[i, j]:.2f}', ha='center', va='center',
                        fontsize=5, color=col)
    ax.set_xticks(np.arange(len(EF)) + 0.5)
    ax.set_xticklabels([f'{v:g}' for v in EF])
    ax.set_yticks(np.arange(len(FS)) + 0.5)
    ax.set_yticklabels([str(v) for v in FS])
    ax.tick_params(length=0, pad=2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_aspect('equal')
    return m


# ---------------------------------------------------------------- layout
# Single column (3.33 in), two maps side by side, one shared horizontal colour bar on top.
SIDE = 1.18
X0 = 0.77
X1 = X0 + SIDE + 0.10
Y0 = 0.36
CB_Y = Y0 + SIDE + 0.30
W, H = 3.33, CB_Y + 0.06 + 0.36
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


axA = axes(X0, Y0, SIDE, SIDE)
axB = axes(X1, Y0, SIDE, SIDE)
m = heatmap(axA, acc, acc_sd)
heatmap(axB, rep)
axA.set_ylabel('fullSamples')
axB.set_yticklabels([])
for ax, title in ((axA, 'Accuracy'), (axB, 'Reproducibility')):
    ax.set_title(title, fontsize=8, pad=3)

cax = axes(X0, CB_Y, X1 + SIDE - X0, 0.06)
cb = fig.colorbar(m, cax=cax, orientation='horizontal')
cb.set_label('RMSD (kcal/mol)', labelpad=2)
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')
cb.outline.set_linewidth(0.6)
cb.ax.tick_params(width=0.6, length=2, pad=1)

fig.text((X0 + X1 + SIDE) / 2 / W, 0.02 / H, 'extendedFluctuation (Å)', ha='center',
         va='bottom')

for t, ax in zip('AB', (axA, axB)):
    p = ax.get_position()
    fig.text(p.x0, (p.y1 * H + 0.02) / H, t, fontsize=12, fontweight='bold',
             ha='left', va='bottom')

out = os.path.join(ROOT, 'Figures', 'Main', 'Fig4_keto.pdf')
fig.savefig(out)
fig.savefig(os.path.join(ROOT, 'build', 'Fig4_keto.png'), dpi=200)
print('wrote', out)
