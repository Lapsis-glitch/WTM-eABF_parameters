"""Figure 3: fullSamples x extendedFluctuation sweeps for deca-alanine and ethanol.

Run from the repository root:
    python Figures/scripts/fig3_2d_sweeps.py
Writes Figures/Main/Fig3_2D_sweeps.pdf (+ .png preview in build/).

Convergence frames are the per-seed CSVs in Data/ (WTM-eABF_parameters @ 2f5bbe7, kT
criterion against reference_median.pmf). Time = (frame + 1) * frame spacing. Each cell shows
the mean (A, C) or standard deviation (B, D) over the converged replicas, with the number of
converged replicas when fewer than all of them converged. The grid values are not evenly
spaced, so cells are drawn on an index grid and nothing is interpolated between them.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
})

FS = [100, 500, 2000, 5000, 10000]
EF = [0.01, 0.1, 0.2, 0.5, 2.0]
NREP = 10
NONE = '#d9d9d9'          # cell without a value (SD of a single replica)


def load_deca():
    d = pd.read_csv(os.path.join(DATA, 'deca_ala_2D', 'deca_ala_2D_convergence_per_seed.csv'))
    v = d['value'].str.split('_', expand=True).astype(float)
    return pd.DataFrame({'fs': v[0], 'ef': v[1], 'frame': d['frame']}), 0.05


def load_ethanol():
    d = pd.read_csv(os.path.join(DATA, 'ethanol_2D', 'convergence_per_seed.csv'))
    return pd.DataFrame({'fs': d['fullSamples'], 'ef': d['extendedFluctuation'],
                         'frame': d['frame']}), 0.2


def grids(d, spacing):
    """Mean, SD (ddof 1) and number of converged replicas, rows = fullSamples, cols = extFluc."""
    t = (d['frame'] + 1) * spacing
    shape = (len(FS), len(EF))
    mean, sd, n = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, int)
    for i, fs in enumerate(FS):
        for j, ef in enumerate(EF):
            x = t[np.isclose(d['ef'], ef) & (d['fs'] == fs)].dropna()
            assert len(t[np.isclose(d['ef'], ef) & (d['fs'] == fs)]) == NREP
            n[i, j] = len(x)
            if len(x):
                mean[i, j] = x.mean()
            # No SD from fewer than three converged replicas (same rule as Figure 2).
            if len(x) >= 3:
                sd[i, j] = x.std(ddof=1)
    return mean, sd, n


def heatmap(ax, Z, n, cmap, vmax):
    cm = mpl.colormaps[cmap].copy()
    cm.set_bad(NONE)
    m = ax.pcolormesh(np.ma.masked_invalid(Z), cmap=cm, vmin=0, vmax=vmax,
                      edgecolors='white', linewidth=0.6)
    norm = m.norm
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                txt, col = '–', 'black'
            else:
                r, g, b, _ = cm(norm(Z[i, j]))
                col = 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else 'white'
                txt = f'{Z[i, j]:.1f}'
            y = i + 0.5
            if n[i, j] < NREP:
                ax.text(j + 0.5, y + 0.15, txt, ha='center', va='center', fontsize=6.5,
                        color=col)
                ax.text(j + 0.5, y - 0.24, f'{n[i, j]}/{NREP}', ha='center', va='center',
                        fontsize=5, color=col)
            else:
                ax.text(j + 0.5, y, txt, ha='center', va='center', fontsize=6.5, color=col)
    ax.set_xticks(np.arange(len(EF)) + 0.5)
    ax.set_xticklabels([f'{v:g}' for v in EF])
    ax.set_yticks(np.arange(len(FS)) + 0.5)
    ax.set_yticklabels([str(v) for v in FS])
    ax.tick_params(length=0, pad=2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel('extendedFluctuation (Å)')
    ax.set_ylabel('fullSamples')
    ax.set_aspect('equal')
    return m


def colorbar(m, cax, label):
    cb = fig.colorbar(m, cax=cax, orientation='horizontal')
    cb.set_label(label, labelpad=2)
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2, pad=1)


# ---------------------------------------------------------------- layout
# Single column (JPCB, 3.33 in). 2 x 2: rows deca-alanine, ethanol; columns mean, SD.
# Axes shared: fullSamples labels on the left column, extendedFluctuation labels on the
# bottom row. One horizontal colour bar per column, same scale for both systems.
VMAX = {'mean': 12, 'sd': 5}
SIDE = 1.18                       # map side (in)
X0 = 0.77
X1 = X0 + SIDE + 0.10
Y0 = 0.36
Y1 = Y0 + SIDE + 0.22
CB_Y = Y1 + SIDE + 0.22
W, H = 3.33, CB_Y + 0.06 + 0.36
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


panels = []
for (load, title), y in (((load_deca, 'Deca-alanine'), Y1), ((load_ethanol, 'Ethanol'), Y0)):
    d, spacing = load()
    mean, sd, n = grids(d, spacing)
    for x, Z, cmap, key in ((X0, mean, 'viridis', 'mean'), (X1, sd, 'magma', 'sd')):
        ax = axes(x, y, SIDE, SIDE)
        m = heatmap(ax, Z, n, cmap, VMAX[key])
        if y == Y1:
            colorbar(m, axes(x, CB_Y, SIDE, 0.06),
                     'Mean convergence time (ns)' if key == 'mean' else 'SD (ns)')
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if y == Y0:
            ax.set_xlabel('')
        if x == X1:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        panels.append(ax)
    fig.text(0.0, (y + SIDE / 2) / H, title, rotation=90, ha='left', va='center',
             fontweight='bold', fontsize=10)

fig.text((X0 + X1 + SIDE) / 2 / W, 0.02 / H, 'extendedFluctuation (Å)', ha='center',
         va='bottom')

# ---------------------------------------------------------------- panel letters
# Just above the top-left corner of each map
for t, ax in zip('ABCD', panels):
    p = ax.get_position()
    fig.text(p.x0, (p.y1 * H + 0.02) / H, t, fontsize=12, fontweight='bold',
             ha='left', va='bottom')

out = os.path.join(ROOT, 'Figures', 'Main', 'Fig3_2D_sweeps.pdf')
fig.savefig(out)
fig.savefig(os.path.join(ROOT, 'build', 'Fig3_2D_sweeps.png'), dpi=200)
print('wrote', out)
