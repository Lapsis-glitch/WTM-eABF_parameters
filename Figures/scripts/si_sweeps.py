"""SI figures: one-dimensional WTM-eABF sweeps (convergence time) and seed-resolved RMSD_ref.

Run from the repository root:
    python Figures/scripts/si_sweeps.py nanma
Writes Figures/SI/FigS_<system>_1D.pdf and Figures/SI/FigS_<system>_rmsd.pdf (+ .png previews
in build/).

Same conventions as Figure 2 (fig2_deca_1d.py): time = (frame + 1) * frame spacing, statistics
over the converged replicas only, values with fewer than 3 converged replicas show the mean only
(orange) without bands, values at which no replica converged are marked with an orange cross at
the top of the panel.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')
OUT = os.path.join(ROOT, 'Figures', 'SI')

# ---------------------------------------------------------------- style (as Figures 1 and 2)
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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
})

C_MED = '#1f5fa8'
C_FEW = '#e08214'
MIN_N = 3
KT = 0.592186869182        # threshold used by the analysis code (kcal/mol)

# Panel order as Figure 2: extended variable, ABF and grid, WT-MtD.
# key, axis label, ticks (tick labels divided by 1000 when the label says 10³)
DECA_PARAMS = [
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

SYSTEMS = {
    'nanma': dict(
        conv=('NANMA', 'nanma_convergence_per_seed.csv'),
        rmsd=('NANMA', 'nanma_rmsd_ref.csv'),
        spacing=0.5, nrep=5, ymax=50, yticks=[0, 10, 20, 30, 40, 50],
        tmax=50, rmin=0.01, rmax=10, ryticks=[0.01, 0.1, 1, 10],
        params=[
            ('extFluc', 'extended\nFluctuation (°)', [0, 2, 4]),
            ('extTime', 'extendedTime\nConstant (fs)', [0, 500, 1000]),
            ('extDamp', 'extendedLangevin\nDamping (ps$^{-1}$)', [0, 5, 10]),
            ('fullSamp', 'full\nSamples (10³)', [0, 2500, 5000]),
            ('colvarWidth', 'colvar\nWidth (°)', [0, 10, 20]),
            ('MTDheight', 'hill\nWeight (kcal/mol)', [0, 2, 4]),
            ('MTDwidth', 'hill\nWidth (bins)', [0, 5, 10]),
            ('MTDnewhill', 'newHill\nFrequency (10³)', [0, 2000, 4000]),
            ('MTDtemp', 'bias\nTemperature (10³ K)', [0, 5000, 10000]),
        ]),
    # Deca-alanine: sweep repeated at fullSamples = 5000 (convergence), and replica RMSD_ref
    # curves of the main sweep of Figure 2 (fullSamples = 500).
    'deca_fs5000': dict(
        conv=('deca_ala_fullSamp5000', 'convergence_per_seed.csv'),
        spacing=0.05, nrep=10, ymax=10, yticks=[0, 2, 4, 6, 8, 10], params=DECA_PARAMS),
    'ethanol': dict(
        conv=('ethanol_1D', 'convergence_per_seed.csv'),
        rmsd=('ethanol_1D', 'rmsd_ref_timeseries.csv'),
        spacing=0.2, nrep=5, ymax=16, yticks=[0, 4, 8, 12, 16],
        tmax=20, rmin=0.05, rmax=10, ryticks=[0.1, 1, 10],
        params=[p if p[0] != 'MTDnewhill' else
                ('MTDnewhill', 'newHill\nFrequency (steps)', [0, 1500, 3000])
                for p in DECA_PARAMS]),
    'deca': dict(
        rmsd=('deca_ala_1D', 'deca_ala_1D_rmsd_ref.csv'),
        tmax=10, rmin=0.1, rmax=20, ryticks=[0.1, 1, 10], params=DECA_PARAMS),
}

NAMES = {'extFluc': 'extendedFluctuation', 'extTime': 'extendedTimeConstant',
         'extDamp': 'extendedLangevinDamping', 'fullSamp': 'fullSamples',
         'colvarWidth': 'colvarWidth', 'MTDheight': 'hillWeight', 'MTDwidth': 'hillWidth',
         'MTDnewhill': 'newHillFrequency', 'MTDtemp': 'biasTemperature'}


def tick_label(t, label):
    # Labels that say 10³ show the tick values divided by 1000.
    return f'{t / 1000:g}' if '10³' in label else f'{t:g}'


def stats(d, key, spacing, nrep):
    out = []
    for v, g in d[d.param == key].groupby('value'):
        assert len(g) == nrep, (key, v, len(g))
        t = ((g.frame.dropna() + 1) * spacing).to_numpy()
        n = len(t)
        out.append((v, n, t.mean() if n else np.nan,
                    *(np.r_[t.std(ddof=1), t.min(), t.max()] if n >= MIN_N else [np.nan] * 3)))
    return pd.DataFrame(out, columns=['v', 'n', 'mean', 'sd', 'min', 'max'])


def conv_panel(ax, s, label, ticks, ymax, yticks):
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
        ax.plot(r.v, ymax * 0.96, 'x', ms=4, mew=1.0, color=C_FEW, clip_on=False, zorder=5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_label(t, label) for t in ticks])
    lo, hi = s.v.min(), s.v.max()
    pad = 0.06 * (hi - lo)
    ax.set_xlim(min(lo - pad, ticks[0] - pad * 0.3), hi + pad)
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_xlabel(label, labelpad=1.5, fontsize=8, linespacing=1.05)


def letters(fig, panels, W, H):
    for t, ax in zip('ABCDEFGHI', panels):
        p = ax.get_position()
        fig.text(p.x0 - (0.21 if p.x0 * W < 1 else 0.10) / W, (p.y1 * H + 0.03) / H, t,
                 fontsize=12, fontweight='bold', ha='left', va='bottom')


def convergence_figure(name, cfg):
    sub, fn = cfg['conv']
    d = pd.read_csv(os.path.join(DATA, sub, fn))
    PW, PH = 0.86, 0.86
    X0, DX = 0.40, 0.98
    Y0, DY = 0.50, 1.50
    W, H = 3.33, Y0 + 2 * DY + PH + 0.52
    fig = plt.figure(figsize=(W, H))
    panels = []
    for k, (key, label, ticks) in enumerate(cfg['params']):
        row, col = divmod(k, 3)
        ax = fig.add_axes([(X0 + col * DX) / W, (Y0 + (2 - row) * DY) / H, PW / W, PH / H])
        conv_panel(ax, stats(d, key, cfg['spacing'], cfg['nrep']), label, ticks,
                   cfg['ymax'], cfg['yticks'])
        if col:
            ax.set_yticklabels([])
        panels.append(ax)
    fig.text(0.0, (Y0 + DY + PH / 2) / H, 'Convergence time (ns)', rotation=90, ha='left',
             va='center')
    handles = [Line2D([], [], color=C_MED, lw=1.0, marker='o', ms=2.6, label='Mean'),
               Patch(color=C_MED, alpha=0.32, lw=0, label=r'$\pm$1 SD'),
               Patch(color=C_MED, alpha=0.10, lw=0, label='Min–max'),
               Line2D([], [], color=C_FEW, lw=0, marker='o', ms=3.0, label='< 3 conv.'),
               Line2D([], [], color=C_FEW, lw=0, marker='x', ms=4, mew=1.0,
                      label='None conv.')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
               handlelength=1.2, handletextpad=0.4, columnspacing=0.8, fontsize=7)
    letters(fig, panels, W, H)
    save(fig, f'FigS_{name}_1D')


def rmsd_figure(name, cfg):
    sub, fn = cfg['rmsd']
    d = pd.read_csv(os.path.join(DATA, sub, fn))
    # Legends sit to the right of each panel, outside the data.
    PW, PH = 1.45, 1.30
    X0, DX = 0.55, 2.15
    Y0, DY = 0.45, 1.75
    W, H = X0 + 2 * DX + PW + 0.6, Y0 + 2 * DY + PH + 0.25
    fig = plt.figure(figsize=(W, H))
    cmap = plt.get_cmap('viridis')
    panels = []
    for k, (key, label, _) in enumerate(cfg['params']):
        row, col = divmod(k, 3)
        ax = fig.add_axes([(X0 + col * DX) / W, (Y0 + (2 - row) * DY) / H, PW / W, PH / H])
        sel = d[d.param == key]
        vals = sorted(sel.value.unique(), key=float)
        for i, v in enumerate(vals):
            c = cmap(0.92 * i / max(len(vals) - 1, 1))
            sv = sel[sel.value == v]
            # Individual replicas faint, median over replicas solid.
            for _, g in sv.groupby('seed'):
                ax.plot(g.time_ns, g.rmsd, color=c, lw=0.3, alpha=0.25)
            med = sv.groupby('time_ns').rmsd.median()
            ax.plot(med.index, med.values, color=c, lw=1.1, label=f'{float(v):g}')
        ax.axhline(KT, color='k', lw=0.7, ls='--', zorder=5)
        ax.set_xlim(0, cfg['tmax'])
        ax.set_xticks(np.linspace(0, cfg['tmax'], 6))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        ax.set_yscale('log')
        ax.set_ylim(cfg['rmin'], cfg['rmax'])
        ax.set_title(NAMES[key], fontsize=8, pad=2)
        ax.set_yticks(cfg['ryticks'])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02), fontsize=6, ncol=1,
                  handlelength=1.0, handletextpad=0.3, labelspacing=0.15, borderaxespad=0.2)
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (ns)')
        if col == 0:
            ax.set_ylabel(r'RMSD$_\mathrm{ref}$ (kcal/mol)')
        panels.append(ax)
    letters(fig, panels, W, H)
    save(fig, f'FigS_{name}_rmsd')


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, stem + '.pdf')
    fig.savefig(out)
    fig.savefig(os.path.join(ROOT, 'build', stem + '.png'), dpi=200)
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    for name in sys.argv[1:] or SYSTEMS:
        if 'conv' in SYSTEMS[name]:
            convergence_figure(name, SYSTEMS[name])
        if 'rmsd' in SYSTEMS[name]:
            rmsd_figure(name, SYSTEMS[name])
