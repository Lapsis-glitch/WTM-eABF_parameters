"""Figure 1: benchmark systems and their reference PMFs.

Run from the repository root:
    python Figures/scripts/fig1_benchmark_reference.py
Writes Figures/Main/Fig1_benchmark_reference.pdf (+ .png preview in build/).

Curves are the reference files in Data/ as written by reference_builder.py
(WTM-eABF_parameters @ 2f5bbe7), unchanged. Molecules are drawn as flat ball-and-stick
models (ballstick.py) from Figures/structures/ (written by prep_structures.py). The water
slab and the POPC bilayer are schematic.
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

import ballstick as bs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')

# ---------------------------------------------------------------- style
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

C_MED = '#1f5fa8'      # median / reference line
C_ALL = '#555555'      # average over all runs
C_FILT = '#e08214'     # average without outliers
C_REF = 'black'        # ketoprofen 3.4 us reference
LW = 1.2


def load(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, -1]


def pmf_panel(ax, d, xlabel):
    """Median, average over all runs and without outliers, with their error bands."""
    x, med = load(os.path.join(d, 'reference_median.pmf'))
    _, avg = load(os.path.join(d, 'reference_average_all.pmf'))
    _, avg_e = load(os.path.join(d, 'reference_average_all_err.pmf'))
    _, flt = load(os.path.join(d, 'reference_average_filtered.pmf'))
    _, flt_e = load(os.path.join(d, 'reference_average_filtered_err.pmf'))
    ax.fill_between(x, avg - avg_e, avg + avg_e, color=C_ALL, alpha=0.18, lw=0,
                    label='Error (all)')
    ax.fill_between(x, flt - flt_e, flt + flt_e, color=C_FILT, alpha=0.25, lw=0,
                    label='Error (filtered)')
    ax.plot(x, med, color=C_MED, lw=LW, label='Median')
    ax.plot(x, avg, color=C_ALL, lw=LW, ls=(0, (4, 2)), label='Average (all)')
    ax.plot(x, flt, color=C_FILT, lw=LW, ls=(0, (1, 1.5)), label='Average (filtered)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PMF (kcal/mol)')
    ax.set_xlim(x[0], x[-1])
    return ax


def nanma_map(ax, cax, path, cmap, cblabel, levels=None):
    a = np.loadtxt(path)
    phi, psi = np.unique(a[:, 0]), np.unique(a[:, 1])
    Z = a[:, 2].reshape(len(phi), len(psi)).T          # rows = psi
    m = ax.pcolormesh(phi, psi, Z, cmap=cmap, shading='nearest', rasterized=True)
    if levels is not None:
        ax.contour(phi, psi, Z, levels=levels, colors='white', linewidths=0.3, alpha=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ticks = [-180, 0, 180]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks([-90, 90], minor=True)
    ax.set_yticks([-90, 90], minor=True)
    ax.tick_params(which='minor', length=1.5, width=0.5)
    ax.set_xlabel(r'$φ$ (°)')
    ax.set_ylabel(r'$ψ$ (°)', labelpad=-10)   # tucked in next to the short '0' tick label
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)
    cb = fig.colorbar(m, cax=cax)
    cb.set_label(cblabel, labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2)
    return ax


# ---------------------------------------------------------------- system drawings
# Molecules in Å (ballstick.py). Water and lipids are flat cartoons in the same idiom:
# flat fills, dark outlines.
WATER = '#cfe2f3'
WATER_TOP = '#e6f0f9'
WATER_SIDE = '#b9d3eb'
WATER_EDGE = '#6f98bf'
CORE = '#f3ead6'               # membrane interior
CORE_SIDE = '#e6dabf'
HEAD = '#e8a33d'
TAIL = '#9c8a66'
HEAD_BACK = '#f2cf98'
TAIL_BACK = '#cbbd9e'
BOXLINE = dict(color='#555555', lw=0.8, solid_capstyle='round')
DEPTH = 0.16                   # back face offset as a fraction of the box width
# Panels E and G: both boxes are drawn with the same front-face width on the page, with
# their left edges aligned, and the z arrows at the same distance from the box (inches).
BOX_X0 = 3.32                  # page position of the front-left box edge
BOX_W = 0.78                   # front-face width
ARROW_GAP = 0.09               # box edge -> z arrow
CELL_MARGIN = 0.22             # box edge -> left edge of the axes (room for the z label)


def dim_arrow(ax, x, y0, y1, text, side='left', z=60, style='<|-|>'):
    """Arrow from y0 to y1 in data coordinates (double-headed by default) with an
    italic label beside it."""
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle=style, color='black',
                                 lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=7, zorder=z))
    ax.annotate(text, (x, 0.5 * (y0 + y1)), xytext=(-3 if side == 'left' else 3, 0),
                textcoords='offset points', ha='right' if side == 'left' else 'left',
                va='center', fontsize=9, style='italic', zorder=z)


def torsion(ax, p, bond, text, sign=1, z0=10):
    """Rotation arrow around a bond: an ellipse (circle seen edge-on) centred on the bond.
    The back half runs behind the molecule, so the bond hides it where they cross, and
    the front half with the arrowhead crosses over it. The label sits
    on the sign side. z0: zorder of the molecule (ballstick.draw)."""
    a, b = p[bond[0]], p[bond[1]]
    m = 0.5 * (a + b)
    u = (b - a) / np.hypot(*(b - a))
    n = np.array([-u[1], u[0]]) * sign

    def arc(t0, t1):
        t = np.linspace(t0, t1, 60)
        return m + np.outer(0.09 * np.cos(t), u) + np.outer(0.52 * np.sin(t), n)

    back = arc(0.85 * np.pi, 1.5 * np.pi)             # cos t < 0: behind the bond (270° in all)
    front = arc(1.5 * np.pi, 2.35 * np.pi)            # cos t > 0: in front
    ax.plot(*back.T, color='black', lw=0.8, solid_capstyle='round', zorder=z0 - 0.5)
    # Head drawn by hand so that its point is the end of the arc: the line stops at the
    # base of the triangle, which points along the chord from the base to the tip.
    tip, hl, hw = front[-1], 0.26, 0.24               # Å: tip, head length and width
    keep = np.hypot(*(front - tip).T) > hl
    line = front[:np.argmin(keep)] if not keep.all() else front
    d = (tip - line[-1]) / np.hypot(*(tip - line[-1]))
    base = tip - hl * d
    nn = np.array([-d[1], d[0]]) * hw / 2
    ax.plot(*np.r_[line, [base]].T, color='black', lw=0.8, solid_capstyle='round',
            zorder=60)
    ax.add_patch(Polygon([tip, base + nn, base - nn], closed=True, fc='black', ec='none',
                         zorder=60.1))
    ax.text(*(m - 1.1 * n), text, ha='center', va='center', fontsize=9, zorder=60)


def ring_frame(xyz, ring):
    """Principal axes of the ring atoms: ring face-on, widest direction along x."""
    c = xyz - xyz[ring].mean(0)
    v = np.linalg.eigh(np.cov(c[ring].T))[1][:, ::-1].T
    if np.linalg.det(v) < 0:
        v[2] *= -1
    return c @ v.T


def nanma_panels(a1, a2):
    """C7eq in the frame of the alanine (x along N->C, y towards CB). C7ax with its
    seven-membered hydrogen-bonded ring nearly face-on, tilted 35° so that the axial CB
    shows below CA. The
    C=O...H-N hydrogen bond that closes the ring is dotted."""
    for ax, name, label in ((a1, 'nanma_c7eq', r'C$_{7\mathrm{eq}}$'),
                            (a2, 'nanma_c7ax', r'C$_{7\mathrm{ax}}$')):
        m = bs.load(name)
        hn = [i + j - 8 for i, j in m['bonds'] if 8 in (i, j)
              and 'H' in (m['elements'][i], m['elements'][j])][0]
        if ax is a1:
            xyz = bs.frame(m['xyz'], 4, 3, 6, 5)
        else:
            xyz = bs.rotate(ring_frame(m['xyz'], [1, 2, 3, 4, 6, 8, hn]), x=-35)
        p, r = bs.draw(ax, m, xyz, fog=0.7, fog_depth=3.5)   # stronger fog, small molecule
        ax.plot(*p[[2, hn]].T, color='black', lw=1.0, ls=(0, (1, 1.2)), zorder=9.9)
        bs.limits(ax, p, r)
        ax.text(0.5, -0.02, label, transform=ax.transAxes, ha='center', va='top')
        if ax is a1:
            torsion(ax, p, m['phi_bond'], r'$φ$', sign=-1)
            torsion(ax, p, m['psi_bond'], r'$ψ$', sign=-1)


def cv_frame(m):
    """y along the CV vector (carbonyl C of Ala1 -> Ala10), x along the widest spread."""
    xyz = m['xyz']
    a, b = m['cv']
    y = (xyz[b] - xyz[a]) / np.linalg.norm(xyz[b] - xyz[a])
    c = xyz - xyz.mean(0)
    perp = c - np.outer(c @ y, y)
    x = np.linalg.eigh(np.cov(perp.T))[1][:, -1]
    x -= x.dot(y) * y
    x /= np.linalg.norm(x)
    return (xyz - xyz[a]) @ np.array([x, y, np.cross(x, y)]).T


DECA_HEAVY = os.environ.get('DECA_HEAVY', '1') == '1'   # heavy atoms only (user choice)


def spin_view(m, xyz):
    """Rotation about the vertical (CV) axis with the least disk overlap."""
    r = np.array([bs.RADIUS[e] for e in m['elements']])
    views = [bs.rotate(xyz, y=a) for a in range(0, 360, 5)]
    return min(views, key=lambda v: bs.clutter(v, r))


def deca_panels(x0, y0, height, gap=0.05):
    """Folded and unfolded deca-alanine at the same scale, d between the CV atoms.
    Creates the two axes (inches) and returns them."""
    mols = []
    for name in ('deca_folded', 'deca_unfolded'):
        m = bs.load(name)
        if DECA_HEAVY:
            m = bs.heavy(m)
        mols.append((m, spin_view(m, cv_frame(m))))
    rmax = max(bs.RADIUS.values())
    span = max(np.ptp(xyz[:, 1]) for _, xyz in mols) + 2 * rmax + 1.0
    s = height / span                    # inches per Å
    out, x = [], x0
    for m, xyz in mols:
        p0 = xyz[:, :2]
        xl, xr = p0[:, 0].min() - rmax - 0.2, p0[:, 0].max() + rmax + 0.9
        wid = xr + 3.0 - xl
        ax = axes(x, y0, wid * s, height)
        x += wid * s + gap
        p, r = bs.draw(ax, m, xyz)
        a, b = m['cv']
        for k in (a, b):
            ax.plot([p[k, 0], xr + 0.6], [p[k, 1]] * 2, color='black', lw=0.5,
                    ls=(0, (1.5, 1.5)), zorder=55)
        dim_arrow(ax, xr, p[a, 1], p[b, 1], 'd', side='right')
        yc = 0.5 * (p[:, 1].min() + p[:, 1].max())
        ax.set_ylim(yc - span / 2, yc + span / 2)
        ax.set_xlim(xl, xl + wid)
        ax.set_aspect('equal')
        ax.set_axis_off()
        out.append(ax)
    for ax, t in zip(out, ('folded', 'unfolded')):
        ax.text(0.5, 0.0, t, transform=ax.transAxes, ha='center', va='top')
    return out


def wave(x, L, amp, phase):
    return amp * (np.sin(4 * np.pi * x / L + phase) + 0.5 * np.sin(6 * np.pi * x / L + 1.7 * phase))


def cell_box(ax, w, h, dx, dy):
    """Wireframe box, hidden edges dashed and drawn behind the content."""
    FLB, FRB, FRT, FLT = (0, 0), (w, 0), (w, h), (0, h)
    BLB, BRB, BRT, BLT = [(x + dx, y + dy) for x, y in (FLB, FRB, FRT, FLT)]
    visible = [(FLB, FRB), (FRB, FRT), (FRT, FLT), (FLT, FLB),
               (BLT, BRT), (BRT, BRB), (FLT, BLT), (FRT, BRT), (FRB, BRB)]
    hidden = [(BLT, BLB), (BLB, BRB), (FLB, BLB)]
    for p0, p1 in visible:
        ax.plot(*zip(p0, p1), zorder=40, **BOXLINE)
    for p0, p1 in hidden:
        ax.plot(*zip(p0, p1), zorder=1, ls=(0, (3, 2)), **BOXLINE)


def block(ax, w, dx, dy, y0, y1, front, side, z=3):
    """Front and right faces of a horizontal layer between y0 and y1 (arrays or scalars)."""
    x = np.linspace(0, w, 200)
    y0 = np.broadcast_to(y0, x.shape)
    y1 = np.broadcast_to(y1, x.shape)
    ax.add_patch(Polygon(np.r_[np.c_[x, y1], np.c_[x, y0][::-1]], fc=front, ec='none',
                         zorder=z + 0.1))
    ax.add_patch(Polygon([(w, y0[-1]), (w + dx, y0[-1] + dy), (w + dx, y1[-1] + dy),
                          (w, y1[-1])], fc=side, ec='none', zorder=z))


def slab_cell(ax):
    """Ethanol above a water slab, in a box twice as tall as the slab (schematic)."""
    w = hs = 24.0
    s = BOX_W / w                        # inches per Å
    dx = dy = DEPTH * w
    h = 2 * hs
    x = np.linspace(0, w, 200)
    front = hs + wave(x, w, 0.6, 0.0)
    back = hs + wave(x, w, 0.6, 2.0) + dy
    block(ax, w, dx, dy, 0, front, WATER, WATER_SIDE)
    ax.add_patch(Polygon(np.r_[np.c_[x, front], np.c_[x + dx, back][::-1]],
                         fc=WATER_TOP, ec='none', zorder=2.5))   # free surface
    ax.plot(x, front, color=WATER_EDGE, lw=0.7, zorder=6)
    ax.plot(x + dx, back, color=WATER_EDGE, lw=0.7, zorder=2.6)
    cell_box(ax, w, h, dx, dy)
    m = bs.load('ethanol')
    xyz = bs.best_view(m, bs.principal(m['xyz']), tilt=60)
    bs.draw(ax, m, xyz, offset=(0.5 * (w + dx), hs + 9.0), scale=3.0, z0=20)
    dim_arrow(ax, -ARROW_GAP / s, h, 0, 'z', style='-|>')   # z along the box, top to bottom
    ax.text(0.5 * (w + dx), hs + 15.5, 'Ethanol', ha='center', va='bottom')
    ax.text(0.5 * (w + dx), h + dy + 1.0, 'Vacuum', ha='center', va='bottom')
    ax.text(0.5 * w, -1.5, 'Water slab', ha='center', va='top')
    ax.set_xlim(-CELL_MARGIN / s, w + dx + 0.5)
    ax.set_ylim(-5.5, h + dy + 6.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return s


def lipid(ax, x, y, up, rng, z, fog=0.0, clip=None):
    """Cartoon POPC: head bead and two wavy tails pointing to the bilayer centre.
    fog: fraction of white mixed in (depth cue for the rows behind)."""
    s = 1 if up else -1                  # +1: head at the top, tails pointing down
    rh, L = 2.1, 15.5
    t = np.linspace(0, 1, 40)
    for off in (-0.8, 0.8):
        yy = y - s * (rh * 0.5 + L * t)
        xx = x + off + 0.45 * np.sin(2 * np.pi * (2.2 * t + rng.uniform(0, 1)))
        ln, = ax.plot(xx, yy, color=bs.mix(TAIL, 'white', fog), lw=0.7,
                      solid_capstyle='round', zorder=z)
        ln.set_clip_path(clip)
    for rr, col in ((rh, bs.INK), (rh - 0.4, HEAD)):
        ax.add_patch(Circle((x, y), rr, fc=bs.mix(col, 'white', fog), ec='none',
                            zorder=z + 0.001, clip_path=clip))


def membrane_cell(ax):
    """Ketoprofen in water above a POPC bilayer (cartoon, real cell proportions)."""
    w, h = 64.5, 104.8
    s = BOX_W / w                        # inches per Å
    dx = dy = DEPTH * w
    c = 0.5 * h                          # bilayer centre (z = 0)
    zp = 19.5                            # head groups
    x = np.linspace(0, w, 200)
    up = c + zp + wave(x, w, 0.6, 0.4)
    lo = c - zp + wave(x, w, 0.6, 1.3)
    block(ax, w, dx, dy, up, h, WATER, WATER_SIDE)
    block(ax, w, dx, dy, lo, up, CORE, CORE_SIDE)
    block(ax, w, dx, dy, 0, lo, WATER, WATER_SIDE)
    ax.add_patch(Polygon([(0, h), (w, h), (w + dx, h + dy), (dx, h + dy)], fc=WATER_TOP,
                         ec='none', zorder=3))
    rng = np.random.default_rng(3)
    # Lipids fill the whole bilayer volume: rows along x, stacked in depth. A row at depth
    # fraction f (0 front, 1 back) is shifted by f*(dx, dy) on the page, drawn behind the
    # rows in front of it and faded towards white. Alternate rows are staggered.
    nx, nd = 13, 9
    sp = w / nx
    clip = Polygon([(0, 0), (w, 0), (w + dx, dy), (w + dx, h + dy), (dx, h + dy), (0, h)],
                   transform=ax.transData)       # box silhouette
    for top in (True, False):
        for row in range(nd - 1, -1, -1):
            f = row / (nd - 1)
            for k in range(nx):
                xl = (k + 0.5 + 0.5 * (row % 2)) * sp
                if xl > w:
                    continue
                yh = (c + zp if top else c - zp) + rng.uniform(-0.5, 0.5)
                lipid(ax, xl + f * dx, yh + f * dy, top, rng,
                      z=8 + 0.2 * (nd - 1 - row) + 0.005 * k, fog=0.55 * f, clip=clip)
    cell_box(ax, w, h, dx, dy)
    m = bs.load('ketoprofen')
    xyz = bs.best_view(m, bs.principal(m['xyz']), tilt=40)
    bs.draw(ax, m, xyz, offset=(0.5 * w, c + 43), scale=3.2, z0=20, fog_color=WATER)
    dim_arrow(ax, -ARROW_GAP / s, h, 0, 'z', style='-|>')   # z along the box, top to bottom
    ax.text(0.5 * w, -2.5, 'Ketoprofen in POPC', ha='center', va='top', gid='keto_label')
    ax.set_xlim(-CELL_MARGIN / s, w + dx + 1)
    ax.set_ylim(-9, h + dy + 1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return s


# ---------------------------------------------------------------- layout
# Full page width (JPCB double column, 7 in), placed at the top of a page.
# All positions in inches from the lower-left corner. Four columns: structure | data |
# structure | data, two rows: NANMA, ethanol (top) and deca-alanine, ketoprofen (bottom).
W, H = 7.0, 5.0
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


a1 = axes(0.0, 3.95, 1.05, 0.82)
a2 = axes(0.0, 2.85, 1.05, 0.82)
nanma_panels(a1, a2)

b = axes(1.55, 3.8, 0.95, 0.95)
bc = axes(2.57, 3.8, 0.06, 0.95)
nanma_map(b, bc, os.path.join(DATA, 'NANMA', 'reference_median.pmf'),
          'viridis', 'PMF (kcal/mol)', levels=np.arange(2, 22, 2))
c = axes(1.55, 2.7, 0.95, 0.95)
cc = axes(2.57, 2.7, 0.06, 0.95)
nanma_map(c, cc,
          os.path.join(DATA, 'NANMA', 'reference_average_all_err.pmf'),
          'magma', 'SD (kcal/mol)')
b.set_xlabel('')
b.set_xticklabels([])



def place_cell(ax, s, y0):
    """Size the box axes at s inches per Å, with data x = 0 (box edge) at BOX_X0."""
    (x0, x1), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    ax.set_position([(BOX_X0 + x0 * s) / W, y0 / H, (x1 - x0) * s / W, (yb - ya) * s / H])


f = axes(3.1, 2.7, 1.3, 2.05)
place_cell(f, slab_cell(f), 2.7)

g = axes(4.85, 2.7, 2.03, 2.05)
pmf_panel(g, os.path.join(DATA, 'ethanol_1D', 'reference_pmfs'), r'$z$ (Å)')
g.set_ylim(-0.5, 7.5)
g.set_xticks([0, 10, 20])

# Legend shared by the deca-alanine and ethanol PMFs, in the empty upper-left corner of D
handles, labels = g.get_legend_handles_labels()
order = [2, 3, 4, 0, 1]
g.legend([handles[j] for j in order], [labels[j] for j in order], loc='upper left',
         handlelength=2.2, borderaxespad=0.1, labelspacing=0.3)

# Bottom row: deca-alanine structures (E) and PMF (F) | ketoprofen system (G) and PMF (H)
d1, d2 = deca_panels(0.0, 0.45, 1.6, gap=0.0)
e = axes(1.65, 0.45, 1.3, 1.6)
pmf_panel(e, os.path.join(DATA, 'deca_ala_1D'), r'$d$ (Å)')
e.set_ylim(0, 40)
e.set_xticks([12, 17, 22, 27, 32])

h = axes(3.2, 0.42, 1.25, 1.6)
place_cell(h, membrane_cell(h), 0.45)
i = axes(4.85, 0.45, 2.03, 1.6)
k = np.loadtxt(os.path.join(DATA, 'ketoprofen', 'reference_pmfs.dat'))
z, ref, med, sd = k[:, 0], k[:, 1], k[:, 2], k[:, 5]
i.fill_between(z, med - sd, med + sd, color=C_MED, alpha=0.2, lw=0,
               label=r'Median $\pm$ 1 SD')
i.plot(z, med, color=C_MED, lw=LW, label='Median (80 runs)')
i.plot(z, ref, color=C_REF, lw=LW, ls=(0, (4, 2)), label=r'3.4 $\mu$s reference')
i.set_xlim(-40, 40)
i.set_xticks([-40, -20, 0, 20, 40])
i.set_ylim(-4.5, 4.2)
i.set_yticks([-4, -2, 0, 2])
i.set_xlabel(r'$z$ (Å)')
i.set_ylabel('PMF (kcal/mol)')
i.legend(loc='upper center', handlelength=2.4, borderaxespad=0.0, labelspacing=0.3)

# ---------------------------------------------------------------- align PMFs with the renders
# Each PMF panel spans the same height as the system depiction next to it (after the
# image axes have shrunk to their aspect ratio).
fig.canvas.draw()


def match_height(ax, *refs):
    boxes = [r.get_position() for r in refs]
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    p = ax.get_position()
    ax.set_position([p.x0, y0, p.width, y1 - y0])


match_height(g, f)
match_height(e, d1, d2)
match_height(i, h)

# 'Ketoprofen in POPC' on the same line as 'folded' and 'unfolded' (top of the text at
# the bottom edge of the deca-alanine axes)
fig.canvas.draw()
lab = [t for t in h.texts if t.get_gid() == 'keto_label'][0]
y = h.transData.inverted().transform(d2.transAxes.transform((0, 0)))[1]
lab.set_y(y)

# ---------------------------------------------------------------- panel letters
# Each letter sits at the top-left corner of its panel (including tick and axis
# labels); letters in the same row share the same height.
fig.canvas.draw()
r = fig.canvas.get_renderer()
dpi = fig.dpi


def bbox(*axs):
    bb = [ax.get_tightbbox(r) for ax in axs]
    return min(x.x0 for x in bb) / dpi, max(x.y1 for x in bb) / dpi


rows = [
    [('A', (a1, a2)), ('B', (b, bc, c, cc)), ('C', (f,)), ('D', (g,))],
    [('E', (d1, d2)), ('F', (e,)), ('G', (h,)), ('H', (i,))],
]
for row in rows:
    pos = {t: bbox(*axs) for t, axs in row}
    top = max(y for _, y in pos.values())
    for t, (x, _) in pos.items():
        fig.text(max(x, 0.03) / W, (top + 0.04) / H, t, fontsize=12, fontweight='bold',
                 ha='left', va='bottom')

out = os.path.join(ROOT, 'Figures', 'Main', 'Fig1_benchmark_reference.pdf')
fig.savefig(out)
print('wrote', out)
