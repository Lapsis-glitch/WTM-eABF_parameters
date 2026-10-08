"""Flat ball-and-stick drawings in matplotlib (vector output).

Atoms are flat disks with a dark outline, bonds are plain dark sticks, all sized in Å so
the drawing scales with the axes. Atoms are painted back to front, each one after its
bonds to atoms behind it. A bond to an atom behind starts at the edge of that atom disk.
Coordinates come from Figures/structures/*.json (written by prep_structures.py).
"""

import json
import os

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(os.path.dirname(HERE), 'structures')

# Palette and proportions sampled from the style reference
COLOR = {'C': '#8c8c8c', 'N': '#3e59ca', 'O': '#c24938', 'H': '#ffffff', 'P': '#e08a2c'}
INK = '#141416'
RADIUS = {'C': 0.50, 'N': 0.50, 'O': 0.50, 'P': 0.55, 'H': 0.31}   # Å
OUTLINE = 0.07      # Å, outline thickness
BOND = 0.13         # Å, bond stick width
FOG = 0.5           # fraction of fog colour mixed in at the back of the molecule
FOG_DEPTH = 5.0     # Å, minimum depth over which the fog builds up


def mix(c, fog_color, f):
    return tuple((1 - f) * np.array(to_rgb(c)) + f * np.array(to_rgb(fog_color)))


def load(name):
    with open(os.path.join(STRUCT, name + '.json')) as f:
        d = json.load(f)
    d['xyz'] = np.array(d['xyz'], float)
    return d


def frame(xyz, origin, xa, xb, yatom):
    """Coordinates in a frame with x along xa->xb, y towards yatom, z to the viewer."""
    x = xyz[xb] - xyz[xa]
    x /= np.linalg.norm(x)
    y = xyz[yatom] - xyz[origin]
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    R = np.array([x, y, np.cross(x, y)])
    return (xyz - xyz[origin]) @ R.T


def principal(xyz, order=(0, 1, 2)):
    """Coordinates on the principal axes: order[0] = largest-variance axis to x, etc."""
    c = xyz - xyz.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    v = v[:, ::-1]
    R = np.zeros((3, 3))
    for k, o in enumerate(order):
        R[o] = v[:, k]
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return c @ R.T


def rotate(xyz, x=0, y=0, z=0):
    """Rotate about the x, y then z axes (degrees)."""
    out = xyz.copy()
    for ax, a in ((0, x), (1, y), (2, z)):
        if a:
            a = np.radians(a)
            c, s = np.cos(a), np.sin(a)
            i, j = [k for k in range(3) if k != ax]
            R = np.eye(3)
            R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
            out = out @ R.T
    return out


def clutter(xyz, r):
    """Overlap of the projected disks (sum of squared overlaps over all atom pairs)."""
    d = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
    o = np.clip(r[:, None] + r[None] - d, 0, None)
    np.fill_diagonal(o, 0)
    return (o ** 2).sum()


def best_view(mol, xyz, tilt=50, step=5, radius=None):
    """Tilt xyz about x and y (within +-tilt degrees) to minimise disk overlap.

    The in-plane orientation of the starting frame is kept as much as possible.
    """
    rad = dict(RADIUS, **(radius or {}))
    r = np.array([rad[e] for e in mol['elements']])
    best = None
    for ax in np.arange(-tilt, tilt + 1, step):
        for ay in np.arange(-tilt, tilt + 1, step):
            v = rotate(xyz, x=ax, y=ay)
            c = clutter(v, r) * (1 + 1e-4 * (ax ** 2 + ay ** 2))
            if best is None or c < best[0]:
                best = (c, v, ax, ay)
    return best[1]


def heavy(mol):
    """Copy of mol without hydrogens."""
    keep = [i for i, e in enumerate(mol['elements']) if e != 'H']
    new = {k: v for k, v in mol.items()}
    idx = {o: n for n, o in enumerate(keep)}
    new['elements'] = [mol['elements'][i] for i in keep]
    new['xyz'] = mol['xyz'][keep]
    new['bonds'] = [[idx[i], idx[j]] for i, j in mol['bonds'] if i in idx and j in idx]
    for key in ('cv', 'phi_bond', 'psi_bond'):
        if key in mol:
            new[key] = [idx[i] for i in mol[key]]
    return new


def stick(ax, p0, p1, w, z, color=INK):
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-6:
        return
    n = np.array([-d[1], d[0]]) / L * w / 2
    ax.add_patch(Polygon([p0 + n, p1 + n, p1 - n, p0 - n], closed=True, fc=color,
                         ec='none', zorder=z))


def draw(ax, mol, xyz=None, offset=(0, 0), scale=1.0, z0=10, radius=None, bond=BOND,
         outline=OUTLINE, fog=FOG, fog_color='white', fog_depth=FOG_DEPTH):
    """Draw mol on ax with 2D offset (Å) and scale. Returns projected 2D coordinates.

    Depth fog: atoms and bonds fade linearly towards fog_color with distance from the
    frontmost atom, reaching the fraction fog at the back (or at FOG_DEPTH for flat
    molecules, fog_depth in Å)."""
    xyz = mol['xyz'] if xyz is None else xyz
    el = mol['elements']
    rad = dict(RADIUS, **(radius or {}))
    p = xyz[:, :2] * scale + np.asarray(offset)
    r = np.array([rad[e] for e in el]) * scale
    nbr = {i: [] for i in range(len(el))}
    for i, j in mol['bonds']:
        nbr[i].append(j)
        nbr[j].append(i)
    order = np.argsort(xyz[:, 2], kind='stable')
    rank = np.empty(len(el), int)
    rank[order] = np.arange(len(el))
    zstep = 1.0 / (3 * len(el))
    zf = xyz[:, 2].max()
    depth = fog * (zf - xyz[:, 2]) / max(np.ptp(xyz[:, 2]), fog_depth)
    for n, i in enumerate(order):
        z = z0 + n * 3 * zstep
        for j in nbr[i]:
            if rank[j] < rank[i]:
                d = p[i] - p[j]
                L = np.hypot(*d)
                start = p[j] + d / L * min(r[j] - outline * scale, L) if L > 0 else p[j]
                stick(ax, start, p[i], bond * scale, z,
                      mix(INK, fog_color, 0.5 * (depth[i] + depth[j])))
        ax.add_patch(Circle(p[i], r[i], fc=mix(INK, fog_color, depth[i]), ec='none',
                            zorder=z + zstep))
        ax.add_patch(Circle(p[i], r[i] - outline * scale,
                            fc=mix(COLOR[el[i]], fog_color, depth[i]), ec='none',
                            zorder=z + 2 * zstep))
    return p, r


def limits(ax, p, r, pad=0.1):
    ax.set_xlim((p[:, 0] - r).min() - pad, (p[:, 0] + r).max() + pad)
    ax.set_ylim((p[:, 1] - r).min() - pad, (p[:, 1] + r).max() + pad)
    ax.set_aspect('equal')
    ax.set_axis_off()
