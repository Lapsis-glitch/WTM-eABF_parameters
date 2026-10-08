"""Atomic coordinates for the ball-and-stick drawings in Figure 1.

Run from the repository root with an environment that has RDKit and MDAnalysis:
    python Figures/scripts/prep_structures.py   (needs RDKit and MDAnalysis)
Writes Figures/structures/<name>.json with elements, coordinates (Å) and bonds.

Sources:
- NANMA (Ace-Ala-NMe): built with RDKit, MMFF94 with phi/psi restrained to the two
  minima of Data/NANMA/reference_median.pmf (C7eq -85/75, C7ax 70/-70).
- Deca-alanine: Deca_ala/common/deca-ala.{psf,pdb} of WTM-eABF_parameters (helix).
  The unfolded form is the same molecule with all backbone phi/psi set to an
  extended conformation.
- Ethanol: Ethanol/common/reference.pdb of WTM-eABF_parameters.
- Ketoprofen: resname LIG of Ketoprofen/common/step5_input.psf and
  step7_production.coor.
"""

import json
import os
import warnings

import numpy as np
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import rdForceFieldHelpers as ffh

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAR = ROOT
OUT = os.path.join(ROOT, 'Figures', 'structures')
os.makedirs(OUT, exist_ok=True)


def save(name, elements, xyz, bonds, **extra):
    d = dict(elements=list(elements), xyz=np.round(np.asarray(xyz), 4).tolist(),
             bonds=[[int(i), int(j)] for i, j in bonds], **extra)
    with open(os.path.join(OUT, name + '.json'), 'w') as f:
        json.dump(d, f)
    print('wrote', name, len(elements), 'atoms', len(bonds), 'bonds', extra)


def dihedral(p):
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - b0.dot(b1) * b1
    w = b2 - b2.dot(b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w)))


# ---------------------------------------------------------------- NANMA
def nanma(phi, psi, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles('CC(=O)N[C@@H](C)C(=O)NC'))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    # heavy-atom indices: 0 CH3, 1 C, 2 O, 3 N, 4 CA, 5 CB, 6 C, 7 O, 8 N, 9 CH3
    conf = m.GetConformer()
    tors = [((1, 3, 4, 6), phi), ((3, 4, 6, 8), psi),
            ((0, 1, 3, 4), 180.0), ((4, 6, 8, 9), 180.0)]
    for t, v in tors:
        rdMolTransforms.SetDihedralDeg(conf, *t, v)
    props = ffh.MMFFGetMoleculeProperties(m)
    ff = ffh.MMFFGetMoleculeForceField(m, props)
    for t, v in tors:
        ff.MMFFAddTorsionConstraint(*t, False, v, v, 1e4)
    ff.Minimize(maxIts=5000)
    xyz = m.GetConformer().GetPositions()
    print('NANMA phi/psi', [round(dihedral(xyz[list(t)])) for t, _ in tors[:2]])
    el = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return el, xyz, bonds


for name, (phi, psi) in {'nanma_c7eq': (-85, 75), 'nanma_c7ax': (70, -70)}.items():
    el, xyz, bonds = nanma(phi, psi)
    # phi bond N-CA (3-4), psi bond CA-C (4-6)
    save(name, el, xyz, bonds, phi_bond=[3, 4], psi_bond=[4, 6])


# ---------------------------------------------------------------- deca-alanine
def element(names):
    return [n[0] for n in names]


u = mda.Universe(os.path.join(PAR, 'Deca_ala/common/deca-ala.psf'),
                 os.path.join(PAR, 'Deca_ala/common/deca-ala.pdb'))
el = element(u.atoms.names)
bonds = [tuple(b.indices) for b in u.bonds]
cv = [9, 91]                                       # carbonyl C of Ala1 and Ala10 (CV atoms)
save('deca_folded', el, u.atoms.positions, bonds, cv=cv)

nbr = {i: set() for i in range(len(el))}
for i, j in bonds:
    nbr[i].add(j)
    nbr[j].add(i)


def side(b, c):
    """Atoms on the c side of bond b-c."""
    seen, todo = {c}, [c]
    while todo:
        k = todo.pop()
        for n in nbr[k]:
            if n not in seen and not (k == c and n == b):
                seen.add(n)
                todo.append(n)
    return sorted(seen - {c})


def set_dihedral(xyz, idx, target):
    a, b, c, d = idx
    ang = np.radians(target - dihedral(xyz[list(idx)]))
    k = (xyz[c] - xyz[b]) / np.linalg.norm(xyz[c] - xyz[b])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    mv = side(b, c)
    xyz[mv] = (xyz[mv] - xyz[c]) @ R.T + xyz[c]


xyz = u.atoms.positions.astype(float).copy()
PHI, PSI = -140.0, 140.0
for r in u.residues:
    s = r.phi_selection()
    if s is not None:
        set_dihedral(xyz, tuple(s.indices), PHI)
    s = r.psi_selection()
    if s is not None:
        set_dihedral(xyz, tuple(s.indices), PSI)
d = np.linalg.norm(xyz[cv[0]] - xyz[cv[1]])
dm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
for i, j in bonds:
    dm[i, j] = dm[j, i] = 9
np.fill_diagonal(dm, 9)
print(f'deca unfolded d = {d:.1f} A, closest non-bonded pair {dm.min():.2f} A')
save('deca_unfolded', el, xyz, bonds, cv=cv)

# ---------------------------------------------------------------- ethanol
e = mda.Universe(os.path.join(PAR, 'Ethanol/common/reference.pdb'))
ex = e.select_atoms('resname ETOH').positions
eb = [(i, j) for i in range(len(ex)) for j in range(i) if np.linalg.norm(ex[i] - ex[j]) < 1.6]
save('ethanol', element(e.select_atoms('resname ETOH').names), ex, eb)

# ---------------------------------------------------------------- ketoprofen
K = os.path.join(ROOT, 'Ketoprofen', 'common')
k = mda.Universe(os.path.join(K, 'step5_input.psf'), os.path.join(OUT, 'keto_step7_production.coor'),
                 format='NAMDBIN')
lig = k.select_atoms('resname LIG')
i0 = lig.indices[0]
save('ketoprofen', element(lig.names), lig.positions,
     [(b.indices[0] - i0, b.indices[1] - i0) for b in lig.bonds])
