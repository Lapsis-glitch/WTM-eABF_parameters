"""
Restart-aware I/O for the ketoprofen / POPC WTM-eABF parameter sweeps.

Data layout (project root):
  Stage 1 (bias-temperature sweep):
      biastemp_{1000,2000,4000,8000}_seed_{10,20,30}/output/window1.abf1.czar.pmf
  Stage 2 (fluctuation x fullSamples plane):
      extFluc_{0.05,0.1,0.2,0.5}_fullSamp_{500,2000,5000,10000}_seed_{10,20,30,40,50}/output/...
  Reference:
      long-time_4000_seed40/output/...   (~3.4 us CZAR estimate)

Key correctness points handled here
------------------------------------
1. The *clean* set of runs is exactly  <run>/output/window1.abf1.czar.pmf .
   Stray copies live in archive_*/ , failed_*/ , prerun_* subfolders and MUST be
   excluded (a naive recursive glob picks up ~30 junk files).

2. The CZAR PMF grid header lower boundary is -40.1, but the actual bin centres in
   the data column start at -40.0.  We therefore read the *explicit* coordinate
   column rather than reconstructing it from the header (the example
   Convergence_evaluation/pmf_io.py reconstructs from the header and is off by
   half a bin here).

3. Restarts.  14/92 live outputs are post-restart continuations whose trajectory
   starts after step 0.  Because NAMD restored `firsttimestep` and Colvars reloaded
   its CZAR accumulators, the *final* czar.pmf still reflects the full 500 ns of
   sampling (verified: restarted runs' asymmetry sits inside their cell-mates'
   spread).  So for the final PMF we always use the live output/ file.
   For *time-dependent* histories, each block is placed at its true cumulative step
   (derived from the trajectory), and pre-restart segments from archives are
   stitched in when present; runs whose early history is unrecoverable are flagged.
"""

import os
import re
import glob
import numpy as np

# --- physical constants -----------------------------------------------------
# Production runs are NPT at 310 K (see WTM_eABF.pdf Methods, ketoprofen section).
TEMPERATURE_K = 310.0
KB_KCAL = 0.0019872041           # Boltzmann constant, kcal/mol/K
KT = KB_KCAL * TEMPERATURE_K     # ~0.6160 kcal/mol
BETA = 1.0 / KT

DT_FS = 2.0                      # integration timestep, fs
STEPS_TO_NS = DT_FS * 1e-6      # steps -> ns
FULL_RUN_STEPS = 250_000_000    # 500 ns nominal production length

CZAR_NAME = "window1.abf1.czar.pmf"
HIST_NAME = "window1.abf1.hist.czar.pmf"
TRAJ_NAME = "window1.colvars.traj"

# subfolders that indicate superseded / non-production data
_JUNK_DIR = re.compile(r"(archive|failed|prerun|backup|prebackup)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Single-PMF reader (explicit coordinate column)
# ---------------------------------------------------------------------------
def read_pmf(path):
    """Read a single-block Colvars .pmf.  Returns (z, g) float arrays.

    Reads the explicit coordinate column (robust to the header/data half-bin
    offset in this dataset).
    """
    z, g = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a = line.split()
            z.append(float(a[0]))
            g.append(float(a[1]))
    return np.asarray(z), np.asarray(g)


def read_pmf_blocks(path):
    """Read a multi-block history .pmf.  Returns (z, G) where G has shape
    (nblocks, nbins); block order is file (append) order = chronological within
    a single continuous segment."""
    z = None
    blocks, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            if not line.strip():
                continue
            a = line.split()
            cur.append((float(a[0]), float(a[1])))
    if cur:
        blocks.append(cur)
    z = np.asarray([c[0] for c in blocks[0]])
    G = np.asarray([[c[1] for c in b] for b in blocks])
    return z, G


# ---------------------------------------------------------------------------
# Trajectory helpers (the authoritative simulation clock)
# ---------------------------------------------------------------------------
def traj_step_range(traj_path):
    """Return (first_step, last_step, nrows, monotonic) from a colvars.traj."""
    first = last = None
    n = 0
    prev = None
    mono = True
    with open(traj_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            s = int(line.split()[0])
            n += 1
            if first is None:
                first = s
            if prev is not None and s <= prev:
                mono = False
            prev = s
            last = s
    return first, last, n, mono


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
class Run:
    """One replica (one seed) of one parameter cell."""

    def __init__(self, root, dirname):
        self.root = root
        self.dirname = dirname
        self.path = os.path.join(root, dirname)
        self.outdir = os.path.join(self.path, "output")
        self.czar = os.path.join(self.outdir, CZAR_NAME)
        self.hist = os.path.join(self.outdir, HIST_NAME)
        self.traj = os.path.join(self.outdir, TRAJ_NAME)
        self.family, self.params, self.seed = _parse_name(dirname)

    @property
    def key(self):
        """Cell key without the seed (groups replicas of the same parameters)."""
        if self.family == "biastemp":
            return f"biastemp_{self.params['biastemp']}"
        return f"extFluc_{self.params['extFluc']}_fullSamp_{self.params['fullSamp']}"

    def is_complete(self):
        f, l, _, mono = traj_step_range(self.traj)
        return (l == FULL_RUN_STEPS) and mono

    def restart_first_step(self):
        f, _, _, _ = traj_step_range(self.traj)
        return f  # 0 for a clean run; >0 for a post-restart continuation

    def __repr__(self):
        return f"<Run {self.dirname}>"


def _parse_name(dirname):
    m = re.match(r"biastemp_(\d+)_seed_(\d+)$", dirname)
    if m:
        return "biastemp", {"biastemp": int(m.group(1))}, int(m.group(2))
    m = re.match(r"extFluc_([\d.]+)_fullSamp_(\d+)_seed_(\d+)$", dirname)
    if m:
        return ("extFluc",
                {"extFluc": float(m.group(1)), "fullSamp": int(m.group(2))},
                int(m.group(3)))
    return None, {}, None


def discover_runs(root, family=None):
    """Return the clean list of production Run objects (excludes junk subdirs).

    family: 'biastemp', 'extFluc', or None for both.
    """
    runs = []
    for dirname in sorted(os.listdir(root)):
        full = os.path.join(root, dirname)
        if not os.path.isdir(full):
            continue
        fam, params, seed = _parse_name(dirname)
        if fam is None:
            continue
        if family and fam != family:
            continue
        r = Run(root, dirname)
        if os.path.isfile(r.czar):
            runs.append(r)
    return runs


def group_by_cell(runs):
    """dict: cell key -> list[Run] sorted by seed."""
    cells = {}
    for r in runs:
        cells.setdefault(r.key, []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r.seed)
    return cells


# ---------------------------------------------------------------------------
# Restart-aware running-PMF history (for convergence-vs-time figures)
# ---------------------------------------------------------------------------
def _segment_dirs(run):
    """All dirs holding a hist+traj pair for one run: the live output plus any
    archived pre-restart segments (junk subdirs), each paired with its traj step
    range.  Handles both <sub>/output/ and <sub>/ layouts."""
    segs = []
    live = run.outdir
    if os.path.isfile(os.path.join(live, HIST_NAME)):
        f, l, _, _ = traj_step_range(os.path.join(live, TRAJ_NAME))
        segs.append((f, l, live))
    for sub in sorted(os.listdir(run.path)):
        if sub == "output" or not _JUNK_DIR.search(sub):
            continue
        for cand in (os.path.join(run.path, sub, "output"), os.path.join(run.path, sub)):
            h = os.path.join(cand, HIST_NAME)
            t = os.path.join(cand, TRAJ_NAME)
            if os.path.isfile(h) and os.path.isfile(t):
                f, l, _, _ = traj_step_range(t)
                segs.append((f, l, cand))
    return segs


def load_running_pmf(run):
    """Reconstruct the time-ordered running PMF for a run.

    Returns (steps, z, G, recovered_from_zero) where
      steps : (nblocks,) cumulative simulation step of each block (sorted, deduped)
      G     : (nblocks, nbins) running CZAR PMFs
      recovered_from_zero : True if the running history starts at (near) step 0,
                            False if pre-restart history is unrecoverable.

    Each block is placed at its true cumulative step: within a segment covering
    traj steps [f, l] with B blocks, block k is assigned step
    round(linspace(f, l, B)[k]).  Because CZAR accumulators are restored across a
    restart, a post-restart block is a *cumulative* PMF and belongs at its
    cumulative step, not at a segment-relative one.
    """
    segs = _segment_dirs(run)
    # Assign a cumulative step to every block, per segment.
    seg_data = []
    z_ref = None
    for f, l, outdir in segs:
        z, G = read_pmf_blocks(os.path.join(outdir, HIST_NAME))
        if z_ref is None:
            z_ref = z
        B = G.shape[0]
        if B == 1:
            steps = np.array([l], dtype=np.int64)
        else:
            steps = np.round(np.linspace(f if f > 0 else l / B, l, B)).astype(np.int64)
        seg_data.append((int(steps[0]), int(steps[-1]), steps, G))

    # Greedy stitch by coverage: start from the earliest segment, and only add a
    # later segment where it EXTENDS beyond the current coverage.  This uses a
    # complete live run as-is (ignoring superseded archives) while filling the
    # early gap of a genuine post-restart run when an archive provides it.
    seg_data.sort(key=lambda s: (s[0], s[1]))
    steps_out, G_out, cover = [], [], -1
    for s0, s1, steps, G in seg_data:
        if s1 <= cover:
            continue                     # fully redundant with what we already have
        mask = steps > cover             # keep only the portion that extends coverage
        steps_out.append(steps[mask])
        G_out.append(G[mask])
        cover = s1
    steps = np.concatenate(steps_out)
    G = np.concatenate(G_out, axis=0)
    order = np.argsort(steps, kind="stable")
    steps, G = steps[order], G[order]
    recovered = steps[0] <= FULL_RUN_STEPS * 0.02   # history starts within ~10 ns of zero
    return steps, z_ref, G, bool(recovered)


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(os.path.dirname(root), 'runs')  # Ketoprofen/runs
    runs = discover_runs(root)
    cells = group_by_cell(runs)
    print(f"kT = {KT:.4f} kcal/mol   (T = {TEMPERATURE_K} K)")
    print(f"discovered {len(runs)} clean runs in {len(cells)} cells")
    incomplete = [r.dirname for r in runs if not r.is_complete()]
    print(f"incomplete (traj != 500 ns): {incomplete or 'none'}")
    restarted = [r.dirname for r in runs if r.restart_first_step() != 0]
    print(f"post-restart live outputs ({len(restarted)}): {restarted}")
