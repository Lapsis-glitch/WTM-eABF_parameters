"""
Running-RMSD-to-reference convergence for every replica, with restart-aware time
ordering and .BAK history recovery.

For the 14 clean-slate-restarted runs the live output holds only the post-restart
tail; the pre-restart history is recovered from window1.abf1.hist.czar.pmf.BAK
(staged from HPC into recovered_hist/).  Each history block is placed at its true
CUMULATIVE simulation time.

Writes results/timeseries_cache.npz and figures/fig7_convergence.png.
Run:  python convergence.py
"""

import os
import re
import glob
import numpy as np

import keto_pmf as k
import pspace as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), 'runs')
RECDIR = os.path.join(HERE, "recovered_hist")
RESDIR = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(RESDIR, exist_ok=True)

STRIDE_STEPS = 100_000          # PMF history output interval (0.2 ns; verified: 2500 blocks/500 ns)
SUBSAMPLE = 4                   # keep every Nth block for the plot
CMP_MASK_ZMAX = 38.0


def recovered_bak(run):
    """(hist.BAK, traj.BAK) staged for this run, or (None, None).
    Looks in the per-run subfolder first, then the flat staging name."""
    sub = os.path.join(RECDIR, run.dirname)
    for hist in (os.path.join(sub, "window1.abf1.hist.czar.pmf.BAK"),
                 os.path.join(RECDIR, f"{run.dirname}.hist.czar.pmf.BAK")):
        if os.path.isfile(hist):
            traj = os.path.join(sub, "window1.colvars.traj.BAK")
            return hist, (traj if os.path.isfile(traj) else None)
    return None, None


def _greedy_merge(segments):
    """segments: list of (steps, G). Concatenate, sort by step, and keep only
    rows that extend coverage (strictly increasing step) -- drops overlaps and
    superseded shorter segments. Each block is a CUMULATIVE PMF at its true
    step, so a monotone-in-step sequence is the correct running history."""
    steps = np.concatenate([s for s, _ in segments])
    G = np.concatenate([g for _, g in segments], axis=0)
    order = np.argsort(steps, kind="stable")
    steps, G = steps[order], G[order]
    keep, cover = [], -1
    for i in range(len(steps)):
        if steps[i] > cover:
            keep.append(i); cover = steps[i]
    keep = np.array(keep)
    return steps[keep], G[keep]


def recovered_prerestart(run):
    """List of (hist, wipe_step) staged 'leads-up-to-the-wipe' backups.
    Each holds history ending at its wipe step; multiple ones (a run restarted
    several times) stitch to cover the whole trajectory."""
    import glob as _glob
    sub = os.path.join(RECDIR, run.dirname)
    out = []
    single = os.path.join(sub, "prerestart.hist.czar.pmf")
    stepf = os.path.join(sub, "prerestart.step")
    if os.path.isfile(single) and os.path.isfile(stepf):
        out.append((single, int(open(stepf).read().strip())))
    for h in sorted(_glob.glob(os.path.join(sub, "prerestart_*.hist.czar.pmf"))):
        m = re.search(r"prerestart_(\d+)\.hist", os.path.basename(h))
        if m:
            out.append((h, int(m.group(1))))
    return out


def running_history(run):
    """(t_ns, G_blocks, z, fully_recovered) with restart-aware cumulative timing.

    Merges every recoverable segment by true cumulative step:
      * live output (+ in-tree archives)
      * the pre-restart backup that leads up to the wipe (0 -> wipe step)
      * the rolling .BAK (timed from traj.BAK)
    """
    # Prefer a pre-built complete-history file if present (self-contained, no
    # recovered_hist/ needed): it already holds the full 0->500 ns stitched run.
    comp = os.path.join(run.outdir, "window1.abf1.hist.czar.complete.pmf")
    if os.path.isfile(comp):
        z, G = k.read_pmf_blocks(comp)
        nb = G.shape[0]
        steps = np.round(np.linspace(k.FULL_RUN_STEPS / nb, k.FULL_RUN_STEPS, nb)).astype(np.int64)
        return steps * k.STEPS_TO_NS, G, z, True

    steps, z, G, recovered = k.load_running_pmf(run)
    segs = [(steps, G)]

    for hist, pstep in recovered_prerestart(run):
        zb, Gb = k.read_pmf_blocks(hist)
        nb = Gb.shape[0]
        # backup ends at its wipe step; block k at step pstep-(nb-1-k)*stride
        bsteps = (pstep - (nb - 1 - np.arange(nb)) * STRIDE_STEPS).astype(np.int64)
        keep = bsteps >= 0
        segs.append((bsteps[keep], Gb[keep]))
        if z is None:
            z = zb

    bak, trajbak = recovered_bak(run)
    if bak is not None:
        zb, Gb = k.read_pmf_blocks(bak)
        nb = Gb.shape[0]
        f = l = None
        if trajbak is not None:
            try:
                f, l, _, _ = k.traj_step_range(trajbak)
            except (ValueError, UnicodeDecodeError):
                f = l = None
        if f is not None and l is not None and 0 <= f < l <= k.FULL_RUN_STEPS:
            bsteps = np.round(np.linspace(f, l, nb)).astype(np.int64)
        else:
            restart = int(steps[0])
            bsteps = restart - (nb - 1 - np.arange(nb)) * STRIDE_STEPS
            Gb = Gb[bsteps >= 0]
            bsteps = bsteps[bsteps >= 0]
        segs.append((bsteps, Gb))

    steps, G = _greedy_merge(segs)
    recovered = steps[0] <= k.FULL_RUN_STEPS * 0.02
    return steps * k.STEPS_TO_NS, G, z, recovered


def build_cache():
    zref, gref = k.read_pmf(os.path.join(ROOT, "long-time_4000_seed40", "output", k.CZAR_NAME))
    ref = ps.symmetrize_free(gref)
    ref = ref - ref[np.abs(zref) >= 35].mean()
    mask = np.abs(zref) <= CMP_MASK_ZMAX

    runs = k.discover_runs(ROOT)
    out = dict(cells=[], seeds=[], t=[], rmsd=[], recovered=[], family=[], extfluc=[])
    for r in runs:
        t, G, z, rec = running_history(r)
        # symmetrize + anchor every block, RMSD to reference (vectorised)
        Gs = ps.symmetrize_free(G)
        Gs = Gs - Gs[:, np.abs(z) >= 35].mean(axis=1, keepdims=True)
        d = Gs - ref
        d = d[:, mask]
        d = d - d.mean(axis=1, keepdims=True)
        rmsd = np.sqrt((d ** 2).mean(axis=1))
        # subsample for storage/plot
        out["cells"].append(r.key)
        out["seeds"].append(r.seed)
        out["t"].append(t[::SUBSAMPLE])
        out["rmsd"].append(rmsd[::SUBSAMPLE])
        out["recovered"].append(rec)
        out["family"].append(r.family)
        out["extfluc"].append(r.params.get("extFluc", np.nan))
        print(f"{r.dirname:40s} nblk={G.shape[0]:5d} t0={t[0]:6.1f} rec={rec} "
              f"final_rmsd={rmsd[-1]:.3f}")
    np.savez_compressed(os.path.join(RESDIR, "timeseries_cache.npz"),
                        cells=np.array(out["cells"]),
                        seeds=np.array(out["seeds"]),
                        t=np.array(out["t"], dtype=object),
                        rmsd=np.array(out["rmsd"], dtype=object),
                        recovered=np.array(out["recovered"]),
                        family=np.array(out["family"]),
                        extfluc=np.array(out["extfluc"]))
    print("cache ->", os.path.join(RESDIR, "timeseries_cache.npz"))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(RESDIR, "timeseries_cache.npz"), allow_pickle=True)
    OKABE = {0.05: "#0072B2", 0.1: "#E69F00", 0.2: "#009E73", 0.5: "#D55E00"}
    KT = k.KT

    fig, ax = plt.subplots(figsize=(9, 5))
    n_recovered_gap = 0
    for i in range(len(d["cells"])):
        fam = d["family"][i]
        if fam != "extFluc":
            continue
        ef = float(d["extfluc"][i])
        t, r = d["t"][i], d["rmsd"][i]
        ls = "-" if d["recovered"][i] else ":"
        if not d["recovered"][i]:
            n_recovered_gap += 1
        ax.plot(t, r, color=OKABE[ef], lw=0.7, alpha=0.55, ls=ls)
    for ef, c in OKABE.items():
        ax.plot([], [], color=c, lw=2, label=f"extFluc {ef} $\\AA$")
    ax.axhline(KT, color="k", lw=1.0, ls="--", label=f"$k_BT$ = {KT:.2f} kcal/mol")
    ax.plot([], [], color="0.5", lw=1, ls=":", label="pre-restart history recovered from .BAK (partial)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"RMSD to 3.4 $\mu$s reference (kcal/mol)")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 2.5)
    ax.set_title("Ketoprofen permeation — running RMSD of each replica to the reference\n"
                 "(colour = extFluc; restart-aware cumulative time, .BAK-recovered histories)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_convergence.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig7_convergence.pdf"), bbox_inches="tight")
    print("figure -> fig7_convergence.png")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        build_cache()
    plot()
