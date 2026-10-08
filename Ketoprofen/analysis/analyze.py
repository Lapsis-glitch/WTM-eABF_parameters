"""
Compute per-cell symmetrized p-space consensus PMFs, error decompositions,
accuracy-vs-reference and inter-seed reproducibility, and physical observables,
for the ketoprofen / POPC WTM-eABF sweeps.  Saves everything to results/.

Run:  python analyze.py
"""

import os
import csv
import pickle
import numpy as np

import keto_pmf as k
import pspace as ps

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
# estimator + output location are overridable so the same code produces the
# median (default) and the mean variants into separate directories.
ESTIMATOR = "median"
# error model: "bootstrap" (default; Kang bootstrap SD of the consensus) or
# "popsd" (1 SD over replicas, matching the other systems, with asymmetry folded in)
ERRMODE = "popsd"
OUTROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(OUTROOT, "results")
os.makedirs(OUTDIR, exist_ok=True)

REFERENCE_RUN = "long-time_4000_seed40"     # ~3.4 us CZAR reference (user choice)
B_BOOT = 8000
CMP_MASK_ZMAX = 38.0                        # ignore the outer 2 A near the walls in RMSD

# parameter axes (Stage 2 plane)
EXTFLUC = [0.05, 0.1, 0.2, 0.5]
FULLSAMP = [500, 2000, 5000, 10000]
BIASTEMP = [1000, 2000, 4000, 8000]


def _apply_popsd(res, G, z):
    """Replace the bootstrap sigmas on a CellResult with the population-SD model
    (1 SD over replicas + folded-in asymmetry). Center line (res.pmf) unchanged."""
    pe = ps.population_errors(G, z)
    res.sigma_boot = pe["sd_repl"]      # inter-replica SD (matches other systems)
    res.sigma_asym = pe["sd_asym"]      # asymmetry (ketoprofen-specific)
    res.sigma_quad = pe["sd_quad"]      # quadrature
    res.sigma_nested = pe["sd_total"]   # exact 2N-pool total (primary band)
    return res


def _popsd_observables(res, G, z):
    """Observable value from the plotted consensus line (res.pmf), error = 1 SD
    over the replicas (population flavour), so value and error are consistent."""
    val = ps.pmf_scalars(res.pmf, z)
    pop = ps.population_observables(G, z)     # (mean, SD-over-seeds) per key
    return {k: (val[k], pop[k][1]) for k in val}


def load_cell_profiles(runs):
    z = None
    G = []
    seeds = []
    for r in runs:
        zz, g = k.read_pmf(r.czar)
        z = zz
        G.append(g)
        seeds.append(r.seed)
    return z, np.asarray(G), seeds


def main():
    runs = k.discover_runs(ROOT)
    cells = k.group_by_cell(runs)

    # --- reference (symmetrized) ---
    zref, gref = k.read_pmf(os.path.join(ROOT, REFERENCE_RUN, "output", k.CZAR_NAME))
    ref_sym = ps.symmetrize_free(gref)
    ref_sym = ref_sym - ref_sym[np.abs(zref) >= 35].mean()
    # reference self-consistency error = its own asymmetry (single 3.4 us run, no replicas)
    ref_asym = 0.5 * np.abs(gref - gref[::-1])
    cmp_mask = np.abs(zref) <= CMP_MASK_ZMAX

    results = {}          # cell key -> dict
    z_grid = None

    for key, cruns in cells.items():
        z, G, seeds = load_cell_profiles(cruns)
        z_grid = z
        res = ps.analyze_cell(G, z, B=B_BOOT, estimator=ESTIMATOR)
        obs = ps.observables(res)
        if ERRMODE == "popsd":
            res = _apply_popsd(res, G, z)
            obs = _popsd_observables(res, G, z)

        # per-seed symmetrized profiles (bulk-anchored) for RMSD metrics
        Gsym = ps.symmetrize_free(G)
        Gsym = Gsym - Gsym[:, np.abs(z) >= 35].mean(axis=1, keepdims=True)

        # accuracy: RMSD of each seed to the reference (mean +/- SD)
        acc = [ps.rmsd(gs, ref_sym, cmp_mask) for gs in Gsym]
        # reproducibility: mean pairwise RMSD among seeds
        pair = [ps.rmsd(Gsym[i], Gsym[j], cmp_mask)
                for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
        # accuracy of the cell CONSENSUS to reference
        cons_acc = ps.rmsd(res.pmf, ref_sym, cmp_mask)

        results[key] = dict(
            seeds=seeds,
            pmf=res.pmf,
            pmf_raw_mean=res.pmf_raw_mean,
            sigma_boot=res.sigma_boot,
            sigma_asym=res.sigma_asym,
            sigma_quad=res.sigma_quad,
            sigma_nested=res.sigma_nested,
            observables=obs,
            acc_mean=float(np.mean(acc)),
            acc_sd=float(np.std(acc, ddof=1)),
            repro=float(np.mean(pair)),
            cons_acc=cons_acc,
        )
        print(f"{key:32s} N={len(seeds)}  acc={np.mean(acc):.3f}+/-{np.std(acc,ddof=1):.3f}  "
              f"repro={np.mean(pair):.3f}  consVSref={cons_acc:.3f}  "
              f"barrier={obs['central_barrier'][0]:.2f}+/-{obs['central_barrier'][1]:.2f}")

    # --- global consensus over the Stage-2 parameter sweep only (Fig 15 analog) ---
    # Exclude the Stage-1 bias-temperature runs; the reference is built purely from
    # the extFluc x fullSamp plane.
    allG = []
    for key, cruns in cells.items():
        if not key.startswith("extFluc"):
            continue
        _, G, _ = load_cell_profiles(cruns)
        allG.append(G)
    allG = np.concatenate(allG, axis=0)
    gres = ps.analyze_cell(allG, z_grid, B=B_BOOT, estimator=ESTIMATOR)
    gres_other = ps.analyze_cell(allG, z_grid, B=1000,
                                 estimator="mean" if ESTIMATOR == "median" else "median")
    gobs = ps.observables(gres)
    if ERRMODE == "popsd":
        gres = _apply_popsd(gres, allG, z_grid)
        gobs = _popsd_observables(gres, allG, z_grid)
    print(f"\nGLOBAL consensus ({ESTIMATOR}) over {allG.shape[0]} runs: "
          f"barrier={gobs['central_barrier'][0]:.2f}+/-{gobs['central_barrier'][1]:.2f}, "
          f"well={gobs['well_depth'][0]:.2f}+/-{gobs['well_depth'][1]:.2f} kcal/mol  "
          f"(other estimator well = {ps.observables(gres_other)['well_depth'][0]:.2f})")
    global_ref = dict(pmf=gres.pmf, pmf_other=gres_other.pmf, estimator=ESTIMATOR,
                      pmf_raw_mean=gres.pmf_raw_mean,
                      sigma_boot=gres.sigma_boot, sigma_asym=gres.sigma_asym,
                      sigma_quad=gres.sigma_quad, sigma_nested=gres.sigma_nested,
                      observables=gobs, n=allG.shape[0])

    # --- per-cell PMF data files (the tangible deliverable) ---
    pmfdir = os.path.join(OUTDIR, "pmf")
    os.makedirs(pmfdir, exist_ok=True)
    if ERRMODE == "popsd":
        cols = "sd_repl  sd_asym  sd_quad  sd_total"
        emo = "error = 1 SD over replicas + folded asymmetry (sd_total)"
    else:
        cols = "sigma_boot  sigma_asym  sigma_quad  sigma_nested"
        emo = "error = bootstrap SD of consensus (sigma_nested)"
    header = (f"z_Angstrom  dG_kcal_mol  {cols}\n"
              f"# symmetrized p-space {ESTIMATOR} consensus; {emo}; bulk (|z|>=35 A) = 0; T=310 K")
    for key, r in results.items():
        M = np.column_stack([z_grid, r["pmf"], r["sigma_boot"], r["sigma_asym"],
                             r["sigma_quad"], r["sigma_nested"]])
        np.savetxt(os.path.join(pmfdir, f"pmf_{key}.dat"), M,
                   fmt="%10.4f", header=header)
    M = np.column_stack([z_grid, global_ref["pmf"], global_ref["sigma_boot"],
                         global_ref["sigma_asym"], global_ref["sigma_quad"],
                         global_ref["sigma_nested"]])
    np.savetxt(os.path.join(pmfdir, "pmf_GLOBAL_consensus.dat"), M, fmt="%10.4f", header=header)

    # --- save ---
    with open(os.path.join(OUTDIR, "results.pkl"), "wb") as f:
        pickle.dump(dict(z=z_grid, results=results, global_ref=global_ref,
                         ref_sym=ref_sym, ref_asym=ref_asym, zref=zref,
                         estimator=ESTIMATOR, errmode=ERRMODE,
                         EXTFLUC=EXTFLUC, FULLSAMP=FULLSAMP, BIASTEMP=BIASTEMP,
                         REFERENCE_RUN=REFERENCE_RUN), f)

    # --- CSV tables ---
    with open(os.path.join(OUTDIR, "per_cell_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "N_seeds", "acc_rmsd_mean", "acc_rmsd_sd",
                    "repro_mean_pairwise_rmsd", "consensus_vs_ref_rmsd",
                    "well_depth", "well_depth_err", "central_barrier",
                    "central_barrier_err", "center_vs_bulk", "center_vs_bulk_err",
                    "head_peak", "head_peak_err",
                    "rms_sigma_boot", "rms_sigma_asym", "rms_sigma_quad", "rms_sigma_nested"])
        for key, r in results.items():
            o = r["observables"]
            rms = lambda x: float(np.sqrt(np.mean(x**2)))
            w.writerow([key, len(r["seeds"]), f"{r['acc_mean']:.4f}", f"{r['acc_sd']:.4f}",
                        f"{r['repro']:.4f}", f"{r['cons_acc']:.4f}",
                        f"{o['well_depth'][0]:.3f}", f"{o['well_depth'][1]:.3f}",
                        f"{o['central_barrier'][0]:.3f}", f"{o['central_barrier'][1]:.3f}",
                        f"{o['center_vs_bulk'][0]:.3f}", f"{o['center_vs_bulk'][1]:.3f}",
                        f"{o['head_peak'][0]:.3f}", f"{o['head_peak'][1]:.3f}",
                        f"{rms(r['sigma_boot']):.4f}", f"{rms(r['sigma_asym']):.4f}",
                        f"{rms(r['sigma_quad']):.4f}", f"{rms(r['sigma_nested']):.4f}"])

    print(f"\nSaved -> {OUTDIR}/results.pkl and per_cell_summary.csv")


if __name__ == "__main__":
    main()
