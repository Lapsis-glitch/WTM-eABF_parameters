NANMA (alanine dipeptide, vacuum) 2D phi/psi WTM-eABF parameter sweeps
Source: simulation directory NANMA/Simulations_final  (265 runs: 53 values x 5 seeds 10..50)
Runs: 50 ns (1e8 x 0.5 fs); history PMF every 1e6 steps = 0.5 ns per frame (100 frames); divisor 2 gives ns.
Reference: reference_median.pmf (this folder), plus reference_average_all and reference_average_filtered (+_err).
  2D Colvars .pmf format: phi psi G (deg, deg, kcal/mol), 72x72 grid, 5 deg bins.
VERIFIED: aggregating nanma_convergence_per_seed.csv reproduces results.dat exactly (all rows).
results_thres0.25.dat copied as-is.
SKIPPED: final 2D PMFs of all 265 runs (param,value,seed,xi1,xi2,G; 72x72 grid each) = 62 MB as CSV,
  over the 50 MB per-file limit. Available on request (11 MB gzipped).
Criterion (WTM-eABF_parameters/Convergence_evaluation @ 2f5bbe7, folder_parser + analyze_ND):
  raw RMSD of each history PMF to the reference (both shifted to min 0, reference interpolated onto the run grid);
  converged at the FIRST frame with raw RMSD < 0.592186869182 kcal/mol (kB*298 K) AND |d/dt of exp fit to the
  Savitzky-Golay-smoothed RMSD| < 0.01 per frame; if no frame satisfies both, falls back to the first frame with
  raw RMSD < 0.5922 (no slope test). Unconverged seeds are dropped; n in results*.dat = converged seeds.
Columns
  results*.dat                 <name> mean std min max n   (frames; produced by folder_parser, copied as-is)
  *_convergence_per_seed.csv   param,value,seed,frame       (frame = 0-based history index; empty = not converged)
  *_rmsd_ref.csv               param,value,seed,frame,time_ns,rmsd  (raw RMSD the criterion sees, kcal/mol;
                               time_ns = (frame+1)*frame spacing, i.e. the time the history block was written)
  The paper figures plot frame/divisor (divisor 20 for deca-ala), which omits the +1.
Extraction: same PMFAnalyzer code with a thin per-seed export wrapper; generated 2026-10-02.
