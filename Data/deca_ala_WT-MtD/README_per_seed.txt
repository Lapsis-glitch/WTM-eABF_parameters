Deca-alanine standalone WT-MtD sweeps, per-seed data (companion to README.txt / results_WT-MtD_combined.dat
on branch wtmtd-pure-rerun; this file is named README_per_seed.txt to avoid a merge conflict with it).
Sources: deca_ala_MTD_v2 (MTDheight, MTDwidth, MTDnewhill; 75 runs)
  + deca_ala_MTD_pureTemp (MTDtemp, pure WT-MtD rerun, 30 runs).
  The old MTD_v2 MTDtemp runs (extended Lagrangian kept) are excluded. 5 seeds 10..50 per value.
Runs: 10 ns; free-energy snapshot every 1e5 steps = 0.05 ns per frame (200 frames), built by prep_files*.sh.
Reference: deca_ala_final/reference_median.pmf (copy in ../deca_ala_1D). Counts reader stubbed (no .count files; unused).
VERIFIED: aggregating wtmtd_combined_convergence_per_seed.csv reproduces results_WT-MtD_combined.dat exactly (16 rows).
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
