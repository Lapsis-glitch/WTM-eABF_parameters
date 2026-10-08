Deca-alanine 2D grid: fullSamples {100,500,2000,5000,10000} x extendedFluctuation {0.01,0.10,0.2,0.5,2.0}
Source: simulation directory deca_ala_final2D  (250 runs, 10 seeds 10..100 per cell)
Runs: 10 ns; 0.05 ns per history frame. param = fullSamp_extFluc, value = <fullSamples>_<extFluc>.
Reference used for the CSVs: deca_ala_final/reference_median.pmf (copy in ../deca_ala_1D).
Per-seed CSV: recomputed exactly from deca_ala_2D_rmsd_ref.csv by Convergence_evaluation/check_convergence_from_rmsd.py.
  The figures and tables use the per-seed CSV. results.dat (2026-03-25) is kept as-is but does not match it
  exactly (max |delta mean| = 2.9 frames = 0.15 ns at fullSamp_500_extFluc_0.01, other cells <= ~1 frame).
  It predates the committed 2D parser (c9ddce5, 2026-03-26) and was likely made with a different reference file.
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
