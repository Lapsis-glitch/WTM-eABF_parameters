Deca-alanine 1D WTM-eABF parameter sweeps (end-to-end distance, atoms 10-92, 12-32 A, width 0.1 A)
Source: simulation directory deca_ala_final  (870 runs: 87 values x 10 seeds 10..100)
Runs: 10 ns (2e7 x 0.5 fs); history PMF every 1e5 steps = 0.05 ns per frame (200 frames); 20 frames = 1 ns.
Reference: reference_median.pmf (this folder; buildref @ 2f5bbe7, T=300, 100 points, all 870 runs).
  Also included: reference_average_all(.pmf/_err.pmf) and reference_average_filtered (MAD cut 3.5).
VERIFIED: aggregating deca_ala_1D_convergence_per_seed.csv reproduces results.dat exactly (all 85 rows, mean/std/min/max/n).
Other results files copied as-is (not re-verified): results_kT.dat (first 5 seeds), results_thres0.1/0.15/0.25.dat.
deca_ala_1D_final_pmf.csv: param,value,seed,xi,G  (last history block = final CZAR PMF, kcal/mol, xi in A)
Extended-variable data (behind extended_distribution.png):
  deca_ala_1D_xi_minus_lambda_hist.csv   param,value,seed,bin_center,count  (xi - lambda in A, 100 bins over each run's range)
  deca_ala_1D_xi_minus_lambda_stats.csv  param,value,seed,n_samples,mean,std,min,max
  From output/abf_00.colvars.traj of ALL 870 runs (AtomDistance - r_AtomDistance, every 2000 steps = 1 ps,
  10000 samples/run, step 0 dropped). The original plotting script is not available; the figure shows several
  seeds per value for all 9 sweeps with Gaussian fits, consistent with these data. Raw samples (~260 MB) not pushed.
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
