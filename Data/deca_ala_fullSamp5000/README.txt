Deca-alanine 1D parameter sweep, fullSamples=5000 baseline, 10 seeds per value (WTM-eABF)
=========================================================================================

Source
  simulation directory deca_ala_final_fullSamp5000
  (unzipped from deca_ala_final_fullSamp5000.zip). 870 run directories
  <param>_<value>_seed_<N> = 87 parameter values x 10 seeds (10,20,...,100).
  Seeds verified in every abf.in (`seed` = directory seed). No namd.log files were kept;
  completeness verified from the history files: all 870 have 200 frames.
  This is the data behind the SI figure deca_ala_convergence_kT_fullSample5000_10replicas.

Parameter labels (directory prefix -> Colvars keyword) and swept values
  MTDheight   -> hillWeight              0.005 0.010 0.015 0.020 0.030 0.050 0.100 0.200 0.500 1.00
  MTDnewhill  -> newHillFrequency        500 700 1000 1500 2000 2500 3000
  MTDtemp     -> biasTemperature         500 1000 1500 2000 3000 4000 5000 8000 10000 15000
  MTDwidth    -> hillWidth               0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0
  colvarWidth -> colvar width            0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0
  extDamp     -> extendedLangevinDamping 0.05 0.10 0.20 0.30 0.50 0.70 1.0 1.5 2.0 5.0
  extFluc     -> extendedFluctuation     0.01 0.05 0.10 0.15 0.2 0.3 0.5 0.7 1.0 2.0
  extTime     -> extendedTimeConstant    10 20 30 50 100 150 200 250 300 500
  fullSamp    -> fullSamples             50 100 200 300 500 1000 2000 3000 5000 10000
  Baseline (non-swept): fullSamples 5000, width 0.1, extendedFluctuation 0.1,
  extendedTimeConstant 200, extendedLangevinDamping 2.0, hillWeight 0.1, hillWidth 3,
  newHillFrequency default (1000), biasTemperature 1000.

Simulation
  Vacuum, CHARMM22/CMAP. CV = distance between atoms 10 and 92, 12-32 A
  (harmonic walls 11.5/32.5, k=10). Langevin 300 K, damping 1 /ps.
  timestep 0.5 fs, numsteps 20,000,000 -> 10 ns per run.

Frames
  ABF historyFreq = 100000 steps = 50 ps. Frame k (0-based) is the k-th block of
  output/abf_00.abf1.hist.czar.pmf, written at step (k+1)*100000, i.e.
  time_ns = (k+1) * 0.05. 200 frames per run (frame 199 = 10 ns).
  results.dat and convergence_per_seed.csv give the 0-based frame index.

Analysis code
  github.com/Lapsis-glitch/WTM-eABF_parameters, Convergence_evaluation @ 2f5bbe7
  (2f5bbe7d2dcf021f4b420928ea73687c5888836d), used unmodified via `git archive`.
  Settings exactly as folder_parser.py @2f5bbe7 (slope_thresh=0.01, n_recent=5,
  use_sliding_window=False, rmsd_thresh=0.592186869182, use_ref_and_slope=True).
  Criterion: first frame with RMSD_ref < 0.592186869182 kcal/mol AND
  |slope of exp fit to SG(11,3)-smoothed RMSD| < 0.01; fallback first frame with
  RMSD_ref < 0.592186869182; else unconverged.
  NB: the threshold is kBT at 298 K, while these runs are at 300 K (kBT = 0.5962).

Reference PMF used (scored against): reference_median.pmf (in this tree's root)
  Verified: aggregating the per-seed results exactly as folder_parser.py does
  reproduces results.dat EXACTLY (84/84 lines identical) with reference_median.pmf;
  reference_average_all and reference_average_filtered give 0/84. See _verification.txt.

Files
  results.dat               original, as-is. Columns: label mean std min max n (frame index stats
                            over converged seeds). 84 lines: MTDwidth_0.1/0.2/0.3 have no converged
                            seed and are absent (folder_parser prints "skipped").
  convergence_per_seed.csv  param,value,seed,frame   (frame empty = unconverged; 61 of 870 rows)
  rmsd_ref_timeseries.csv   param,value,seed,frame,time_ns,rmsd  (raw, unsmoothed, kcal/mol)
  reference_pmfs/           reference_median.pmf (used), reference_average_all(.pmf,_err.pmf),
                            reference_average_filtered(.pmf,_err.pmf) = MAD-filtered mean.
  _verification.txt         reproduction check output.
Units: PMF/RMSD kcal/mol; xi (distance) Angstrom; time ns.
Nothing skipped (largest file 6.5 MB).
