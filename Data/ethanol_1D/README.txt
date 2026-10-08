Ethanol 1D parameter sweep, 5 seeds per value (WTM-eABF)
========================================================

Source
  simulation directory ethanol_final
  450 run directories <param>_<value>_seed_<N> = 90 parameter values x 5 seeds (10,20,30,40,50).
  Seeds verified: abf.in `seed` and namd.log "RANDOM NUMBER SEED" equal the directory seed
  for 450/450 runs. All runs complete (TIMESTEP 2, last step 10,000,000, "End of program").
  NB: all files in that directory carry 2026-09-29 timestamps (copied without preserving
  mtimes), so the original run/analysis dates are not recoverable from it.
  This is the data behind the SI figure ethanol_convergence_kT.

Parameter labels (directory prefix -> Colvars keyword) and swept values
  MTDheight   -> hillWeight              0.005 0.010 0.015 0.020 0.030 0.050 0.100 0.200 0.500 1.00
  MTDnewhill  -> newHillFrequency        100 200 300 500 700 1000 1500 2000 2500 3000
  MTDtemp     -> biasTemperature         500 1000 1500 2000 3000 4000 5000 8000 10000 15000
  MTDwidth    -> hillWidth               0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0
  colvarWidth -> colvar width            0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0
  extDamp     -> extendedLangevinDamping 0.05 0.10 0.20 0.30 0.50 0.70 1.0 1.5 2.0 5.0
  extFluc     -> extendedFluctuation     0.01 0.05 0.10 0.15 0.2 0.3 0.5 0.7 1.0 2.0
  extTime     -> extendedTimeConstant    10 20 30 50 100 150 200 250 300 500
  fullSamp    -> fullSamples             50 100 200 300 500 1000 2000 3000 5000 10000
  Baseline (non-swept): fullSamples 5000, width 0.2, extendedFluctuation 0.1,
  extendedTimeConstant default (200 fs), extendedLangevinDamping default (1 /ps),
  hillWeight 0.02, hillWidth 2, newHillFrequency default (1000), biasTemperature 1000.

Simulation
  1 ethanol + 456 TIP3P waters (water slab), box 23.875 x 23.875 x 71.625 A, CGenFF 2b6.
  NVT, Langevin 298 K (damping 2 /ps), initial velocities at 298 K from the seed.
  Cutoff 11 A (switch 10, pairlist 12), PME spacing 1.2 A, rigidBonds all, GPUresident.
  CV "ProjectionZ": distanceZ of ethanol COM (atoms 1-9) relative to the COM of all
  water oxygens; 0-28 A; walls at -1.5 / 29.5 A (k=100).
  timestep 2.0 fs, numSteps 10,000,000 -> 20 ns per run.

Frames
  ABF historyFreq = 100000 steps = 200 ps. Frame k (0-based) is the k-th block of
  output/window1.abf1.hist.czar.pmf, written at step (k+1)*100000, i.e.
  time_ns = (k+1) * 0.2. 100 frames per run (frame 99 = 20 ns).
  results.dat and convergence_per_seed.csv give the 0-based frame index.

Analysis code
  github.com/Lapsis-glitch/WTM-eABF_parameters, Convergence_evaluation @ 2f5bbe7
  (2f5bbe7d2dcf021f4b420928ea73687c5888836d), used unmodified via `git archive`.
  Settings exactly as folder_parser.py @2f5bbe7 (slope_thresh=0.01, n_recent=5,
  use_sliding_window=False, rmsd_thresh=0.592186869182, use_ref_and_slope=True).
  Criterion: first frame with RMSD_ref < 0.592186869182 kcal/mol (kBT, 298 K) AND
  |slope of exp fit to SG(11,3)-smoothed RMSD| < 0.01; fallback first frame with
  RMSD_ref < kBT; else unconverged.

Reference PMF used: reference_median.pmf
  Verified: aggregating the per-seed results exactly as folder_parser.py does reproduces
  results.dat EXACTLY (90/90 lines identical) with reference_median.pmf
  (reference_average_all: 14/90, reference_average_filtered: 50/90). See _verification.txt.

Files
  results.dat               original, as-is. Columns: label mean std min max n
                            (frame-index stats over converged seeds; n = converged seeds).
  convergence_per_seed.csv  param,value,seed,frame   (frame empty = unconverged; 5 of 450 rows:
                            MTDwidth 0.1 (3 seeds), MTDwidth 0.2 (2 seeds))
  rmsd_ref_timeseries.csv   param,value,seed,frame,time_ns,rmsd  (raw, unsmoothed, kcal/mol)
  final_pmf_per_run.csv     param,value,seed,xi,G  final CZAR PMF of every run = last history
                            block (frame 99, 20 ns) of window1.abf1.hist.czar.pmf, as written by
                            Colvars (not re-zeroed); xi in A, G in kcal/mol. Grid follows each
                            run's colvar width.
  reference_pmfs/           reference_median.pmf (used); reference_average_all(.pmf, _err.pmf) = mean;
                            reference_average_filtered(.pmf, _err.pmf) = MAD-filtered mean (cut 3.5 MAD),
                            all as written by reference_builder.py.
  _verification.txt         reproduction check output.
Units: PMF/RMSD kcal/mol; xi Angstrom; time ns.
Nothing skipped (largest file 2.7 MB).
