Ethanol 2D sweep: fullSamples x extendedFluctuation, 10 seeds per grid point (WTM-eABF)
=====================================================================================

Source
  simulation directory ehtanol_2D
  250 run directories fullSamp_<F>_extFluc_<S>_seed_<N>, all complete
  (each namd.log reaches step 10,000,000 and ends with "End of program").

Grid (exact values, as in the directory names / colvar.in)
  fullSamples          : 100, 500, 2000, 5000, 10000
  extendedFluctuation  : 0.01, 0.10, 0.2, 0.5, 2.0   (Angstrom)
  seeds                : 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
                         (NAMD `seed`; verified in all 250 namd.log "RANDOM NUMBER SEED")
  Everything else fixed: width 0.2, extendedTimeConstant default (200 fs),
  extendedLangevinDamping default (1 /ps), hillWeight 0.02, hillWidth 2,
  newHillFrequency default (1000), biasTemperature 1000.

Simulation
  timestep 2.0 fs, numSteps 10,000,000 -> 20 ns per run, Langevin 298 K (NVT).
  CV: distanceZ of the ethanol COM relative to the COM of all water oxygens, 0-28 A.

Frames
  ABF historyFreq = 100000 steps = 200 ps. Frame k (0-based) is the k-th block of
  output/window1.abf1.hist.czar.pmf, written at step (k+1)*100000, i.e.
  time_ns = (k+1) * 0.2. 100 frames per run (frame 99 = 20 ns).
  NB: results.dat and convergence_per_seed.csv give the 0-based frame index;
  convert to time as (frame+1)*0.2 ns (or frame*0.2 ns if you want to match
  the convention of the original convergence plots, which multiply the index).

Analysis code
  github.com/Lapsis-glitch/WTM-eABF_parameters, Convergence_evaluation @ 2f5bbe7
  (2f5bbe7d2dcf021f4b420928ea73687c5888836d, 2026-03-20), used unmodified via
  `git archive` (the local checkout has uncommitted edits; they were NOT used).
  Settings exactly as in folder_parser.py @2f5bbe7:
    PMFAnalyzer(hist.czar.pmf, hist.count, slope_thresh=0.01, n_recent=5,
                use_sliding_window=False, reference_pmf_file=<reference>,
                rmsd_thresh=0.592186869182, use_ref_and_slope=True)
  Criterion: RMSD_ref(frame) = RMS over bins of (PMF - min) - (REF - min), reference
  interpolated onto the run grid. Convergence frame = first frame with
  RMSD_ref < 0.592186869182 kcal/mol (kBT at 298 K) AND |d/dt of exp fit| < 0.01,
  where the exp fit A*exp(-B t)+C is fitted to the Savitzky-Golay(11,3)-smoothed RMSD;
  if no frame satisfies both, fallback = first frame with RMSD_ref < kBT; else unconverged.

Reference PMF used: reference_median.pmf
  Verified: re-running the code above on every run and aggregating as folder_parser.py
  does reproduces results.dat EXACTLY (25/25 lines identical) with reference_median.pmf
  (reference_average_all: 1/25, reference_average_filtered: 7/25). See _verification.txt.

Files
  results.dat                original, as-is. Columns: label mean std min max n
                             (convergence frame index statistics over converged seeds; n = converged seeds)
  convergence_per_seed.csv   fullSamples,extendedFluctuation,seed,frame   (frame empty = unconverged; none here)
  rmsd_ref_timeseries.csv    fullSamples,extendedFluctuation,seed,frame,time_ns,rmsd
                             (raw, unsmoothed RMSD_ref in kcal/mol vs reference_median.pmf)
  reference_pmfs/            reference_median.pmf (used), reference_average_all(.pmf, _err.pmf),
                             reference_average_filtered(.pmf, _err.pmf) = MAD-filtered mean (cut 3.5 MAD),
                             as written by reference_builder.py; kcal/mol, xi in Angstrom.
  _verification.txt          reproduction check output.
  NB: the CSV column layout for this 2D sweep uses the two grid parameters instead of param,value.
Units: PMF/RMSD kcal/mol; xi (ProjectionZ) Angstrom; time ns.
Nothing skipped (all files < 1 MB).
