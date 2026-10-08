Deca-alanine standalone WT-MtD convergence data (Figures/Main/deca_ala_convergence_kT_WT-MtD.png)

Files
  results_pureTemp.dat          biasTemperature sweep rerun as pure WT-MtD (no extended Lagrangian, no ABF)
  results_WT-MtD_combined.dat   deca_ala_MTD_v2/results.dat with its MTDtemp rows replaced by results_pureTemp.dat
                                (hillWeight, hillWidth, newHillFrequency rows unchanged from deca_ala_MTD_v2)
  Columns: <param>_<value> mean std min max n   (times in 50-ps history frames; n = converged seeds of 5;
  values with 0/5 converged are absent: hillWidth 0.1/0.5/1.0, newHillFrequency 3000, biasTemperature 2000)

Runs
  simulation directories deca_ala_MTD_pureTemp (bias temperature) and deca_ala_MTD_v2 (other sweeps)
  NAMD 3 multicore, 2 cores per run
  Template = deca_ala_MTD_v2/MTDheight_0.100_seed_10; only biasTemperature (2000-30000 K) and seed (10-50) differ.
  10 ns (2e7 x 0.5 fs), free-energy snapshot every 1e5 steps (50 ps); history built by prep_files.sh.

Analysis
  WTM-eABF_parameters/Convergence_evaluation at commit 2f5bbe7 (folder_parser + analyze_ND):
  RMSD to deca_ala_final/reference_median.pmf < 0.592186869182 kcal/mol (kT at 298 K) AND |slope of exp fit| < 0.01,
  first frame satisfying it (falls back to RMSD-only if none). Counts reader stubbed (no .count files; unused).
  Plot: plot_results.py (2f5bbe7) results_WT-MtD_combined.dat --divisor 20  (frames -> ns). Orange = n < 3.
