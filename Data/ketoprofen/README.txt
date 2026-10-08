Ketoprofen permeation through a POPC bilayer (CV = z projection of the ketoprofen-midplane separation)
Source: Ketoprofen/runs (raw output, histories xz-compressed, see ../../unpack_keto.sh). Generated 2026-10-08 by
  Figures/scripts/export_keto_data.py, which uses the I/O and PMF code of
  Ketoprofen/analysis (keto_pmf.py, pspace.py, convergence.py).
VERIFIED: keto_cells.csv acc/repro reproduce Ketoprofen/analysis/results/per_cell_summary.csv exactly,
  and the running RMSD of the last history block equals the final-PMF RMSD for every run.

Runs (92 + reference), 2 fs, 500 ns each (2.5e8 steps), T = 310 K:
  stage 1: biastemp_{1000,2000,4000,8000}_seed_{10,20,30}         extFluc 0.1, fullSamples 5000 (main.tex Methods)
  stage 2: extFluc_{0.05,0.1,0.2,0.5}_fullSamp_{500,2000,5000,10000}_seed_{10..50}   biasT 4000 K
  reference: long-time_4000_seed40 (~3.4 us CZAR PMF)
  14 runs were restarted mid-way. Their final czar.pmf covers the full 500 ns. Their running history is the
  stitched output/window1.abf1.hist.czar.complete.pmf (2500 blocks), placed on an even 0.2-ns grid.

Processing
  Every PMF symmetrized in G space (0.5 * (G(z) + G(-z))) and anchored to bulk (mean over |z| >= 35 A = 0).
  RMSD over |z| <= 38 A after removing the mean difference, against the symmetrized reference, kcal/mol.
  kT = kB * 310 K = 0.616 kcal/mol (not the 0.592 used for the other systems).
  First crossing = first history block (0.2 ns resolution) with RMSD < kT. Diagnostic only: the running PMF
  can cross and move away again.

Files
  keto_per_seed.csv   run,cell,stage,extFluc,fullSamples,biasT,seed,final_rmsd,last_block_rmsd,
                      first_crossing_ns,restart_first_step (0 = clean run, else first step of the live traj)
  keto_cells.csv      cell,stage,extFluc,fullSamples,biasT,n,acc_mean,acc_sd (ddof 1, over seeds),
                      repro_mean_pairwise (mean pairwise RMSD of the final PMFs),n_crossed,
                      first_crossing_mean,first_crossing_sd
  keto_rmsd_ref.csv   run,...,seed,time_ns,rmsd   running RMSD to the reference, every fifth block (1 ns)
  keto_xi_minus_lambda_hist.csv   run,...,seed,bin_center,count   xi - lambda (ProjectionZ - r_ProjectionZ, A),
                      0.01-A bins over -6..6 A, only non-empty bins written
  keto_xi_minus_lambda_stats.csv  run,...,seed,n_samples,n_outside,mean,std,min,max
                      From output/window1.colvars.traj, every 0.1 ns (5e4 steps), step 0 dropped.
                      Restarted runs only have the post-restart part (down to ~300 samples instead of 5000).
  reference_pmfs.dat  reference and 80-run median PMFs used in Figure 1 (from results.pkl, see header)
