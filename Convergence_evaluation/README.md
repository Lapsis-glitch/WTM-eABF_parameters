# Convergence_evaluation

Small toolkit to analyze PMF convergence and build robust reference PMFs.

Tools:
- `analyze_ND.py` — Analyze PMF histories and sampling counts:
  - Visualizes RMSD convergence, sequential PMFs, sampling evolution, and post-convergence vs final PMF.
  - Supports RMSD to final PMF, sliding-window RMSD, or RMSD to an external reference PMF.
  - Command-line options for thresholds, smoothing, annotation control, and saving figures.

- `buildref.py` — Build a reference PMF from multiple simulations:
  - Scans subdirectories for PMF files, interpolates to a common grid, computes median, average, and outlier-filtered averages.
  - Writes PMF files in the repository's sequential format and saves comparison plots.

- `pmf_io.py` — IO & utilities:
  - Read/write single PMF format and history (multi-block) format.
  - Interpolate PMFs to common grids.

- `folder_parser.py` / `folder_parser_2D.py` — Apply the convergence criterion used in the paper to every run of a
  one- or two-parameter sweep and write `results.dat` (mean, SD, min, max of the convergence frame, number of
  converged runs). `plot_results.py` / `plot_results2D.py` plot it.

- `check_convergence_from_rmsd.py` — Recompute the per-replica convergence frames from the RMSD series saved in
  `../Data` and compare them with the saved values.

- `RMSD_curve_plotter.py` — Convenience script to compare RMSD curves across grouped runs:
  - Groups folders by prefix before the first underscore, filters invalid runs, and plots group-wise RMSD overlays.

Requirements:
- Python 3.12 (tested), see ../requirements.txt
- numpy, scipy, matplotlib, pandas, seaborn

Quick examples:

Analyze a PMF history with counts:
```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count --reference-pmf reference_filtered.pmf
```

Build a robust reference PMF from subdirectories:
```bash
python buildref.py --dir /path/to/base_dir --npoints 100 --temp 300
```

RMSD grouping plot:
```bash
python RMSD_curve_plotter.py
# edit the top-of-file configuration for parent_dir, filenames and thresholds
```

For details on formats and options, consult the headers of each script (`--help` where supported).
