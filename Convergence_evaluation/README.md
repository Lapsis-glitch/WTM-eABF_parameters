# Convergence_evaluation
Toolkit for analysing WTM-eABF PMF convergence and building robust reference PMFs from multiple simulation replicas.
## Modules
| File | Purpose |
|---|---|
| `pmf_io.py` | Read / write / interpolate PMF files (header and multi-block formats). |
| `analyze_ND.py` | `PMFAnalyzer` class — N-D RMSD convergence analysis, feature detection, multi-panel plotting. |
| `reference_builder.py` | Compute median, average, and outlier-filtered reference PMFs from replicas. |
| `plotting_config.py` | Shared Matplotlib style and parameter-label mappings. |
| `buildref.py` | CLI — scan subdirectories, interpolate PMFs, and build reference files. |
| `folder_parser.py` | CLI — batch convergence analysis across a directory tree (1-D and 2-D parameter sweeps). |
| `plot_results.py` | CLI — plot 1-D convergence results from `results.dat`. |
| `plot_results2D.py` | CLI — plot 2-D heatmap of convergence results from `results.dat`. |
| `RMSD_curve_plotter.py` | CLI — overlay RMSD curves from grouped simulation folders (glob-based). |
| `RMSD_curve_plotter_seeds.py` | CLI — overlay RMSD curves from seeded replicas (`Prefix_Value_seed_N`). |
## Requirements
- Python 3.10+
- numpy, scipy, matplotlib, seaborn, pandas
```bash
pip install numpy scipy matplotlib seaborn pandas
```
## Quick start
### Analyse a PMF history
```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count \
    --reference-pmf reference_filtered.pmf \
    --conv-threshold 0.01
```
### Build a robust reference PMF
```bash
python buildref.py --dir /path/to/base_dir --npoints 100 --temp 300
```
### Batch convergence (1-D sweep)
```bash
python folder_parser.py /path/to/runs --reference-pmf ref.pmf
```
### Batch convergence (2-D sweep)
```bash
python folder_parser.py /path/to/runs --reference-pmf ref.pmf --mode 2d
```
### Plot results
```bash
python plot_results.py results.dat --divisor 20
python plot_results2D.py results.dat --divisor 20
```
### RMSD curve overlays
```bash
python RMSD_curve_plotter.py --pattern "MTDwidth*" \
    --pmf output/abf_00.abf1.hist.czar.pmf \
    --counts output/abf_00.abf1.hist.zcount \
    --reference reference_median.pmf
python RMSD_curve_plotter_seeds.py --dir ./ \
    --pmf output/window1.abf1.hist.czar.pmf \
    --counts output/window1.abf1.hist.zcount \
    --divisor 5
```
## Custom fonts
Set the `MATPLOTLIB_FONT_PATH` environment variable to a `.ttf` file to use a custom font across all plots:
```bash
export MATPLOTLIB_FONT_PATH=/path/to/arial.ttf
```
If unset, the system default sans-serif font is used.
## Convergence detection modes
`PMFAnalyzer` supports several convergence strategies, selectable via constructor / CLI flags:
| Flag | Mode |
|---|---|
| `--reference-pmf` | RMSD to an external reference PMF |
| *(default)* | RMSD to the final PMF snapshot |
| `--use-sliding-window` | RMSD to the rolling mean of the last `n_recent` snapshots |
| `--rmsd-threshold` | Fixed RMSD cutoff |
| `--use-ref-and-slope` | Combined: RMSD < threshold **and** |slope| < threshold |
| `--counts-std-thresh` | Additional sampling-uniformity requirement |
For full option lists, run any script with `--help`.
