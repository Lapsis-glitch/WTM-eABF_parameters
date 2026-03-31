# WTM-eABF Parameters
Utilities for evaluating PMF convergence and building robust reference PMFs from WTM-eABF (Well-Tempered Metadynamics enhanced Adaptive Biasing Force) simulations.
## Repository layout
```
WTM-eABF_parameters/
├── Convergence_evaluation/   # Python package — analysis & plotting toolkit
│   ├── pmf_io.py             # PMF file I/O and interpolation
│   ├── analyze_ND.py         # PMFAnalyzer: N-D convergence analysis
│   ├── reference_builder.py  # Build median / average / filtered reference PMFs
│   ├── plotting_config.py    # Shared Matplotlib style & label mappings
│   ├── buildref.py           # CLI: build reference PMF from replicas
│   ├── folder_parser.py      # CLI: batch convergence across directory trees
│   ├── plot_results.py       # CLI: 1-D convergence bar plots
│   ├── plot_results2D.py     # CLI: 2-D convergence heatmaps
│   ├── RMSD_curve_plotter.py          # CLI: grouped RMSD overlays
│   └── RMSD_curve_plotter_seeds.py    # CLI: seed-aware RMSD overlays
├── Deca_ala/                 # Deca-alanine simulation inputs & run scripts
├── Ethanol/                  # Ethanol simulation inputs & run scripts
└── output/                   # Sample output figures
```
## Installation
```bash
pip install numpy scipy matplotlib seaborn pandas
```
Python 3.10+ is required.
## Quick start
```bash
# Analyse a PMF history with convergence detection
python Convergence_evaluation/analyze_ND.py \
    pmf.hist.czar.pmf counts.hist.count \
    --reference-pmf reference.pmf --conv-threshold 0.01
# Build a robust reference PMF from multiple runs
python Convergence_evaluation/buildref.py \
    --dir /path/to/runs --npoints 100 --temp 300
# Batch convergence analysis (1-D parameter sweep)
python Convergence_evaluation/folder_parser.py \
    /path/to/runs --reference-pmf ref.pmf
# Plot convergence results
python Convergence_evaluation/plot_results.py results.dat --divisor 20
```
## Custom fonts
All plotting scripts read `$MATPLOTLIB_FONT_PATH` for a custom `.ttf` font.
If unset, the system default sans-serif font is used.
```bash
export MATPLOTLIB_FONT_PATH=/path/to/arial.ttf
```
See [`Convergence_evaluation/README.md`](Convergence_evaluation/README.md) for detailed documentation of every module and CLI option.
