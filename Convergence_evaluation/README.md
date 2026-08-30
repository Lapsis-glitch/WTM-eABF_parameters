# Convergence evaluation

The analysis code separates numerical calculations, input discovery, plotting,
and output management. All analyses consume normalized `RunRecord` objects
from `input_discovery.py` and write beneath one output root.

## Publication figures

Use the installed package in the `main` environment:

```python
import pubready as pr
```

The shared plotting configuration defaults to ACS PubReady geometry:

- multipanels: `publisher=acs`, `target=si`, `fraction=full`;
- standalone panels: `publisher=acs`, `target=double`, `fraction=quarter`.

Figures are saved as PDF and PNG by default (`--figure-formats pdf png`) at
300 DPI for PNG. Every analysis that renders plots writes a multipanel and
re-renders each constituent panel through the same panel function into
`Figures/panels/`; standalone panels are never cropped from the multipanel.
Multipanels use shared figure-level labels when panels represent the same
quantity, PubReady-aware renderer spacing, and at most two columns under the
default ACS `si/full` geometry. Standalone panels retain their own labels and
use the unchanged ACS `double/quarter` geometry. RMSD legends use two columns
and a line sample one third of the active Matplotlib/PubReady default length.

## Input modes

Automatic discovery is recursive and configurable:

```bash
conda run -n main python Convergence_evaluation/folder_parser.py data \
  --pmf-pattern '**/*czar.pmf' --count-pattern '**/*count*'
```

PMF/count pairs are matched by their compatible file stems in the same
directory. Diagnostics report zero matches, unmatched files, and ambiguous
pairs. Legacy names such as `output/abf_00...` remain discoverable when their
files match the configured patterns, but folder names are not required.

Explicit files are supported for single-run analyses:

```bash
python Convergence_evaluation/analyze_ND.py path/to/history.pmf path/to/counts
```

A CSV or TSV manifest is the most flexible mode. Paths are relative to the
manifest file and arbitrary parameter names are supported:

```text
run_id,pmf_file,count_file,group,seed,parameter_name,parameter_value
alpha,data/a.czar.pmf,data/a.count,set-A,1,spring_constant,0.5
```

For multiple dimensions, use `parameter_values` JSON or columns such as
`parameter_temperature` and `parameter_lambda`.

## Results hierarchy

Use `--output-root PATH` to override the default `Results/`. Generated files
never go beside input data by default:

```text
input data
   │
   ▼
general discovery / manifest
   │
   ▼
normalized run records
   ├───────────────┬────────────────┐
   ▼               ▼                ▼
convergence     reference PMF     RMSD analysis
   │               │                │
   ▼               ▼                ▼
Results/<analysis>/...
   │
   └── Figures/
       ├── *_multipanel.pdf   ← ACS / SI / full
       └── panels/*.pdf       ← ACS / double / quarter
```

Stable analysis directories are `pmf_convergence`, `reference_pmf`,
`convergence_summary`, `convergence_surface`, `rmsd_curves`, and
`rmsd_seed_curves`. Optional per-snapshot exports belong in
`Figures/snapshots/` and are distinct from semantic standalone panels.

## Commands

```bash
python Convergence_evaluation/analyze_ND.py PMF COUNT --output-root Results
python Convergence_evaluation/buildref.py --dir runs --output-root Results
python Convergence_evaluation/plot_results.py Results/convergence_summary/convergence_summary.csv
python Convergence_evaluation/plot_results2D.py \
  Results/convergence_summary/convergence_summary.csv \
  --x-parameter temperature --y-parameter lambda
python Convergence_evaluation/RMSD_curve_plotter.py runs --output-root Results
python Convergence_evaluation/RMSD_curve_plotter_seeds.py runs --output-root Results
```

To additionally write a clean median-only reference PMF figure, use
`--simple-reference-plot`. It is saved as
`Results/reference_pmf/Figures/reference_median.pdf` and `.png` by default.
The default x-axis label is `Coordinate`; supply a publication-specific label
when needed:

```bash
python Convergence_evaluation/buildref.py --dir runs --output-root Results \
  --simple-reference-plot --xlabel "Collective variable"
python Convergence_evaluation/buildref.py --dir runs --output-root Results \
  --simple-reference-plot --xlabel 'Distance (Å)'
python Convergence_evaluation/buildref.py --dir runs --output-root Results \
  --simple-reference-plot --xlabel '$z$ (Å)'
```

The simple figure contains only the existing median reference PMF, with no
legend or title. The regular comparison, outlier, PMF files, and diagnostic
panels are still produced in the same run.

`plot_results.py` falls back to unknown parameter names as axis labels. The
2D plotter requires explicit `--x-parameter` and `--y-parameter` selection,
while discovery itself supports any number of metadata dimensions.

## Tests

Run the tests in the PubReady environment:

```bash
conda run -n main pytest -q
```
