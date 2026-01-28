Absolutely — here is the **full README in clean Markdown**, ready for copy‑paste into your repo.  
No extra commentary, no formatting glitches, just a clean, production‑ready document.

---

```markdown
# PMF Convergence & Reference Tools

![Plots helping evaluate convergence](plots.png)

Command-line tools for generating PMF convergence and comparison plots from trajectory data, and for building reference PMFs from multiple simulations.

- `analyze_ND.py`: visualize convergence of a **PMF history** against a reference (final PMF, sliding window, or external PMF).
- `buildref.py`: build median and averaged reference PMFs (with outlier filtering) from multiple PMF files.

It visualizes:

- RMSD convergence
- Sequential PMFs
- Sampling evolution
- Post-convergence vs final PMF profiles

---

## Features

- Convergence detection with:
  - RMSD threshold on raw RMSD
  - Slope threshold on fitted RMSD
  - Optional sampling-based criterion (counts std)
- Support for:
  - Sliding-window RMSD
  - RMSD to final PMF
  - RMSD to an external reference PMF
- Toggle minima/maxima/plateau annotations on or off
- Customize font sizes for axes and annotations
- Save figures in PNG, PDF, or other formats at specified DPI
- Build reference PMFs with interpolation and outlier rejection

---

## Requirements

- Python 3.6 or higher  
- NumPy  
- SciPy  
- Matplotlib  

Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

### Convergence analysis (`analyze_ND.py`)

Run the script with your **PMF history** and **sampling counts history** as positional arguments.  
The PMF and counts files must contain a **history** of the simulation (multiple snapshots), not just the final PMF/counts:

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count [OPTIONS]
```

The PMF file can also be a **single PMF** (non-`hist`), in which case it is treated as a 1‑snapshot “history”.  
The reference PMF (if provided) is always a **single PMF** in the new header format.

---

## Command-Line Options (analyze_ND.py)

| Flag                   | Type    | Default | Description                                                                                 |
|------------------------|---------|---------|---------------------------------------------------------------------------------------------|
| `--font-size`          | int     | 14      | Base font size (pt) for titles, labels, ticks                                              |
| `--annotation-fs`      | int     | None    | Font size (pt) for annotation text; falls back to `--font-size`                            |
| `--no-annotations`     | flag    | False   | Disable all minima/maxima/plateau annotations                                              |
| `--save-path`          | str     | None    | File path (with extension) to save the figure                                              |
| `--dpi`                | int     | 300     | Resolution (dots per inch) for the saved figure                                            |
| `--conv-threshold`     | float   | 0.01    | Slope cutoff for convergence detection on the fitted RMSD                                  |
| `--rmsd-threshold`     | float   | None    | Fixed RMSD threshold on **raw** RMSD; if set, overrides slope-based logic                  |
| `--n-recent`           | int     | 10      | Number of PMF snapshots used for sliding-window RMSD (if `--use-sliding-window` is set)    |
| `--use-sliding-window` | flag    | False   | Use sliding-window RMSD instead of RMSD to final or reference PMF                          |
| `--counts-std-thresh`  | float   | None    | Additional convergence criterion based on std of the sampling counts                       |
| `--reference-pmf`      | str     | None    | Path to an external **single** PMF used as RMSD reference (new header format)              |

---

## Reference marginal PMF builder (`buildref.py`)

Builds median, average, and outlier-filtered reference PMFs from multiple PMF files in subdirectories, with interpolation to a common grid.

```bash
python buildref.py --dir /path/to/base_dir [OPTIONS]
```

### Command-Line Options (buildref.py)

| Flag        | Type  | Default     | Description                                         |
|-------------|-------|-------------|-----------------------------------------------------|
| `--dir`     | str   | (required)  | Base directory containing subdirectories with PMFs  |
| `--temp`    | float | 300         | Temperature in Kelvin                               |
| `--name`    | str   | abf_00.abf1 | Prefix of PMF files                                 |
| `--npoints` | int   | 100         | Number of grid points per dimension for interpolation |

---

## Examples

### Build a reference PMF from multiple windows

```bash
python buildref.py --dir /home/user/decaalanine/
```

### Quick convergence plot with default settings

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count
```

### Increase text size and save as high-res PNG

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count \
  --font-size 16 \
  --save-path pmf_overview.png \
  --dpi 600
```

### Stricter convergence threshold and shorter sliding window

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count \
  --conv-threshold 0.001 \
  --n-recent 5 \
  --use-sliding-window \
  --save-path pmf_report.pdf
```

### Use an external reference PMF

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count \
  --reference-pmf reference_average_filtered.pmf \
  --save-path pmf_vs_reference.png
```

### Disable annotations for a cleaner overlay

```bash
python analyze_ND.py pmf.hist.czar.pmf counts.hist.count --no-annotations
```
```

---

If you want, I can also generate:

- a **short version** for PyPI  
- a **GitHub‑friendly version** with badges  
- or a **docs/ folder** with examples and images

Just tell me what direction you want to take it.