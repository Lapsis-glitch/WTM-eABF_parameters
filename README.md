# WTM-eABF parameter sensitivity: inputs, analysis code and data

Simulation inputs, convergence-analysis code, processed data and figure scripts for
"WTM-eABF under the Microscope: Parameter Sensitivity and Reproducibility"
(C. Chipot, C. G. Chen, R. A. Talmazan, C. Kang).

## Contents

| Path | What |
|---|---|
| `NANMA/`, `Deca_ala/`, `Ethanol/` | NAMD/Colvars inputs. `reference_<sweep>/` hold the template `abf.in` and `colvar.in` of each sweep, `runme_<sweep>.sh` create one folder per value and seed and run NAMD. Deca-alanine: `*_fullSamp5000_*` = repeat at `fullSamples`=5000, `*_fullSamp_extFluc` = two-dimensional grid, `*_WTMtD_*` = standalone WT-MtD. Ethanol: `reference_2D` = two-dimensional grid. |
| `Ketoprofen/` | NAMD/Colvars inputs of the ketoprofen/POPC benchmark (`runme_MTDtemp.sh` = stage one, `runme_extFluc_fullSamp.sh` = stage two, `runme_longtime.sh` = ~3.4 µs reference run). |
| `Ketoprofen/runs/<run>/output/` | Raw output of all 92 ketoprofen runs and the reference run: final CZAR PMF, Colvars trajectory and xz-compressed PMF history. |
| `Ketoprofen/analysis/` | Ketoprofen PMF analysis (symmetrization, bulk anchoring, accuracy, reproducibility, reference and median PMFs). |
| `Convergence_evaluation/` | Convergence analysis used for NANMA, deca-alanine and ethanol (reference PMF builder, convergence criterion, result tables). |
| `Data/<sweep>/` | Processed data behind every figure and table: per-replica convergence times, running RMSD to the reference PMF, reference PMFs, final PMFs and ξ−λ statistics. Each folder has a `README.txt` with columns, units and frame spacing. |
| `Figures/scripts/` | One script per figure of the main text and the Supporting Information. |
| `Figures/structures/` | Coordinates used for the molecule drawings of Figure 1. |

## Requirements

Python 3.12 with the packages in `requirements.txt` (`pip install -r requirements.txt`). The figures use
Arial when it is installed and fall back to the matplotlib default font otherwise.

## Reproducing the figures

Run from the repository root:

```bash
python Figures/scripts/fig1_benchmark_reference.py   # Figure 1
python Figures/scripts/fig2_deca_1d.py               # Figure 2
python Figures/scripts/fig3_2d_sweeps.py             # Figure 3
python Figures/scripts/fig4_keto.py                  # Figure 4
python Figures/scripts/fig5_wtmtd.py                 # Figure 5
python Figures/scripts/si_sweeps.py                  # SI sweep and RMSD figures
python Figures/scripts/si_coupling.py                # SI xi-lambda figures
python Figures/scripts/si_keto.py                    # SI ketoprofen figures
```

PDFs are written to `Figures/Main/` and `Figures/SI/`, preview PNGs to `build/`. The tables of the
Supporting Information are aggregates of the per-replica files in `Data/`.

## Convergence criterion (NANMA, deca-alanine, ethanol)

The reference PMF is the pointwise median of the final PMFs of all runs of a system, computed in
probability space (`Convergence_evaluation/buildref.py`). For every history PMF of a run, the RMSD to
the reference is computed after shifting both profiles to zero at their minimum. The RMSD series is
smoothed with a Savitzky–Golay filter (11 frames, third order) and fitted with A·exp(−Bt)+C. A run has
converged at the first frame where the RMSD is below kT = 0.5922 kcal/mol and the absolute slope of the
fit is below 0.01 per frame. If no frame meets both conditions, the first frame with RMSD below kT is
used. Runs that never fall below kT, or whose fit fails, are counted as unconverged. The settings are
those of `Convergence_evaluation/folder_parser.py`.

To apply it to a sweep directory with one folder per run (`<param>_<value>_seed_<n>/output/`):

```bash
python Convergence_evaluation/buildref.py --dir <sweep_dir> --name abf_00.abf1 --temp 300   # window1.abf1 for ethanol
python Convergence_evaluation/folder_parser.py <sweep_dir> --reference-pmf <sweep_dir>/reference_median.pmf
python Convergence_evaluation/folder_parser_2D.py <sweep_dir> --reference-pmf ...   # two-parameter grids
python Convergence_evaluation/plot_results.py --help                                 # plot results.dat
```

The raw PMF histories of these systems are not included (several GB). The saved RMSD series in `Data/`
contain everything the criterion uses, and

```bash
python Convergence_evaluation/check_convergence_from_rmsd.py Data
```

feeds them through the same code and reproduces all 3060 per-replica convergence times exactly.

## Ketoprofen

The ketoprofen runs are analysed through the accuracy and reproducibility of their final PMFs against
the ~3.4 µs reference run, with kT = 0.616 kcal/mol at 310 K (see `Data/ketoprofen/README.txt`). To
rebuild `Data/ketoprofen/` from the raw output:

```bash
./unpack_keto.sh                                     # decompress the PMF histories (~4.5 GB)
python Figures/scripts/export_keto_data.py           # rewrites Data/ketoprofen/*.csv
cd Ketoprofen/analysis && python analyze.py          # per-cell accuracy/reproducibility -> results/
```

Both reproduce the shipped files exactly.

## Inputs not included

The NANMA structure and parameter files (`vacuum.psf`, `par_all22_prot.inp`, `equilvaco.coor`, see
`NANMA/common/README.txt`) still need to be added.

## License

GPL-3.0, see `LICENSE`.
