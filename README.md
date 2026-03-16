# WTM-eABF_parameters

Utilities to evaluate PMF convergence and to build reference PMFs for WTM-eABF experiments.

Subpackages:
- `Convergence_evaluation/` — scripts to analyze PMF convergence, build reference PMFs, and visualize RMSD curves.

Quick start:
- Install requirements: `pip install numpy scipy matplotlib`
- Use `analyze_ND.py` to inspect PMF histories and sampling counts.
- Use `buildref.py` to compute median/average reference PMFs from multiple runs.

See `Convergence_evaluation/README.md` for more details.
