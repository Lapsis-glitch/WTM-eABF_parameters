#!/bin/sh
# Decompress the ketoprofen ABF PMF histories (needed only by Figures/scripts/export_keto_data.py
# and Ketoprofen/analysis/analyze.py). About 4.5 GB once unpacked. Run from the repository root.
find Ketoprofen/runs -name '*.pmf.xz' -exec xz -dk {} \;
