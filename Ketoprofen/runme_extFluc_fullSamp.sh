#!/bin/bash
# Stage 2: extendedFluctuation x fullSamples grid at biasTemperature = 4000 K
# (the Stage 1 choice), 5 seed replicas per cell (4 x 4 x 5 = 80 runs).
#
# extendedFluctuation is in CV units (A; the CV width is 0.2 A).
# Folder names keep a literal '.' so Convergence_evaluation/folder_parser_2D.py
# can parse them.
#
# Usage:
#   ./runme_extFluc_fullSamp.sh          # create folders and run NAMD
#   ./runme_extFluc_fullSamp.sh --dry    # create folders only
set -euo pipefail

# Define values to substitute
extflucs=(0.05 0.1 0.2 0.5)
fullsamps=(500 2000 5000 10000)
seeds=(10 20 30 40 50)

# Define source folder
src_folder="reference_extFluc_fullSamp"

run=1
[[ "${1:-}" == "--dry" ]] && run=0

for ef in "${extflucs[@]}"; do
    for fs in "${fullsamps[@]}"; do
        for seed in "${seeds[@]}"; do
            dest_folder="extFluc_${ef}_fullSamp_${fs}_seed_${seed}"
            if [[ -e "$dest_folder" ]]; then
                echo "skip: $dest_folder already exists"
                continue
            fi

            cp -r "$src_folder" "$dest_folder"
            mkdir -p "$dest_folder/output"

            sed -i "s/__EXTFLUC__/${ef}/g"  "$dest_folder/colvar.in"
            sed -i "s/__FULLSAMP__/${fs}/g" "$dest_folder/colvar.in"
            sed -i "s/__SEED__/${seed}/g"   "$dest_folder/abf.in"

            echo "✅ Created $dest_folder"

            if (( run )); then
                echo "Running NAMD"
                (cd "$dest_folder" && namd3 +p8 abf.in > namd.log)
            fi
        done
    done
done
