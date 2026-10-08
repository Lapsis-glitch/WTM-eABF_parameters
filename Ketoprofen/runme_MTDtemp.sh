#!/bin/bash
# Stage 1: biasTemperature scan, 3 seed replicas each (12 runs).
# extendedFluctuation and fullSamples are fixed at 0.1 and 5000 in reference/.
#
# Usage:
#   ./runme_MTDtemp.sh          # create folders and run NAMD
#   ./runme_MTDtemp.sh --dry    # create folders only
set -euo pipefail

# Define values to substitute
values=(1000 2000 4000 8000)
seeds=(10 20 30)

# Define source folder and placeholder string
src_folder="reference"
placeholder="__TEMPVALUE__"

run=1
[[ "${1:-}" == "--dry" ]] && run=0

for val in "${values[@]}"; do
    for seed in "${seeds[@]}"; do
        dest_folder="biastemp_${val}_seed_${seed}"
        if [[ -e "$dest_folder" ]]; then
            echo "skip: $dest_folder already exists"
            continue
        fi

        cp -r "$src_folder" "$dest_folder"
        mkdir -p "$dest_folder/output"

        sed -i "s/$placeholder/${val}.0/g" "$dest_folder/colvar.in"
        sed -i "s/__SEED__/${seed}/g"      "$dest_folder/abf.in"

        echo "✅ Created $dest_folder with value $val, seed $seed"

        if (( run )); then
            echo "Running NAMD"
            (cd "$dest_folder" && namd3 +p8 abf.in > namd.log)
        fi
    done
done
