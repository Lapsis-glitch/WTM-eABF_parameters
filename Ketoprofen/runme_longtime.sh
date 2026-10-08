#!/bin/bash
# Long reference run used to build the ketoprofen reference PMF:
# biasTemperature 4000 K, extendedFluctuation 0.1, fullSamples 5000, seed 40,
# up to 2e9 steps (4 us at 2 fs). All values are set in reference_longtime/.
#
# The production run was done in restart segments (1e9 steps, then continued
# toward 2e9 from its own restart files and colvars state); the reference PMF
# was taken at ~3.4 us. NAMD3 reads numSteps as int32, so 2e9 is close to the
# ceiling (2^31 - 1).
#
# Usage:
#   ./runme_longtime.sh          # create folder and run NAMD
#   ./runme_longtime.sh --dry    # create folder only
set -euo pipefail

src_folder="reference_longtime"
dest_folder="long-time_4000_seed40"

run=1
[[ "${1:-}" == "--dry" ]] && run=0

if [[ -e "$dest_folder" ]]; then
    echo "skip: $dest_folder already exists"
    exit 0
fi

cp -r "$src_folder" "$dest_folder"
mkdir -p "$dest_folder/output"

echo "✅ Created $dest_folder"

if (( run )); then
    echo "Running NAMD"
    (cd "$dest_folder" && namd3 +p8 abf.in > namd.log)
fi
