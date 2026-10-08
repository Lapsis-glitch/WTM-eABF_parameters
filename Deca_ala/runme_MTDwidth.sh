#!/bin/bash
# Creates one folder per value and seed from reference_MTDwidth/ and runs NAMD (namd3 must be on PATH).
values=(0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)
seeds=(10 20 30 40 50 60 70 80 90 100)

max_jobs=4
job_count=0
src_folder="reference_MTDwidth"

for val in "${values[@]}"; do
    for seed in "${seeds[@]}"; do
        (
            dest_folder="MTDwidth_${val}_seed_${seed}"
            cp -r "$src_folder" "$dest_folder"
            mkdir -p "$dest_folder/output"
            sed -i "s/__VALUE__/$val/g" "$dest_folder/colvar.in"
            sed -i "s/__SEED__/$seed/g" "$dest_folder/abf.in"
            cd "$dest_folder" || exit
            namd3 +p8 abf.in > namd.log
        ) &
        ((job_count++))
        if (( job_count >= max_jobs )); then
            wait
            job_count=0
        fi
    done
done
wait
