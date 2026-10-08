#!/bin/bash
# 2D grid fullSamples x extendedFluctuation: one folder per pair and seed from reference_fullSamp_extFluc/
# (namd3 must be on PATH).
fullSampValues=(100 500 2000 5000 10000)
extFlucValues=(0.01 0.10 0.2 0.5 2.0)
seeds=(10 20 30 40 50 60 70 80 90 100)

max_jobs=4
job_count=0
src_folder="reference_fullSamp_extFluc"

for sampval in "${fullSampValues[@]}"; do
    for flucval in "${extFlucValues[@]}"; do
        for seed in "${seeds[@]}"; do
            (
                dest_folder="fullSamp_${sampval}_extFluc_${flucval}_seed_${seed}"
                cp -r "$src_folder" "$dest_folder"
                mkdir -p "$dest_folder/output"
                sed -i "s/__SAMPVALUE__/$sampval/g" "$dest_folder/colvar.in"
                sed -i "s/__FLUCVALUE__/$flucval/g" "$dest_folder/colvar.in"
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
done
wait
