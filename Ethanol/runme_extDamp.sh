#!/bin/bash
# Ethanol 1D sweep of extendedLangevinDamping (20 ns per run, 5 seeds).

# Define values to substitute
values=(0.05 0.10 0.20 0.30 0.50 0.70 1.0 1.5 2.0 5.0)
seeds=(10 20 30 40 50)

# Define source folder and placeholder string
src_folder="reference_extDamp"
placeholder="__VALUE__"
placeholder2="__SEED__"

job_count=0
max_jobs=2

for val in "${values[@]}"; do
    for seed in "${seeds[@]}"; do

        current_job=$job_count

        (
            dest_folder="extDamp_${val}_seed_${seed}"
            cp -r "$src_folder" "$dest_folder"
            mkdir -p "$dest_folder/output"

            sed -i "s/$placeholder/$val/g" "$dest_folder/colvar.in"
            sed -i "s/$placeholder2/$seed/g" "$dest_folder/abf.in"

            echo "✅ Created $dest_folder"

            cd "$dest_folder" || exit

            gpu_id=$((current_job % 2))
            echo "Running on GPU $gpu_id"

            namd3 +p2 +devices ${gpu_id} abf.in > namd.log
        ) &

        ((job_count++))

        if (( job_count >= max_jobs )); then
            wait
            job_count=0
        fi

    done
done

wait
