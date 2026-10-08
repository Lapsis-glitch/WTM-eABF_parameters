#!/bin/bash
# Ethanol 2D sweep: fullSamples x extendedFluctuation (5 x 5 values, 10 seeds, 20 ns per run).

# Define values to substitute
fullSampValues=(100 500 2000 5000 10000)
extFlucValues=(0.01 0.10 0.2 0.5 2.0)
seeds=(10 20 30 40 50 60 70 80 90 100)

# Parallel job limit
max_jobs=4
job_count=0

# GPU assignment counter
gpu_id=0
num_gpus=4

# Define source folder and placeholder string
src_folder="reference_2D"
placeholder="__SAMPVALUE__"
placeholder1="__FLUCVALUE__"
placeholder2="__SEED__"

# Loop over values
for sampval in "${fullSampValues[@]}"; do
    for flucval in "${extFlucValues[@]}"; do
        for seed in "${seeds[@]}"; do

            (
                # Create destination folder name
                dest_folder="fullSamp_${sampval}_extFluc_${flucval}_seed_${seed}"

                # Copy the folder
                cp -r "$src_folder" "$dest_folder"
                mkdir -p "$dest_folder/output"

                # Replace placeholder in all text files
                sed -i "s/$placeholder/$sampval/g" "$dest_folder/colvar.in"
                sed -i "s/$placeholder1/$flucval/g" "$dest_folder/colvar.in"
                sed -i "s/$placeholder2/$seed/g" "$dest_folder/abf.in"

                echo "✅ Created $dest_folder with fullSamp=$sampval, extFluc=$flucval, seed=$seed"

                cd "$dest_folder" || exit

                echo "Running NAMD on GPU $gpu_id in $dest_folder"
                namd3 +p2 +devices $gpu_id abf.in > namd.log

                cd ..
            ) &

            ((job_count++))

            # Move to next GPU (0→1→2→3→0…)
            gpu_id=$(( (gpu_id + 1) % num_gpus ))

            # Limit to max_jobs parallel jobs
            if (( job_count >= max_jobs )); then
                wait
                job_count=0
            fi

        done
    done
done

# Wait for remaining jobs
wait

echo "🎉 All simulations completed!"
