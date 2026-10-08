#!/bin/bash
# Deca-alanine 1D sweep of fullSamp, repeated with fullSamples = 5000 (10 ns per run, 10 seeds).

# Define values to substitute
values=(50 100 200 300 500 1000 2000 3000 5000 10000)
seeds=(10 20 30 40 50 60 70 80 90 100)

# Parallel job limit
max_jobs=4
job_count=0

# Define source folder and placeholder string
src_folder="reference_fullSamp5000_fullSamp"
placeholder="__VALUE__"
placeholder2="__SEED__"

# Loop over values
for val in "${values[@]}"; do
    for seed in "${seeds[@]}"; do
        (
            # Create destination folder name
            dest_folder="fullSamp_${val}_seed_${seed}"

            # Copy the folder
            cp -r "$src_folder" "$dest_folder"
            mkdir -p "$dest_folder/output"

            # Replace placeholder in all text files
            sed -i "s/$placeholder/$val/g" "$dest_folder/colvar.in"
            sed -i "s/$placeholder2/$seed/g" "$dest_folder/abf.in"

            echo "✅ Created $dest_folder with value $val and seed $seed"

            cd "$dest_folder" || exit

            echo "Running NAMD in $dest_folder"
            namd3 +p8 abf.in > namd.log

            cd ..
        ) &  # run in background

        ((job_count++))

        # Limit to max_jobs parallel jobs
        if (( job_count >= max_jobs )); then
            wait
            job_count=0
        fi
    done
done

# Wait for remaining jobs
wait

echo "🎉 All simulations completed!"
