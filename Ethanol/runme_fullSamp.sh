#!/bin/bash

# Define values to substitute
values=(50 100 200 300 500 1000 2000 3000 5000 10000)

# Define source folder and placeholder string
src_folder="reference_fullSamp"
placeholder="__VALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="fullSamp_${val}"

    # Copy the folder
    cp -r "$src_folder" "$dest_folder"

    # Replace placeholder in all text files
    sed -i "s/$placeholder/$val/g" "$dest_folder/colvar.in"

    echo "✅ Created $dest_folder with value $val"
    
    cd "$dest_folder"
    
    echo "Running NAMD"
    namd3 +devices 1 abf.in > namd.log 
    cd ..
done

