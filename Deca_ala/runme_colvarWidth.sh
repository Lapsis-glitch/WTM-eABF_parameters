#!/bin/bash

# Define values to substitute
values=(0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0)

# Define source folder and placeholder string
src_folder="reference_colvarWidth"
placeholder="__VALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="colvarWidth_${val}"

    # Copy the folder
    cp -r "$src_folder" "$dest_folder"

    # Replace placeholder in all text files
    sed -i "s/$placeholder/$val/g" "$dest_folder/colvar.in"

    echo "✅ Created $dest_folder with value $val"
    
    cd "$dest_folder"
    
    echo "Running NAMD"
    namd3 abf.in > namd.log 
    cd ..
done

