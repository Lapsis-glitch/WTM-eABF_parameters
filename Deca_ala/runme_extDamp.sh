#!/bin/bash

# Define values to substitute
values=(0.05 0.10 0.20 0.30 0.50 0.70 1.0 1.5 2.0 5.0 )

# Define source folder and placeholder string
src_folder="reference_extDamp"
placeholder="__VALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="extDamp_${val}"

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

