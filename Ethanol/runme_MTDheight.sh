#!/bin/bash

# Define values to substitute
values=(0.005 0.010 0.015 0.020 0.030 0.050 0.100 0.200 0.500 1.00  )

# Define source folder and placeholder string
src_folder="reference_MTDheight"
placeholder="__VALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="MTDheight_${val}"

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

