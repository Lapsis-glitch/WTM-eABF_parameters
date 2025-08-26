#!/bin/bash

# Define values to substitute
values=(10 20 30 50 100 150 200 250 300 500)

# Define source folder and placeholder string
src_folder="reference_extTime"
placeholder="__VALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="extTime_${val}"

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

