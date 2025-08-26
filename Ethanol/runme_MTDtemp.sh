#!/bin/bash

# Define values to substitute
values=(500 1000 1500 2000 3000 4000 5000 8000 10000 15000)

# Define source folder and placeholder string
src_folder="reference"
placeholder="__TEMPVALUE__"

# Loop over values
for val in "${values[@]}"; do
    # Create destination folder name
    dest_folder="biastemp_${val}"

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

