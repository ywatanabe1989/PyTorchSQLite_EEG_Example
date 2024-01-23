#!/bin/bash

# Declare ds_list as a bash array
ds_ID_list=(
    ds002778
    ds003478
    ds004504
    ds003775
)

function download_an_opeanneuro_dataset() {
    ds_ID=$1
    dest_path="./data/${ds_ID}-download/"

    if [[ ! -d $dest_path ]]; then
        echo "Trying to download ${ds_ID}"
        aws s3 sync --no-sign-request "s3://openneuro.org/${ds_ID}" "$dest_path" || echo "${ds_ID} download failed"
    else
        echo "${ds_ID} already exists. Skipping."
    fi
}

export -f download_an_opeanneuro_dataset

# Extract numeric portions of dataset strings and execute doit function in parallel
echo "${ds_ID_list[@]}" | tr ' ' '\n' | parallel -j10 download_an_opeanneuro_dataset

# Remove any empty directories
find . -type d -empty -delete

## EOF
