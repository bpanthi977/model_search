#!/bin/sh

folder=$1
if [ -z "$folder" ]; then
    echo "Provide directory as first argument. And optionally an integer as second argument."
    exit 1
fi

if [ -d "$folder" ]; then
    true
else
    echo "Invalid directory." $folder
    exit 1
fi

if [ $# -ge 2 ] && echo "$2" | grep -Eq '^-?[0-9]+$'; then
    # $2 is an integer
    dir=$(find "$folder" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" \
        | sort -n -r \
        | sed -n "${2}p" \
        | cut -d' ' -f2-)

    if [ -z "$dir" ]; then
        echo "There are fewer than $2 directories in $folder"
        exit 1
    fi
else
    dir="$folder"
fi

tail -f "$dir/train_loss.csv" "$dir/val_loss.csv" "$dir/stdout.txt"