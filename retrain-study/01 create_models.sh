#!/bin/sh


for folder in */; do
    python /mnt/SharedOne/bpanthi/model_search/main.py --create-model --checkpoint $folder/
done
