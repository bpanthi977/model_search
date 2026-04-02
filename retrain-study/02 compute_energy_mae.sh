#!/bin/bash

show_help() {
    echo "Usage: $0 <grid_size> [parallel]"
    echo "Example: $0 30 4"
    exit 1
}

if [ $# -lt 1 ] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

grid_size=$1
max_iter=1
case $grid_size in
    30) max_iter=1500 ;; 
    40) max_iter=2000 ;;
    50) max_iter=2500 ;;
    60) max_iter=2500 ;;
esac     

parallel=${2:-1}
script=./energy_mae.sh

cleanup() {
    # Remove the trap to avoid infinite recursion when signaling the process group
    trap - SIGINT SIGTERM
    echo -e "\nCaught SIGINT/SIGTERM, terminating all child processes in the group..."
    # Kill the entire process group (including descendants)
    kill -TERM 0 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

run_parallel() {
    "$@" &
    while [[ $(jobs -p | wc -l) -ge $parallel ]]; do
        wait -n
    done
}

epochs=(10 20 30 40 50 60 70 80 90 100) # UmHQ, rfbA, + 50 aAiu

# Models #3
for epoch in "${epochs[@]}";
do
    run_parallel $script 30/20260306-020855-UmhQ/model-${epoch}.pt "${grid_size}" 30/20260306-020855-UmhQ/e${epoch}-s${grid_size}/ $max_iter
    if [[ $grid_size == 30 || $grid_size == 50 ]]; then
        run_parallel $script 50/20260309-151938-rfbA/model-${epoch}.pt "${grid_size}" 50/20260309-151938-rfbA/e${epoch}-s${grid_size}/ $max_iter
        run_parallel $script 50/20260305-204235-aAiu/model-${epoch}.pt "${grid_size}" 50/20260305-204235-aAiu/e${epoch}-s${grid_size}/ $max_iter
    fi

    if [[ $grid_size == 30 || $grid_size == 60 ]]; then
	run_parallel $script 60/20260312-160922-FmUU/model-${epoch}.pt "${grid_size}" 60/20260312-160922-FmUU/e${epoch}-s${grid_size}/ $max_iter
	run_parallel $script 60/20260312-170036-ZVXo/model-${epoch}.pt "${grid_size}" 60/20260312-170036-ZVXo/e${epoch}-s${grid_size}/ $max_iter
    fi

    if [[ $grid_size == 30 || $grid_size == 40 ]]; then
	run_parallel $script 40/20260312-174017-FUQV/model-${epoch}.pt "${grid_size}" 40/20260312-174017-FUQV/e${epoch}-s${grid_size}/ $max_iter
	run_parallel $script 40/20260312-185952-SXIL/model-${epoch}.pt "${grid_size}" 40/20260312-185952-SXIL/e${epoch}-s${grid_size}/ $max_iter
    fi
done

# Models #2
epochs=(5 10 15 20 25 30 35 40) # ioQW, tRiJ, + 20 cGPK
for epoch in "${epochs[@]}";
do
    run_parallel $script 30/20260305-195222-ioQW/model-${epoch}.pt "${grid_size}" 30/20260305-195222-ioQW/e${epoch}-s${grid_size}/ $max_iter

    if [[ $grid_size == 30 || $grid_size == 50 ]]; then
        run_parallel $script 50/20260305-192304-tRiJ/model-${epoch}.pt "${grid_size}" 50/20260305-192304-tRiJ/e${epoch}-s${grid_size}/ $max_iter
        run_parallel $script 50/20260305-185353-cGPK/model-${epoch}.pt "${grid_size}" 50/20260305-185353-cGPK/e${epoch}-s${grid_size}/ $max_iter
    fi

    if [[ $grid_size == 30 || $grid_size == 60 ]]; then
	run_parallel $script 60/20260312-175059-ThjA/model-${epoch}.pt "${grid_size}" 60/20260312-175059-ThjA/e${epoch}-s${grid_size}/ $max_iter
	run_parallel $script 60/20260312-182409-QyaY/model-${epoch}.pt "${grid_size}" 60/20260312-182409-QyaY/e${epoch}-s${grid_size}/ $max_iter
    fi

    if [[ $grid_size == 30 || $grid_size == 40 ]]; then
	run_parallel $script 40/20260312-193859-IbpD/model-${epoch}.pt "${grid_size}" 40/20260312-193859-IbpD/e${epoch}-s${grid_size}/ $max_iter
	run_parallel $script 40/20260312-183104-mieu/model-${epoch}.pt "${grid_size}" 40/20260312-183104-mieu/e${epoch}-s${grid_size}/ $max_iter
    fi

done

# Pruning study
thresholds=("t1e-09")
for threshold in "${thresholds[@]}";
do
    run_parallel $script 30/20260305-195222-ioQW_pruned/model_${threshold}_.pt "${grid_size}" 30/20260305-195222-ioQW_pruned/${threshold}_s${grid_size}/ $max_iter
    run_parallel $script 30/20260306-020855-UmhQ_pruned/model_${threshold}_.pt "${grid_size}" 30/20260306-020855-UmhQ_pruned/${threshold}_s${grid_size}/ $max_iter
done    
wait
