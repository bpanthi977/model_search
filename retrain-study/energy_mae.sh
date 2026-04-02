#!/bin/bash

show_help() {
    echo "Usage: $0 <path_to_model.pt> <grid_size> [experiment_dir]"
    echo "Example: $0 /path/to/model.pt 30"
    echo "Example with custom dir: $0 /path/to/any_model.pt 30 my_custom_dir"
    exit 1
}

if [ $# -lt 2 ] || [[ " $@ " =~ " --help " ]]; then
    show_help
fi

LULESH_DIR=/mnt/SharedOne/bpanthi/LULESH
SCRIPT_DIR=/mnt/SharedOne/bpanthi/model_search

model=$1 # Path to the model.pt file
grid_size=$2
max_iters=$4
echo "All args: " $@
echo "max_iters" $max_iters
if [ ! -f "$model" ]; then
    echo "Error: Model file '$model' not found."
    exit 1
fi

if [ $# -ge 3 ]; then
    experiment_dir=$3
    model_name=$(basename "$(dirname "$model")")
else
    model_name=$(basename "$(dirname "$model")")
    filename=$(basename "$model")
    if [[ "$filename" == "model.pt" ]]; then
	:
    elif [[ "$filename" =~ ^model[-_](.*)\.pt$ ]]; then
	model_name="${model_name}-e${BASH_REMATCH[1]}"
    else
	echo "Error: Model filename '$filename' must be 'model.pt' or match 'model[-_]*.pt'"
	exit 1
    fi
    experiment_dir="${model_name}-${grid_size}"
fi

if [ -f "$experiment_dir/energy_mae.txt" ] && [ -f "$experiment_dir/execution_time.txt" ]; then
    echo "Experiment results already exist in $experiment_dir. Skipping."
    exit 0
fi

mkdir -p "$experiment_dir"
echo "Running experiment: " $experiment_dir

start_time=$(date +%s.%N)
SURROGATE_MODEL=$model ENERGY_DUMP_FILE_NAME=$experiment_dir/Energy-$model_name-$grid_size.bin ENERGY_DUMP_TYPE=last ./lulesh2.0 -p -s $grid_size -i $max_iters
end_time=$(date +%s.%N)

# Store time taken in execution_time.txt file

time_diff=$(echo "$end_time - $start_time" | bc)
total_seconds=${time_diff%.*}
[ -z "$total_seconds" ] && total_seconds=0
h=$((total_seconds / 3600))
m=$(( (total_seconds % 3600) / 60 ))
s=$(( total_seconds % 60 ))
formatted_time=$(printf "%02d:%02d:%02d" $h $m $s)
echo "$time_diff seconds" > "$experiment_dir/execution_time.txt"
echo "$formatted_time" >> "$experiment_dir/execution_time.txt"


source $SCRIPT_DIR/.venv/bin/activate

# Print the output of the following program and store the output in $experiment_dir/energy_mae.txt
python $SCRIPT_DIR/energy.py --original $LULESH_DIR/Energy_Original-$grid_size.bin --model "$experiment_dir/Energy-$model_name-$grid_size.bin" --visualize | tee "$experiment_dir/energy_mae.txt"
