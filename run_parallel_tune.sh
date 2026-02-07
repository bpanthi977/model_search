#!/bin/bash

CONFIG_PATH="$1"
N="$2"

if [[ -z "$CONFIG_PATH" || -z "$N" ]]; then
  echo "Usage: $0 <config-path> <number-of-parallel-runs>"
  exit 1
fi

PIDS=()

# Cleanup on Ctrl-C
cleanup() {
  echo "Caught Ctrl-C. Killing all child processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  exit 1
}

# Trap SIGINT (Ctrl-C)
trap cleanup INT

# Launch N parallel runs
for ((i=1; i<=N; i++)); do
  echo "Starting process $i/$N..."
  python main.py --config "$CONFIG_PATH" --tune --postgres &
  PIDS+=($!)
done

# Wait for all to finish
wait
