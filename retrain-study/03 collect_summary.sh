#/bin/bash

OUTPUT_FILE=summary.csv
echo "run,dataset,is_retrain,epoch,grid_size,mae,val_loss,execution_time" > $OUTPUT_FILE

for folder in */*/; do
    folder=${folder%/}
    [ -f "$folder/config.yaml" ] || continue

    dataset=$(grep "db_file:" "$folder/config.yaml" | sed 's/.*db_file: //')
    is_retrain="False"
    [ -f "$folder/continue_from" ] && is_retrain="True"

    for run_path in "$folder"/e*-s*; do
        [ -d "$run_path" ] || continue
        
        run=$(basename "$run_path")
        epoch=$(echo $run | sed -E 's/e([0-9]+)-s[0-9]+.*/\1/')
        grid=$(echo $run | sed -E 's/e[0-9]+-s([0-9]+).*/\1/')

        mae_file="$run_path/energy_mae.txt"
        time_file="$run_path/execution_time.txt"
        val_csv="$folder/val_loss.csv"
        
        mae=""
        [ -f "$mae_file" ] && mae=$(grep "Avg ABS Diff:" "$mae_file" | sed 's/.*Avg ABS Diff: //')
        
        val_loss=""
        [ -f "$val_csv" ] && val_loss=$(awk -F',' -v ep="$epoch" '$1 == ep {print $2}' "$val_csv")

        exec_time=""
        [ -f "$time_file" ] && exec_time=$(head -n 1 "$time_file" | awk '{print $1}')
        
        if [ -n "$mae" ]; then
            echo "$folder,$dataset,$is_retrain,$epoch,$grid,$mae,$val_loss,$exec_time" >> $OUTPUT_FILE
        fi
    done
done
