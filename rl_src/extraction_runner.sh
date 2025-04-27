#!/bin/bash

source ../prerequisites/venv/bin/activate

max_procs=3

models_dir="models_paynt_experiments"
one_hot_mem_sizes=(3 5 10 20)
tanh_mem_sizes=(1 2 3)
arr_use_residual_connections=(True False)
arr_use_one_hots=(False True)

flags_array=""
procs_pids=()

trap "kill 0; exit" SIGINT SIGTERM

for i in $(seq 1 1); do
    echo "Starting iteration $i"
    for use_one_hot in "${arr_use_one_hots[@]}"; do
        if [[ $use_one_hot == "True" ]]; then
            mem_sizes=("${one_hot_mem_sizes[@]}")
            flags_array="--use-one-hot"
        else
            mem_sizes=("${tanh_mem_sizes[@]}")
            flags_array=""
        fi
        for use_residual in "${arr_use_residual_connections[@]}"; do
            if [[ $use_residual == "True" ]]; then
                flags_array="$flags_array --use-residual-connection"
            else # Remove the flag
                flags_array=$(echo $flags_array | sed -e 's/--use-residual-connection//g')
            fi
            for mem_size in "${mem_sizes[@]}"; do
                for model_path in "$models_dir"/*; do
                    echo "Calling python3 interpreters/fsc_trained_actor.py --prism-path $model_path/sketch.templ --properties-path $model_path/sketch.props --memory-size $mem_size $flags_array"
                    # Uncomment the following line to actually run the Python script
                    python3 interpreters/direct_fsc_extraction/direct_extractor.py --prism-path $model_path --memory-size "$mem_size" $flags_array &
                    procs_pids+=($!)
                    if [ ${#procs_pids[@]} -ge $max_procs ]; then
                        wait "${procs_pids[@]}"
                        procs_pids=()
                    fi
                done
            done
        done
    done
done

# Wait for any remaining processes
if [ ${#procs_pids[@]} -gt 0 ]; then
    wait "${procs_pids[@]}"
fi
