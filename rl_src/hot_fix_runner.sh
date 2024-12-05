#!/bin/bash

# Activate virtual environment
source ../prerequisites/venv/bin/activate

# Define arrays and parameters
list_of_learning_rates=(0.00005 0.0001)
list_of_batch_sizes=(256 512)
list_of_model_paths=("models_selected")
num_of_iterations=10

# Trap signals and clean up child processes
trap "echo 'Terminating script...'; kill 0; exit" SIGINT SIGTERM

# Start the experiment loop
for iteration in $(seq 1 $num_of_iterations); do
    for learning_rate in "${list_of_learning_rates[@]}"; do
        for batch_size in "${list_of_batch_sizes[@]}"; do
            for model_path in "${list_of_model_paths[@]}"; do
                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path --random_start_simulator"
                python3 experiment_runner.py --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path --random_start_simulator &
                pid1=$!  # Capture the process ID of the first experiment

                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path"
                python3 experiment_runner.py --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path &
                pid2=$!  # Capture the process ID of the second experiment

                # Wait for both child processes to finish
                wait $pid1
                wait $pid2
            done
        done
    done
done