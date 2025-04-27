#!/bin/bash

# Activate virtual environment
source ../prerequisites/venv/bin/activate

# Define arrays and parameters
list_of_learning_rates=(0.00016)
list_of_batch_sizes=(256)
# model_memory_sizes=(2 5 10)
use_rnn_less=(True)
# use_rnn_less=(True False)
list_of_model_paths=("models_minimal")
num_of_iterations=3

# Trap signals and clean up child processes
trap "kill 0; exit" SIGINT SIGTERM

# Start the experiment loop
for iteration in $(seq 1 $num_of_iterations); do
    echo "Starting iteration $iteration"
    for learning_rate in "${list_of_learning_rates[@]}"; do
        echo "Starting experiment with learning rate $learning_rate"
        for batch_size in "${list_of_batch_sizes[@]}"; do
            echo "Starting experiment with batch size $batch_size"
            for model_path in "${list_of_model_paths[@]}"; do
                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path --use-rnn-less $1"
                python3 experiment_runner.py --learning-rate=$learning_rate --batch-size=$batch_size --path-to-models=$model_path --use-rnn-less $1  &
                pid1=$!  # Capture the process ID of the first experiment

                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path $1"
                python3 experiment_runner.py --learning-rate=$learning_rate --batch-size=$batch_size --path-to-models=$model_path $1 &
                pid2=$!  # Capture the process ID of the second experiment

                # Wait for both child processes to finish
                wait $pid1
                wait $pid2
            done
        done
    done
done

# Start the experiment loop
pids=()

# for iteration in $(seq 1 $num_of_iterations); do
    
#     for model_path in "${list_of_model_paths[@]}"; do
#         pids=()
#         for model_memory_size in "${model_memory_sizes[@]}"; do
#             echo "Running experiment_runner.py with --model-memory_size=$model_memory_size --path-to-models=$model_path --use_rnn_less"
#             python3 experiment_runner.py --model-memory-size=$model_memory_size --path-to-models=$model_path  --use-rnn-less --model-condition drone &
#             pids+=($!)  # Capture the process ID of the first experiment
#         done
#         wait ${pids[@]}
#         pids=()
#         echo "Running experiment_runner.py with --model_memory_size=$model_memory_size --path-to-models=$model_path --use-rnn-less"
#         python3 experiment_runner.py --path-to-models=$model_path --use-rnn-less --model-condition drone &
#         pids+=($!)  # Capture the process ID of the first experiment
#         echo "Running experiment_runner.py with --path_to_models=$model_path"
#         python3 experiment_runner.py --path-to-models=$model_path --model-condition drone &
#         pids+=($!)  # Capture the process ID of the first experiment
#         wait ${pids[@]}
#     done
# done
