#!/bin/bash

source ../prerequisites/venv/bin/activate

list_of_learning_rates=(0.001)
list_of_batch_sizes=(256 512)
list_of_model_paths=("models_large" "models")

num_of_iterations=10

for iteration in $(seq 1 $num_of_iterations); do
    for learning_rate in "${list_of_learning_rates[@]}"; do
        for batch_size in "${list_of_batch_sizes[@]}"; do
            for model_path in "${list_of_model_paths[@]}"; do
                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path --random_start_simulator"
                python3 experiment_runner.py --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path --random_start_simulator &
                echo "Running experiment_runner.py with --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path"
                python3 experiment_runner.py --learning_rate=$learning_rate --batch_size=$batch_size --path_to_models=$model_path &
                wait
            done
        done
    done
done

exit 0
