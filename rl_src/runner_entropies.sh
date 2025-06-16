#!/bin/bash

# Activate virtual environment
source ../prerequisites/venv/bin/activate



models_path="models"
models=("maze-10")
num_of_iterations=10

# Trap signals and clean up child processes
trap "kill 0; exit" SIGINT SIGTERM

# Start the experiment loop
for iteration in $(seq 1 $num_of_iterations); do
    echo "Iteration: $iteration"

    echo "First without entropy"
    for model in "${models[@]}"; do
        echo "Running model: $model"
        python3 experiment_runner.py --model-condition "$model" 
    done

    echo "Now with entropy"
    for model in "${models[@]}"; do
        echo "Running model with entropy: $model"
        python3 experiment_runner.py --model-condition "$model" --use-entropy-reward &
        pid_entropy=$!

        echo "Now with full observability"
        python3 experiment_runner.py --model-condition "$model" --use-entropy-reward --full-observable-entropy-reward &
        pid_full_observable=$!

        wait $pid_entropy
        wait $pid_full_observable

        echo "Now with binary entropy reward"
        python3 experiment_runner.py --model-condition "$model" --use-entropy-reward --use-binary-entropy-reward &
        pid_binary_entropy=$!

        echo "Now with binary entropy reward and full observability"
        python3 experiment_runner.py --model-condition "$model" --use-entropy-reward --full-observable-entropy-reward --use-binary-entropy-reward &
        pid_binary_full_observable=$!

        wait $pid_binary_entropy
        wait $pid_binary_full_observable
    done
    echo "Iteration $iteration completed."
done
