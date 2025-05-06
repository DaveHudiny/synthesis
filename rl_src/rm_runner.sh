#!/bin/bash

source ../prerequisites/venv/bin/activate

model_names=("refuel-10" "refuel-20")
experiment_flags=("" "--go-explore" "--curiosity-automata-reward" "--predicate-automata-obs")

nr_runs=5


for model_name in "${model_names[@]}"; do
    for run in $(seq 1 $nr_runs); do
        for flags in "${experiment_flags[@]}"; do
            echo "Running model: $model_name, run: $run, flags: $flags"
            python3 experiment_runner.py --model-condition $model_name $flags
        done
    done
done


