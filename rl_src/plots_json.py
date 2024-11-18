# Author: David Hudák
# Description: This file contains plot functions for new json evaluation results of RL experiments.

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

class PreviousStats:
    """Class for storing previous statistics."""
    def __init__(self, best_return_spaynt, best_reach_probs_spaynt, best_return_rl, best_reach_probs_rl, best_return_unstable, best_reach_probs_unstable):
        self.best_return = best_return_spaynt
        self.best_reach_probs_spaynt = best_reach_probs_spaynt
        self.best_return_rl = best_return_rl
        self.best_reach_probs_rl = best_reach_probs_rl
        self.best_return_unstable = best_return_unstable
        self.best_reach_probs_unstable = best_reach_probs_unstable


dict_of_prev_stats = {
    "evade": PreviousStats(),
    "evade-n=5-r=23": PreviousStats(),
    "evade-n6-r2": PreviousStats(),
    "grid-large-10-5": PreviousStats(),
    "grid-large-30-5": PreviousStats(),
    "intercept": PreviousStats(),
    "intercept-n7-r1": PreviousStats(),
    "mba": PreviousStats(),
    "mba-small": PreviousStats(),
    "network-3-8-20": PreviousStats(),
    "obstacle": PreviousStats(),
    "obstacles-uniform": PreviousStats(),
    "refuel-10": PreviousStats(),
    "refuel-20": PreviousStats(),
    "rocks-16": PreviousStats(),
    "super-intercept": PreviousStats(),
    "geo-2-8": PreviousStats(),
    "network-5-10-8": PreviousStats(),
    "rocks-4-20": PreviousStats()
    }

METRICS = ["returns", "returns_episodic", "reach_probs", "losses", "combined_variance"]

def load_from_json(path):
    with open(path, "r") as file:
        json_dict = json.load(file)
    return json_dict

def load_jsons_from_folder(folder):
    
    jsons = {}
    for file in os.listdir(folder):
        if file.endswith(".json"):
            jsons[file] = load_from_json(os.path.join(folder, file))
    return jsons

def get_experiment_setting_from_name(string):
    splitted = string.split("_")
    model_name = splitted[0]
    algorithm_name = splitted[1]
    if algorithm_name == "Stochastic":
        algorithm_name = "Stochastic_PPO"
    return model_name, algorithm_name

def plot_single_curve(data, learning_algorithm, is_trap=False):
    data = ast.literal_eval(data)
    numpy_data = np.array(data).astype(np.float32)
    print(learning_algorithm)
    plt.plot(numpy_data, label=learning_algorithm, linestyle='dashed' if is_trap else 'solid')
    
def plot_single_metric_for_model(jsons, metric, model, save_folder):
    plt.figure(figsize=(6, 4))
    try:
        for key in jsons:
            model_name, algorithm_name = get_experiment_setting_from_name(key)
            if model_name == model and not metric == "reach_probs":
                data = jsons[key][metric]
                plot_single_curve(data, algorithm_name)
            elif model_name == model and metric == "reach_probs":
                data = jsons[key][metric]
                try:
                    trap_data = jsons[key]["trap_reach_probs"]
                    plot_single_curve(trap_data, f"{algorithm_name}_trap", is_trap=True)
                except:
                    pass
                plot_single_curve(data, algorithm_name)
    except Exception as e:
        print(f"Error in {model} with {metric}: {e}")

    plt.title(f"Graph for {model} with {metric}")
    plt.xlabel("i-th hundred iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{save_folder}/{model}_{metric}.png")
    plt.close()

def run_plots(folder, save_folder):
    jsons = load_jsons_from_folder(folder)
    models = set()
    for key in jsons: # Get all models
        model_name, _ = get_experiment_setting_from_name(key)
        models.add(model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for model in models: # Plot all metrics for each model
        for metric in METRICS:
            plot_single_metric_for_model(jsons, metric, model, save_folder)
            
run_plots("experiments_original_boosted", "./plots_original")
