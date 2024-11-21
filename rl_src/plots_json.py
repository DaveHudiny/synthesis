# Author: David Hudák
# Description: This file contains plot functions for new json evaluation results of RL experiments.

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ast


class PreviousStats:
    """Class for storing previous statistics."""

    def __init__(self, best_return_spaynt = None, best_reach_probs_spaynt = None, best_return_rl = None, 
                 best_reach_probs_rl = None, best_return_unstable = None, best_reach_probs_unstable = None):
        self.best_return_spaynt = best_return_spaynt
        self.best_reach_probs_spaynt = best_reach_probs_spaynt
        self.best_return_rl = best_return_rl
        self.best_reach_probs_rl = best_reach_probs_rl
        self.best_return_unstable = best_return_unstable
        self.best_reach_probs_unstable = best_reach_probs_unstable


dict_of_prev_stats = {
    "evade": PreviousStats(-24.999, 1.0, None, 0.698, None, 1.0),
    "evade-n=5-r=23": PreviousStats(-17.0, 1.0, -20.600, 1.0, -16.095),
    "evade-n6-r2": PreviousStats(-21.0, 1.0, -28.0, 1.0, None, None),
    "grid-large-10-5": PreviousStats(-38.719, 1.0, -36.65, 1.0, None, None),
    "grid-large-30-5": PreviousStats(None, None, None, 1.0),
    "intercept": PreviousStats(-15.060, 1.0, -13.45, 1.0, None, None),
    "intercept-n7-r1": PreviousStats(-15.453, 1.0, -16.9, 1.0),
    "mba": PreviousStats(-6.265, 1.0, -5.5, 1.0, None, None),
    "mba-small": PreviousStats(-4.598, 1.0, -3.45, 1.0),
    "network-3-8-20": PreviousStats(-11.284, 1.0, -8.399, 1.0, -3.231, 1.0),
    "obstacle": PreviousStats(-4.2825, 1.0, -34.9, 1.0),
    "obstacles-uniform": PreviousStats(-14.758, 1.0, -49.55, 1.0),
    "refuel-10": PreviousStats(None, 0.457, None, 0.2, None, 0.375),
    "refuel-20": PreviousStats(None, 0.296, None, 0.0, None, 0.3),
    "rocks-16": PreviousStats(-36.911, 1.0, -32.5, 1.0),
    "super-intercept": PreviousStats(None, None, None, 0.85),
    "geo-2-8": PreviousStats(None, 0.616, None, 0.8),
    "network-5-10-8": PreviousStats(-16.050, 1.0, -13.0, 1.0, -10.898, 1.0),
    "rocks-4-20": PreviousStats(-76.0, 1.0, -75.9, 1.0)
}

METRICS = ["returns", "returns_episodic",
           "reach_probs", "losses", "combined_variance"]


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


def plot_single_curve(data, shown_metric, is_trap=False):
    data = ast.literal_eval(data)
    numpy_data = np.array(data).astype(np.float32)
    print(shown_metric)
    plt.plot(numpy_data, label=shown_metric,
             linestyle='dashed' if is_trap else 'solid',
             color='orange' if is_trap else 'blue')    

def add_line_to_plot(model, metric):
    if model in dict_of_prev_stats:
        if metric == "returns":
            if dict_of_prev_stats[model].best_return_spaynt is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_return_spaynt, color='r', linestyle='dashed')
                plt.plot([], label='Best of (S)PAYNT', color='r', linestyle='dashed')
            if dict_of_prev_stats[model].best_return_rl is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_return_rl, color='g', linestyle='dashed')
                plt.plot([], label='Best of Previous RL', color='g', linestyle='dashed')
            if dict_of_prev_stats[model].best_return_unstable is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_return_unstable, color='pink', linestyle='dotted')
                plt.plot([], label='Best of Somewhere', color='pink', linestyle='dotted')
        elif metric == "reach_probs":
            if dict_of_prev_stats[model].best_reach_probs_spaynt is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_reach_probs_spaynt, color='r', linestyle='dashed')
                plt.plot([], label='Best of (S)PAYNT', color='r', linestyle='dashed')
            if dict_of_prev_stats[model].best_reach_probs_rl is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_reach_probs_rl, color='g', linestyle='dashed')
                plt.plot([], label='Best of Previous RL', color='g', linestyle='dashed')
            if dict_of_prev_stats[model].best_reach_probs_unstable is not None:
                plt.axhline(y=dict_of_prev_stats[model].best_reach_probs_unstable, color='pink', linestyle='dotted')
                plt.plot([], label='Best of Somewhere', color='pink', linestyle='dotted')
        

    # add legend to axhline plots
            

def plot_single_metric_for_model(jsons, metric, model, save_folder):
    plt.figure(figsize=(6, 4))
    try:
        for key in jsons:
            model_name, algorithm_name = get_experiment_setting_from_name(key)
            if model_name == model and not metric == "reach_probs":
                data = jsons[key][metric]
                plot_single_curve(data, algorithm_name)
                add_line_to_plot(model, metric)

            elif model_name == model and metric == "reach_probs":
                data = jsons[key][metric]
                try:
                    trap_data = jsons[key]["trap_reach_probs"]
                    # plot_single_curve(
                    #     trap_data, f"Trap Reachability", is_trap=True)
                except:
                    pass
                plot_single_curve(data, "Goal Reachability")
                add_line_to_plot(model, metric)
            
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
    for key in jsons:  # Get all models
        model_name, _ = get_experiment_setting_from_name(key)
        models.add(model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for model in models:  # Plot all metrics for each model
        for metric in METRICS:
            plot_single_metric_for_model(jsons, metric, model, save_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="experiments_original_boosted")
    parser.add_argument("--save_folder", type=str, default="./plots_original")
    
    args = parser.parse_args()
    run_plots(args.folder, args.save_folder)

# run_plots("experiments_original_boosted", "./plots_original")
