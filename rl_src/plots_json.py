# Author: David Hudák
# Description: This file contains plot functions for new json evaluation results of RL experiments.

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

import pandas as pd


class PreviousStats:
    """Class for storing previous statistics."""

    def __init__(self, best_return_spaynt=None, best_reach_probs_spaynt=None, best_return_rl=None,
                 best_reach_probs_rl=None, best_return_unstable=None, best_reach_probs_unstable=None):
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
    "super-intercept": PreviousStats(None, 0.848, None, 0.85),
    "geo-2-8": PreviousStats(None, 0.616, None, 0.8),
    "network-5-10-8": PreviousStats(-16.050, 1.0, -13.0, 1.0, -10.898, 1.0),
    "rocks-4-20": PreviousStats(-76.0, 1.0, -75.9, 1.0),
    "geo-2-8-large": PreviousStats(None, 0.124, None, None),
    "obstacle-large": PreviousStats(25.78, 1.0),
    "intercept-large": PreviousStats(None, 0.78),
    "evade-large": PreviousStats(None, 0.0),
    "maze-10": PreviousStats(),
    "avoid": PreviousStats(),
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


def plot_single_curve(data, shown_metric, is_trap=False, plot_color='b'):
    data = ast.literal_eval(data)
    numpy_data = np.array(data).astype(np.float32)
    # print(shown_metric)
    if plot_color is not None:
        plt.plot(numpy_data, label=shown_metric,
                 linestyle='dashed' if is_trap else 'solid',
                 color='orange' if is_trap else plot_color)
    else:
        plt.plot(numpy_data, label=shown_metric,
                 linestyle='dashed' if is_trap else 'solid')


def add_line_to_plot(model, metric):
    if model in dict_of_prev_stats:
        if metric == "returns":
            if dict_of_prev_stats[model].best_return_spaynt is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_return_spaynt, color='r', linestyle='dashed')
                plt.plot([], label='Best of (S)PAYNT',
                         color='r', linestyle='dashed')
            if dict_of_prev_stats[model].best_return_rl is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_return_rl, color='g', linestyle='dashed')
                plt.plot([], label='Best of Previous RL',
                         color='g', linestyle='dashed')
            if dict_of_prev_stats[model].best_return_unstable is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_return_unstable, color='pink', linestyle='dotted')
                plt.plot([], label='Best of Somewhere',
                         color='pink', linestyle='dotted')
        elif metric == "reach_probs":
            if dict_of_prev_stats[model].best_reach_probs_spaynt is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_reach_probs_spaynt, color='r', linestyle='dashed')
                plt.plot([], label='Best of (S)PAYNT',
                         color='r', linestyle='dashed')
            if dict_of_prev_stats[model].best_reach_probs_rl is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_reach_probs_rl, color='g', linestyle='dashed')
                plt.plot([], label='Best of Previous RL',
                         color='g', linestyle='dashed')
            if dict_of_prev_stats[model].best_reach_probs_unstable is not None:
                plt.axhline(
                    y=dict_of_prev_stats[model].best_reach_probs_unstable, color='pink', linestyle='dotted')
                plt.plot([], label='Best of Somewhere',
                         color='pink', linestyle='dotted')

    # add legend to axhline plots


def pre_compute_keys(jsons, is_multiple_experiments):
    pre_computed_keys = set()
    if is_multiple_experiments:
        experiment_names = jsons.keys()
        for experiment_name in experiment_names:
            for key in jsons[experiment_name]:
                pre_computed_keys.add(key)
    else:
        for key in jsons:
            pre_computed_keys.add(key)

    return pre_computed_keys


def plot_single_metric_for_model(jsons, metric, model, save_folder, is_multiple_experiments=False):
    plt.figure(figsize=(10, 8))
    pre_computed_keys = pre_compute_keys(jsons, is_multiple_experiments)
    try:
        if is_multiple_experiments:
            plot_multiple_experiments(jsons, metric, model, pre_computed_keys)
        else:
            plot_single_experiment(jsons, metric, model, pre_computed_keys)
    except Exception as e:
        print(f"Error in {model} with {metric}: {e}")

    plt.title(f"Graph for {model} with {metric}")
    plt.xlabel("i-th 50 iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{save_folder}/{model}_{metric}.png")
    plt.close()


def plot_multiple_experiments(jsons, metric, model, pre_computed_keys):
    first = True
    color_index = 0
    color_plots = ['b', 'y', 'm', 'c', 'g', 'k']
    for key in pre_computed_keys:
        model_name, algorithm_name = get_experiment_setting_from_name(key)
        for experiment_name in jsons:
            # plot_color = color_plots[color_index % len(color_plots)]
            plot_color = None
            color_index += 1
            if model_name == model and not metric == "reach_probs":
                try:
                    data = jsons[experiment_name][key][metric]
                    plot_single_curve(
                        data, algorithm_name + " " + experiment_name, plot_color=plot_color)
                except Exception as e:
                    print(
                        f"Error in {model} with experiment {experiment_name} and metric {metric}: {e}")
            elif model_name == model and metric == "reach_probs":
                try:
                    data = jsons[experiment_name][key][metric]
                    plot_single_curve(
                        data, "Goal Reachability" + " " + experiment_name, plot_color=plot_color)
                except Exception as e:
                    print(f"Error in {model} with {metric}: {e}")
    add_line_to_plot(model, metric)


def plot_single_experiment(jsons, metric, model, pre_computed_keys):
    for key in pre_computed_keys:
        model_name, algorithm_name = get_experiment_setting_from_name(key)
        if model_name == model and not metric == "reach_probs":
            data = jsons[key][metric]
            plot_single_curve(data, algorithm_name)
            add_line_to_plot(model, metric)
        elif model_name == model and metric == "reach_probs":
            data = jsons[key][metric]
            try:
                trap_data = jsons[key]["trap_reach_probs"]
                # plot_single_curve(trap_data, f"Trap Reachability", is_trap=True)
            except:
                pass
            plot_single_curve(data, "Goal Reachability")
            add_line_to_plot(model, metric)


def run_plots(results_folder, save_folder):
    is_multiple_experiments = False
    if isinstance(results_folder, dict):
        is_multiple_experiments = True
        jsons = {}
        for experiment_name in results_folder:
            jsons[experiment_name] = load_jsons_from_folder(
                results_folder[experiment_name])
        models = set()
        for experiment_name in jsons:  # Get all models
            for key in jsons[experiment_name]:
                print(key)
                model_name, _ = get_experiment_setting_from_name(key)
                models.add(model_name)
    else:
        jsons = load_jsons_from_folder(results_folder)
        models = set()
        for key in jsons:
            model_name, _ = get_experiment_setting_from_name(key)
            models.add(model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for model in models:  # Plot all metrics for each model
        for metric in METRICS:
            plot_single_metric_for_model(
                jsons, metric, model, save_folder, is_multiple_experiments)
    summary_table_return, summary_table_reachability = get_summary_table(
        jsons, models)
    df = pd.DataFrame(summary_table_return).T
    df.to_excel(f"{save_folder}/summary_table_return.xlsx")
    df = pd.DataFrame(summary_table_reachability).T
    df.to_excel(f"{save_folder}/summary_table_reachability.xlsx")


def get_summary_table(jsons, models):
    summary_table = {}
    for metric in [("Best Return", "best_return"), ("Best Reachability", "best_reach_prob")]:
        summary_table[metric[0]] = {}
        for model in models:
            summary_table[metric[0]][model] = {}
            summary_table[metric[0]
                          ][model]["spaynt"] = dict_of_prev_stats[model].best_return_spaynt
            summary_table[metric[0]
                          ][model]["rl"] = dict_of_prev_stats[model].best_return_rl
            summary_table[metric[0]
                          ][model]["unstable"] = dict_of_prev_stats[model].best_return_unstable
            for key in jsons:
                for sub_key in jsons[key]:
                    model_name, _ = get_experiment_setting_from_name(sub_key)
                    if model_name == model:
                        summary_table[metric[0]
                                      ][model][key] = jsons[key][sub_key][metric[1]]

    return summary_table["Best Return"], summary_table["Best Reachability"]


if __name__ == "__main__":
    if False:
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--folder", type=str,
                            default="experiments_original_boosted")
        parser.add_argument("--save_folder", type=str,
                            default="./plots_original")

        args = parser.parse_args()
        run_plots(args.folder, args.save_folder)
    else:
        # dict_of_folders = {
        #     "Masked Randomized RNN": "experiments_tuning_rnn_random/experiments_0.001_256/",
        #     "Masked RNN": "experiments_tuning_rnn/experiments_0.001_256/",
        #     "Randomized FFNN": "experiments_various_settings/experiments_tuning_f_random/experiments_0.0005_512/",
        #     "Unmasked RNN": "experiments_tuning_rnn_demasked/experiments_0.001_256/",
        #     "Unmasked Randomized RNN": "experiments_tuning_rnn_random_demasked/experiments_0.001_256/",
        # }
        dict_of_folders = {
            # "Randomized FFNN": "experiments_various_settings/experiments_tuning_f_random/experiments_0.0005_512/",
            "Randomized RNN Batch 256 LR 0.001": "experiments_tuning_rnn_random_demasked/experiments_0.001_256/",
            "RNN Batch 256 LR 0.001": "experiments_tuning_rnn_demasked/experiments_0.001_256/",
            "Regularized Randomized Batch 256 LR 0.0001": "experiments_tuning_rnn_random_demasked_regularized/experiments_0.0001_256/",
            "Regularized Randomized Batch 256 LR 0.001": "experiments_tuning_rnn_random_demasked_regularized/experiments_0.001_256/",
            "Regularized Randomized Batch 512 LR 0.0001": "experiments_tuning_rnn_random_demasked_regularized/experiments_0.0001_512/",
            "Regularized Randomized Batch 256 LR 5e-05": "experiments_tuning_rnn_random_demasked_regularized/experiments_5e-05_256/",
            "Regularized Batch 256 LR 0.0001": "experiments_tuning_rnn_demasked_regularized/experiments_0.0001_256/",
            "Regularized Batch 256 LR 0.001": "experiments_tuning_rnn_demasked_regularized/experiments_0.001_256/",
            "Regularized Batch 512 LR 0.0001": "experiments_tuning_rnn_demasked_regularized/experiments_0.0001_512/",
            "Regularized Batch 256 LR 5e-05": "experiments_tuning_rnn_demasked_regularized/experiments_5e-05_256/",
        }
        save_folder = "./plots_comparison_ablation"

        run_plots(dict_of_folders, save_folder)

# run_plots("experiments_original_boosted", "./plots_original")
