# Author: David Hudák
# Description: This file contains plot functions for new json evaluation results of RL experiments.

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

METRICS = ["returns", "returns_episodic", "reach_probs", "losses"]

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

def plot_single_curve(data, learning_algorithm):
    data = ast.literal_eval(data)
    numpy_data = np.array(data).astype(np.float32)
    print(learning_algorithm)
    plt.plot(numpy_data, label=learning_algorithm)
    
def plot_single_metric_for_model(jsons, metric, model, save_folder):
    plt.figure(figsize=(6, 4))
    try:
        for key in jsons:
            model_name, algorithm_name = get_experiment_setting_from_name(key)
            if model_name == model:
                data = jsons[key][metric]
                plot_single_curve(data, algorithm_name)
    except Exception as e:
        print(f"Error in {model} with {metric}: {e}")

    plt.title(f"Graph for {model} with {metric}")
    plt.xlabel("i-th hundred iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{save_folder}/{model}_{metric}.pdf")
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
            
run_plots("experiment_json_adhoc", "./")
