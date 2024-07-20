# Description: This file contains functions to plot the results of the experiments. The functions load the data from the files and plot the results.
# Finds the files in current directory for each learning algorithm and plots the results.
# Author: David Hudák
# Login: xhudak03
# File: plots.py
 

from matplotlib import pyplot as plt

import ast

import numpy as np
import os


def plot_final_reward(rewards_without_final, rewards_final, title="Final probability",
                      learning_algorithms=["PPO", "DQN", "DDQN"], y_line=5, save_file=None):
    rewards = rewards_final
    plt.plot(rewards / 40.0)
    plt.axvline(x=y_line, color='r', linestyle='--',
                label="Paynt FSC data only")
    plt.title(title)
    plt.xlabel("i-th hundred iteration")
    plt.ylabel("Probability of reaching the goal - reaching the traps")


def show_or_save(save_file):
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()


def plot_final_rewards(rewards_final, title="Final probability", learning_algorithms=["PPO", "DQN", "DDQN"],
                       current_optimum=None, max_reward=50.0, save_file=None):
    plt.figure(figsize=(6, 4))
    for i, key in enumerate(rewards_final):
        plt.plot(rewards_final[key] / max_reward, label=key)
    if current_optimum is not None:
        plt.axhline(y=current_optimum, color='k', linestyle='--',
                    label="random policy")
    plt.title(title)
    plt.xlabel("i-th hundred iteration")
    plt.ylabel("Probability of reaching the goal - reaching the traps")
    plt.legend()
    show_or_save(save_file)


def plot_cumulative_rewards(rewards_without_final, learning_algorithms=["PPO", "DQN", "DDQN"],
                            current_optimum=None, title="Cumulative reward without goal (higher = better)",
                            save_file=None):
    plt.figure(figsize=(6, 4))
    for i, key in enumerate(rewards_without_final):
        plt.plot(rewards_without_final[key], label=learning_algorithms[i])
    if current_optimum is not None:
        plt.axhline(y=current_optimum, color='k', linestyle='--',
                    label="random policy")
    plt.title(title)
    plt.xlabel("i-th hundred iteration")
    plt.ylabel("Cumulative reward")
    plt.legend()
    show_or_save(save_file)


def windowed_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def plot_losses(losses, learning_algorithms=["PPO", "DQN", "DDQN"], title="Losses (lower = better)", smoothing=True, window_size=5, save_file=None):
    plt.figure(figsize=(6, 4))

    for i, key in enumerate(losses):
        if smoothing:
            smooth_losses = windowed_average(
                losses[key], window_size=window_size)
            plt.plot(smooth_losses, label=learning_algorithms[i])
        else:
            plt.plot(losses[key], label=learning_algorithms[i])
    plt.title(title)
    plt.xlabel("i-th hundred iteration")
    plt.ylabel("Loss")
    plt.legend()
    show_or_save(save_file)


def load_files(name, expected_learning_algorithms=["ppo", "dqn", "ddqn", "stochastic_ppo"]):
    rewards_final = {}
    rewards_without_final = {}
    labels = {}
    learning_algorithms = []
    for learning_algorithm in expected_learning_algorithms:
        try:
            with open(f"./{learning_algorithm}/average_return_with_final.txt", "r") as file:
                rewards_final[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            with open(f"./{learning_algorithm}/average_return_without_final.txt", "r") as file:
                rewards_without_final[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            with open(f"./{learning_algorithm}/losses.txt", "r") as file:
                labels[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            learning_algorithms.append(learning_algorithm)
        except FileNotFoundError:
            print(f"Files not found for method {learning_algorithm}")
    return rewards_final, rewards_without_final, labels, learning_algorithms


def load_files_from_experiments(model_name, expected_learning_algorithms=["ppo", "dqn", "ddqn", "stochastic_ppo"]):
    rewards_final = {}
    rewards_without_final = {}
    labels = {}
    learning_algorithms = []

    for learning_algorithm in expected_learning_algorithms:
        uppercase_learning_algorithm = learning_algorithm.upper()
        try:
            with open(f"./{model_name}_{uppercase_learning_algorithm}/average_return_with_final.txt", "r") as file:
                rewards_final[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            with open(f"./{model_name}_{uppercase_learning_algorithm}/average_return_without_final.txt", "r") as file:
                rewards_without_final[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            with open(f"./{model_name}_{uppercase_learning_algorithm}/losses.txt", "r") as file:
                labels[learning_algorithm] = np.array(
                    ast.literal_eval(file.read()))
            learning_algorithms.append(learning_algorithm)
        except FileNotFoundError:
            print(f"Files not found for method {learning_algorithm}")
    return rewards_final, rewards_without_final, labels, learning_algorithms


def plots_new(name, max_reward, load_files=load_files):
    rewards_final, rewards_without_final, labels, learning_algorithms = load_files(
        name)
    if len(rewards_final) == 0:
        print(f"No data found for {name}")
        return

    plot_final_rewards(rewards_final, learning_algorithms=learning_algorithms,
                       title=name + " (higher = better)", max_reward=max_reward, save_file="./imgs/" + name + "_final_rewards.pdf", current_optimum=0.17893)
    plot_cumulative_rewards(rewards_without_final,
                            learning_algorithms=learning_algorithms, title=name +
                            " cumulative reward without goal (higher = better)",
                            save_file="./imgs/" + name + "_cumulative_rewards.pdf", current_optimum=-74.5)
    plot_losses(labels, learning_algorithms=learning_algorithms,
                title=name + " loss function (lower = better)", smoothing=False, save_file="./imgs/" + name + "_losses.pdf")
    plot_losses(labels, learning_algorithms=learning_algorithms, title=name +
                " loss function (lower = better) with averaging with window = 5",
                smoothing=True, window_size=5, save_file="./imgs/" + name + "_losses_windowed_5.pdf")
    plot_losses(labels, learning_algorithms=learning_algorithms, title=name +
                " loss function (lower = better) with averaging with window = 30",
                smoothing=True, window_size=30, save_file="./imgs/" + name + "_losses_windowed_30.pdf")
    


if __name__ == "__main__":
    reward_dictionary = {
        "evade": 300.0,
        "evade_n=5_r=23": 300.0,
        "rocks-16": 150.0,
        "network-3-8-20": 50.0,
        "mba-small": 600.0,
        "obstacle": 50.0,
        "mba": 100.0,
        "refuel-20": 50.0,
        "grid-large-30-5": 300.0,
        "intercept": 50.0,
        "intercept-n7-r1" : 150.0
    }
    # plots_new("Grid Large", 50.0)
    os.makedirs("imgs", exist_ok=True)
    for model in reward_dictionary:
        print(f"Processing {model}")
        plots_new(model, reward_dictionary[model], load_files_from_experiments)
