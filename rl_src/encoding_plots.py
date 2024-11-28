# Description: Plotting the convergence curves of the encoding methods. Place to wanted folder and it finds the files and plots the convergence curves.
# Author: David Hudák
# Login: xhudak03
# File: encoding_plots.py

import os

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd


def recursive_find_files(folder_path, encoding_method, filename):
    """Find files in the folder recursively.
    Args:
        folder_path (str): Path to the folder.
        encoding_method (str): Encoding method.
        filename (str): Filename.
    Returns:
        list: List of paths to the files.
    """
    files = []
    for root, _, filenames in os.walk(folder_path):
        for file in filenames:
            if encoding_method in root and filename in file:
                files.append(os.path.join(root, file))
    return files


def read_files(files):
    """Read files.
    Args:
        files (list): List of paths to the files.
    """
    returns = []
    for file in files:
        with open(file, "r") as f:
            data = f.read()
            values = eval(data)
            returns.append(np.array(values))
    return returns


def plot_convergence_curves(returns, denominator: float = 100, filename: str = "convergence_curves.pdf",
                            y_name: str = "Probability of reaching goal minus probability of trap"):
    """Plot convergence curves.
    Args:
        returns (list): List of returns.
        denominator (float): Denominator for data.
        filename (str): Save filename.
        y_name (str): Name of the y-axis.
    """
    for key in returns:
        data = returns[key]
        medians = np.median(data, axis=0) / denominator
        q1 = np.percentile(data, 25, axis=0) / denominator
        q3 = np.percentile(data, 75, axis=0) / denominator

        plt.plot(range(1, len(medians) + 1), medians, label=f"Median of {key}")
        plt.fill_between(range(1, len(medians) + 1), q1, q3,
                         alpha=0.3, label=f"Interquartile Range of {key}")

    plt.xlabel("i-th hundred iteration")
    plt.ylabel(y_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def plot_convergence_curves_seaborn(returns, denominator: float = 100, filename: str = "convergence_curves_seaborn.pdf",
                                    y_name: str = "Probability of reaching goal minus probability of trap", random_policy_value=None):
    """Plot convergence curves using seaborn.
    Args:
        returns (list): List of returns.
        denominator (float): Denominator for data.
        filename (str): Save filename.
        y_name (str): Name of the y-axis.
    """

    data = []
    for key in returns:
        for i, values in enumerate(returns[key]):
            for j, value in enumerate(values):
                data.append({"i-th hundred iteration": j + 1, y_name: value / denominator,
                             "Encoding Method": key, "Experiment": i})

    data = pd.DataFrame(data)
    sns.lineplot(x="i-th hundred iteration", y=y_name,
                 hue="Encoding Method", data=data)
    if random_policy_value is not None:
        plt.axhline(y=random_policy_value, color='k',
                    linestyle='--', label="Random policy")
    plt.grid(False)
    plt.savefig(filename)
    plt.close()


def plot_function(folder_path, filename, save_filename, y_name, denominator=1, random_policy_value=None):
    """Plot function.
    Args:
        folder_path (str): Path to the folder.
        filename (str): Filename.
    """
    returns = {}
    for encoding_method in ["Valuations", "Integer", "One-Hot"]:
        files = recursive_find_files(folder_path, encoding_method, filename)
        returns[encoding_method] = read_files(files)
    plot_convergence_curves_seaborn(returns, denominator=denominator, filename=save_filename, y_name=y_name,
                                    random_policy_value=random_policy_value)


if __name__ == "__main__":
    plot_function("./", "average_return_with_final.txt", "convergence_curves_seaborn_final.pdf",
                  y_name="Probability of reaching goal minus probability of trap", denominator=100,
                  random_policy_value=0.25)
    plot_function("./", "average_return_without_final.txt", "convergence_curves_seaborn_without.pdf",
                  y_name="Average return without virtual goal", denominator=1, random_policy_value=-70.0)
