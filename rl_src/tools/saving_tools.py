
from rl_src.environment.environment_wrapper_vec import os


import os
import pickle

from rl_src.tools.evaluators import EvaluationResults

import logging
logger = logging.getLogger(__name__)


def save_dictionaries(name_of_experiment, model, learning_method, refusing_typ, obs_action_dict, memory_dict, labels):
    """ Save dictionaries for Paynt oracle.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        refusing_typ (str): Whether to use refusing when interpreting.
        obs_action_dict (dict): The observation-action dictionary.
        memory_dict (dict): The memory dictionary.
        labels (dict): The labels dictionary.
    """
    if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}"):
        os.makedirs(
            f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}")
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/obs_action_dict.pickle", "wb") as f:
        pickle.dump(obs_action_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/memory_dict.pickle", "wb") as f:
        pickle.dump(memory_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/labels.pickle", "wb") as f:
        pickle.dump(labels, f)


def save_statistics_to_new_json(name_of_experiment, model, learning_method, evaluation_result: EvaluationResults, args: dict = None, evaluation_time: float = float("nan")):
    """ Save statistics to a new JSON file.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        evaluation_result (EvaluationResults): The evaluation results.
        args (dict, optional): The arguments. Defaults to None.
    """
    if args is None:
        max_steps = 300
    else:
        max_steps = args.max_steps

    evaluation_result.set_experiment_settings(
        learning_algorithm=learning_method, max_steps=max_steps)
    if not os.path.exists(f"{name_of_experiment}"):
        os.mkdir(f"{name_of_experiment}")
    if os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training.json"):
        i = 1
        while os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json"):
            i += 1
        logger.info(
            f"Saving evaluation results to {name_of_experiment}/{model}_{learning_method}_training_{i}.json")
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json", evaluation_time)
    else:
        logger.info(
            f"Saving evaluation results to {name_of_experiment}/{model}_{learning_method}_training.json")
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training.json", evaluation_time)
