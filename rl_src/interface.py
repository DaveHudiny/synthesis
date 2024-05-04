from rl_initializer import Initializer, ArgsEmulator, save_dictionaries, save_statistics

import pickle
import os

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def get_dictionaries(args, with_refusing=False):
    """ Get dictionaries for Paynt oracle.
    Args:
        args (ArgsEmulator): Arguments for the initialization.
        with_refusing (bool, optional): Whether to use refusing when interpreting. Defaults to False.

    Returns:
        tuple: Tuple of dictionaries (obs_act_dict, memory_dict, labels).
    """
    initializer = Initializer(args)
    dictionaries = initializer.main(with_refusing=with_refusing)
    return dictionaries


def run_single_experiment(args, model="network-3-8-20", learning_method="PPO", refusing=None, 
                          name_of_experiment="results_of_interpretation", encoding_method="Valuations"):
    """ Run a single experiment for Paynt oracle.
    Args:
        args (ArgsEmulator): Arguments for the initialization.
        model (str, optional): The name of the model. Defaults to "network-3-8-20".
        learning_method (str, optional): The learning method. Defaults to "PPO".
        refusing (bool, optional): Whether to use refusing when interpreting. Defaults to False.
        name_of_experiment (str, optional): The name of the experiment. Defaults to "results_of_interpretation".
        encoding_method (str, optional): The encoding method. Defaults to "Valuations".
    """
    refusing = None
    initializer = Initializer(args)
    dicts = initializer.main(with_refusing=refusing)

    if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}"):
        os.makedirs(f"{name_of_experiment}/{model}_{learning_method}")
    for quality in ["last", "best"]:
        if refusing is None:
            for typ in ["with_refusing", "without_refusing"]:
                quality_typ = quality + "_" + typ
                obs_action_dict = dicts[quality_typ][0]
                memory_dict = dicts[quality_typ][1]
                labels = dicts[quality_typ][2]
                save_dictionaries(name_of_experiment, model, learning_method,
                                  quality_typ, obs_action_dict, memory_dict, labels)
        else:
            obs_action_dict = dicts[0]
            memory_dict = dicts[1]
            labels = dicts[2]
            save_dictionaries(name_of_experiment, model, learning_method,
                              refusing, obs_action_dict, memory_dict, labels)
            
    save_statistics(name_of_experiment, model, learning_method, initializer.agent.stats_without_ending, 
                    initializer.agent.stats_with_ending, initializer.agent.losses, args.evaluation_goal)


def run_experiments(name_of_experiment="results_of_interpretation", path_to_models="./models"):
    for model in os.listdir(f"{path_to_models}"):
        prism_model = f"{path_to_models}/{model}/sketch.templ"
        prism_properties = f"{path_to_models}/{model}/sketch.props"
        refusing = None
        if "rocks" not in model:
            continue
        for learning_method in ["PPO", "Stochastic_PPO"]:
            for encoding_method in ["Valuations"]:
                logger.info(f"Running iteration {1} on {model} with {learning_method}, refusing set to: {refusing}, encoding method: {encoding_method}.")
                args = ArgsEmulator(prism_model=prism_model, prism_properties=prism_properties,
                                    restart_weights=0, learning_method=learning_method, using_logits=False, action_filtering=False, reward_shaping=False,
                                    nr_runs=4000, encoding_method=encoding_method, agent_name=model, load_agent=False, evaluate_random_policy=False,
                                    max_steps=150, evaluation_goal=150, evaluation_antigoal=-150, trajectory_num_steps=25)

                run_single_experiment(
                    args, model=model, learning_method=learning_method, refusing=None, name_of_experiment=name_of_experiment + f"_{encoding_method}")


if __name__ == "__main__":
    run_experiments("rocks-16-repaired", "models")
