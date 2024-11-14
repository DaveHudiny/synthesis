# Used for running various experiments with training. Used primarily for multi-agent training.
# Author: David Hudák
# Login: xhudak03
# File: interface.py
import sys
sys.path.append("../")
import time
import logging
import os
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions
from rl_src.experimental_interface import ExperimentInterface


from rl_src.tools.saving_tools import save_dictionaries, save_statistics_to_new_json



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
    initializer = ExperimentInterface(args)
    dictionaries = initializer.perform_experiment(with_refusing=with_refusing)
    return dictionaries


def save_dictionaries_caller(dicts, name_of_experiment, model, learning_method, refusing):
    for quality in ["last", "best"]:
        if refusing is None:
            for typ in ["with_refusing", "without_refusing"]:
                quality_typ = quality + "_" + typ
                try:
                    obs_action_dict = dicts[quality_typ][0]
                    memory_dict = dicts[quality_typ][1]
                    labels = dicts[quality_typ][2]
                    save_dictionaries(name_of_experiment, model, learning_method,
                                      quality_typ, obs_action_dict, memory_dict, labels)
                except:
                    logger.error("Storing dictionaries failed!")
        else:
            try:
                obs_action_dict = dicts[0]
                memory_dict = dicts[1]
                labels = dicts[2]

                save_dictionaries(name_of_experiment, model, learning_method,
                                  refusing, obs_action_dict, memory_dict, labels)
            except:
                logger.error("Saving stats failed!")


def run_single_experiment(args: ArgsEmulator, model="network-3-8-20", learning_method="PPO", refusing=None,
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
    start_time = time.time()
    refusing = None
    initializer = ExperimentInterface(args)
    dicts = initializer.perform_experiment(with_refusing=refusing)
    if args.perform_interpretation:
        if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}"):
            os.makedirs(f"{name_of_experiment}/{model}_{learning_method}")
        save_dictionaries_caller(dicts, name_of_experiment, model, learning_method, refusing)
    end_time = time.time()
    evaluation_time = end_time - start_time
    save_statistics_to_new_json(name_of_experiment, model, learning_method,
                                initializer.agent.evaluation_result, evaluation_time=evaluation_time, args=args)

    # Old way of saving statistics
    # save_statistics(name_of_experiment, model, learning_method, initializer.agent.evaluation_result, args.evaluation_goal)


def run_experiments(name_of_experiment="results_of_interpretation", path_to_models="./models_large", learning_rate=0.0001, batch_size=256):
    """ Run multiple experiments for PAYNT oracle."""
    for model in os.listdir(f"{path_to_models}"):
        if "drone" in model:  # Currently not supported model
            continue
        # if "network" not in model:
        #     continue
        prism_model = f"{path_to_models}/{model}/sketch.templ"
        prism_properties = f"{path_to_models}/{model}/sketch.props"
        encoding_method = "Valuations"
        refusing = None
        for learning_method in ["Stochastic_PPO"]:
            if "drone" in model:  # Currently not supported model
                continue
            # if not "network" in model:
            #     continue
            for replay_buffer_option in [ReplayBufferOptions.ON_POLICY]:
                logger.info(
                    f"Running iteration {1} on {model} with {learning_method}, refusing set to: {refusing}, encoding method: {encoding_method}.")
                args = ArgsEmulator(prism_model=prism_model, prism_properties=prism_properties, learning_rate=learning_rate,
                                    restart_weights=0, learning_method=learning_method, evaluation_episodes=30,
                                    nr_runs=4001, encoding_method=encoding_method, agent_name=model, load_agent=False, evaluate_random_policy=False,
                                    max_steps=400, evaluation_goal=10, evaluation_antigoal=-2, trajectory_num_steps=64, discount_factor=0.99, num_environments=batch_size,
                                    normalize_simulator_rewards=False, buffer_size=50000, random_start_simulator=False, replay_buffer_option=replay_buffer_option, batch_size=batch_size,
                                    vectorized_envs=True)

                run_single_experiment(
                    args, model=model, learning_method=learning_method, refusing=None, name_of_experiment=name_of_experiment)

if __name__ == "__main__":
    # for _ in range(10):
    #     for learning_rate in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
    #         for batch_size in [32, 64, 128, 256, 512, 1024]:
    #             logger.info(f"Running experiments with learning rate: {learning_rate} and batch size: {batch_size}")
    #             run_experiments(f"experiments_tuning/experiments_{learning_rate}_{batch_size}", "./models", learning_rate=learning_rate, batch_size=256)
    run_experiments("experiments_single_original_env", "./models_large", learning_rate=0.0001, batch_size=64)
