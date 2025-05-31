# Used for running various experiments with training. Used primarily for multi-agent training.
# Author: David Hudák
# Login: xhudak03
# File: interface.py

import sys
sys.path.append("../")

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
import copy
import argparse
from rl_src.tools.saving_tools import save_dictionaries, save_statistics_to_new_json
from rl_src.experimental_interface import ExperimentInterface
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions
import os
import logging
import time


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
                except Exception as e:
                    logger.error("Storing dictionaries failed!")
                    logger.error(e)
        else:
            try:
                obs_action_dict = dicts[0]
                memory_dict = dicts[1]
                labels = dicts[2]

                save_dictionaries(name_of_experiment, model, learning_method,
                                  refusing, obs_action_dict, memory_dict, labels)
            except Exception as e:
                logger.error("Storing dictionaries failed!")
                logger.error(e)
                logger.error("Saving stats failed!")


MODEL_TO_OBSERMODEL = {
    "geo-2-8": "models_large/obsergeo-2-8",
    "drone-2-8-1": "models_large/obserdrone-2-8-1",
    "refuel-n": "models/obserfuel-n"
}


def get_fully_observable_environment(args: ArgsEmulator, model="refuel-n") -> EnvironmentWrapperVec:
    """ Get the fully observable environment for the given model.
    Args:
        args (ArgsEmulator): Arguments for the initialization.
        model (str, optional): The name of the model. Defaults to "refuel-n".

    Returns:
        EnvironmentWrapperVec: The environment.
    """
    modified_args = copy.deepcopy(args)
    modified_args.prism_model = f"{MODEL_TO_OBSERMODEL[model]}/sketch.templ"
    modified_args.prism_properties = f"{MODEL_TO_OBSERMODEL[model]}/sketch.props"
    if "-n" in model:
        modified_args.constants = "N=20"
    env, _, _ = ExperimentInterface.initialize_environment(modified_args)
    return env


def run_single_experiment(args: ArgsEmulator, model="network-3-8-20", learning_method="PPO", refusing=None,
                          name_of_experiment="results_of_interpretation", save_statistics: bool = True):
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

    interface = ExperimentInterface(args)
    dicts = interface.perform_experiment(with_refusing=refusing, model=model)
    if args.perform_interpretation:
        if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}"):
            os.makedirs(f"{name_of_experiment}/{model}_{learning_method}")
        save_dictionaries_caller(
            dicts, name_of_experiment, model, learning_method, refusing)
    end_time = time.time()
    evaluation_time = end_time - start_time
    if save_statistics:
        save_statistics_to_new_json(name_of_experiment,
                                    model,
                                    learning_method,
                                    interface.agent.evaluation_result,
                                    evaluation_time=evaluation_time,
                                    args=args)

    # Old way of saving statistics
    # save_statistics(name_of_experiment, model, learning_method, initializer.agent.evaluation_result, args.evaluation_goal)


def run_experiments(name_of_experiment="results_of_interpretation", path_to_models="./models_large", learning_rate=0.0001, batch_size=256,
                    random_start_simulator=False, model_condition: str = "", model_memory_size=0, use_rnn_less=False, state_estimation=False,
                    train_state_estimator_continuously=False,
                    curiosity_automata_reward=False, predicate_automata_obs = False, go_explore = False):
    """ Run multiple experiments for PAYNT oracle.
    Args:
        name_of_experiment (str, optional): The name of the experiment. Defaults to "results_of_interpretation".
        path_to_models (str, optional): The path to the models. Defaults to "./models_large".
        learning_rate (float, optional): The learning rate. Defaults to 0.0001.
        batch_size (int, optional): The batch size. Defaults to 256.
        random_start_simulator (bool, optional): Whether to start the simulator randomly. Defaults to False.
        model_condition (str, optional): The condition of the model (rule condition is in the name of the model, e.g. "network"). Defaults to "".
    """
    for model in os.listdir(f"{path_to_models}"):
        # if "drone" in model:  # Currently not supported model
        #     continue
        if model_condition not in model:
            continue
        # if "network" not in model:
        #     continue
        prism_model = f"{path_to_models}/{model}/sketch.templ"
        prism_properties = f"{path_to_models}/{model}/sketch.props"
        encoding_method = "Valuations"
        refusing = None

        for learning_method in ["PPO"]:
            # if not "network" in model:
            #     continue
            for replay_buffer_option in [ReplayBufferOptions.ON_POLICY]:
                logger.info(
                    f"Running iteration {1} on {model} with {learning_method}, refusing set to: {refusing}, encoding method: {encoding_method}.")
                args = ArgsEmulator(prism_model=prism_model, prism_properties=prism_properties, learning_rate=learning_rate,
                                    restart_weights=0, learning_method=learning_method, evaluation_episodes=30,
                                    nr_runs=2001, encoding_method=encoding_method, agent_name=model, load_agent=False,
                                    evaluate_random_policy=False, max_steps=1001, evaluation_goal=60, evaluation_antigoal=-0.0,
                                    trajectory_num_steps=32, discount_factor=0.99, num_environments=batch_size,
                                    normalize_simulator_rewards=False, buffer_size=1005, random_start_simulator=random_start_simulator,
                                    replay_buffer_option=replay_buffer_option, batch_size=batch_size,
                                    vectorized_envs_flag=True, flag_illegal_action_penalty=False, perform_interpretation=False,
                                    use_rnn_less=use_rnn_less, model_memory_size=model_memory_size if model_memory_size > 0 else 0,
                                    name_of_experiment=name_of_experiment, continuous_enlargement=False, continuous_enlargement_step=3,
                                    constants="", state_supporting=(state_estimation), train_state_estimator_continuously=train_state_estimator_continuously,
                                    curiosity_automata_reward=curiosity_automata_reward, predicate_automata_obs=predicate_automata_obs, 
                                    go_explore=go_explore, stacked_observations=False,
                                    env_see_reward=False, env_see_last_action=False, env_see_num_steps=False)

                run_single_experiment(
                    args, model=model, learning_method=learning_method, refusing=False, name_of_experiment=name_of_experiment)


if __name__ == "__main__":
    args_from_cmd = argparse.ArgumentParser()

    args_from_cmd.add_argument("--batch-size", type=int, default=1024)
    args_from_cmd.add_argument("--learning-rate", type=float, default=1.6e-4)
    args_from_cmd.add_argument(
        "--path-to-models", type=str, default="./models")
    args_from_cmd.add_argument("--random-start-simulator", action="store_true")
    args_from_cmd.add_argument("--model-condition", type=str, default="")
    args_from_cmd.add_argument("--use-rnn-less", action="store_true", default=False,
                               help="Removes LSTM layers from PPO Actor and Critic.")
    args_from_cmd.add_argument("--model-memory-size", type=int, default=0,
                               help="The size of the memory in the model. If 0, the memory is not used.")
    args_from_cmd.add_argument("--state-estimation", action="store_true",
                               help="Turn-off external state estimation.")
    args_from_cmd.add_argument("--train-state-estimator-continuously", action="store_true",
                               help="Train state estimator continuously.")

    args_from_cmd.add_argument("--curiosity-automata-reward", action="store_true",
                               help="Use curiosity automata reward.")
    args_from_cmd.add_argument("--predicate-automata-obs", action="store_true",
                                 help="Use predicate automata observation.")
    args_from_cmd.add_argument("--go-explore", action="store_true",
                                    help="Use Go-Explore.")
    

    args = args_from_cmd.parse_args()

    # Run experiments with the given arguments
    if args.random_start_simulator:
        name = "experiments_32_random"
    else:
        name = "experiments_drone_only"

    if args.use_rnn_less:
        name += "_rnn_less"
    if args.model_memory_size > 0:
        name += f"_memory_{args.model_memory_size}"
    if args.state_estimation:
        name += "_state_est"
    if args.train_state_estimator_continuously:
        name += "_train_state_estimator_continuously"
    if args.curiosity_automata_reward:
        name += "_curiosity_automata_reward"
    if args.predicate_automata_obs:
        name += "_predicate_automata_obs"
    if args.go_explore:
        name += "_go_explore"
    if not os.path.exists(name):
        os.makedirs(name)
    

    run_experiments(f"{name}/experiments_{args.learning_rate}_{args.batch_size}", args.path_to_models, learning_rate=args.learning_rate,
                    batch_size=args.batch_size, random_start_simulator=args.random_start_simulator, model_condition=args.model_condition,
                    model_memory_size=args.model_memory_size, use_rnn_less=args.use_rnn_less, state_estimation=args.state_estimation,
                    train_state_estimator_continuously=args.train_state_estimator_continuously,
                    curiosity_automata_reward=args.curiosity_automata_reward, predicate_automata_obs=args.predicate_automata_obs,
                    go_explore=args.go_explore)
    # for _ in range(10):
    #     # 0.00001
    #     for learning_rate in [0.00005, 0.0001, 0.0005, 0.001]:
    #         for batch_size in [32, 64, 128, 256, 512, 1024]:
    #             logger.info(f"Running experiments with learning rate: {learning_rate} and batch size: {batch_size}")
    #             run_experiments(f"experiments_tuning/experiments_{learning_rate}_{batch_size}", "./models_large", learning_rate=learning_rate, batch_size=batch_size)
    # run_experiments("experiments_action_masking", "./models", learning_rate=0.0001, batch_size=128)
