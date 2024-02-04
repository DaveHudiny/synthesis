# File: rl_interface.py
# Author: David Hudák
# Purpose: Interface between paynt and safe reinforcement learning algorithm

from cmd_commands import *
import logging
import os
from collections import Counter
from tqdm import tqdm
import argparse
import stormpy
import tf_agents as tfa
import tensorflow as tf
import rl_simulator_v2 as rl_simulator
import shield_v2
import dill
import pickle
import sys

import numpy as np
sys.path.append('./safe_rl')


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RLInterface:
    def __init__(self):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self._model = None
        self.unique_observations = None
        self._model = None
        self.storm_model = None
        self.tf_environment = None

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Safe reinforcement learning interface")
        parser.add_argument("--grid-model", "-m", type=str,
                            default="evade", help="Model of environment")
        parser.add_argument("--learning_method", type=str,
                            default="SAC", help="Learning method")
        parser.add_argument("--eval-episodes", type=int,
                            default=10, help="Number of episodes to evaluate")
        return parser.parse_args()

    def save_model(self, path_to_model):
        with open(path_to_model, "wb") as file:
            logger.debug(f"Saving model to {path_to_model}")
            dill.dump(self._model, file)
            logger.debug(f"Saved model to {path_to_model}")

    def load_model(self, path_to_model):
        with open(path_to_model, "rb") as file:
            logger.debug(f"Loading model from {path_to_model}")
            self._model = dill.load(file)
            logger.debug(f"Loaded model from {path_to_model}")

    def ask_model(self, storm_state):
        tensor_state = None
        self._model.agent.policy.action(tensor_state)

    def create_model(self, strategy="SAC", num_episodes=10, model_type="evade"):
        if model_type == "evade":
            CmdCommands.add_to_sys_argv_evade()
        elif model_type == "intercept":
            CmdCommands.add_to_sys_argv_intercept()
        elif model_type == "refuel":
            CmdCommands.add_to_sys_argv_refuel()
        elif model_type == "obstacle":
            CmdCommands.add_to_sys_argv_obstacle()
        elif model_type == "avoid":
            CmdCommands.add_to_sys_argv_avoid()
        elif model_type == "rocks":
            CmdCommands.add_to_sys_argv_rocks()
        else:
            logger.error(f"Model type {model_type} not supported.")
            return
        CmdCommands.add_to_sys_argv_strategy(strategy)
        CmdCommands.add_to_sys_argv_num_episodes(10)
        self._model, self.storm_model, self.tf_environment = shield_v2.improved_main()

    def create_timestep(self, mask_size):
        discount = tf.constant([1.0], dtype=tf.float32)
        observation = {
            'mask': tf.constant([[False for i in range(mask_size)]], dtype=tf.bool),
            'obs': tf.constant([[0]], dtype=tf.int32)
        }
        reward = tf.constant([-1.0], dtype=tf.float32)
        step_type = tf.constant([1], dtype=tf.int32)

        time_step = tfa.trajectories.time_step.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation
        )
        return time_step

    def _policy_step_limits_from_set(self, policy_steps=None):
        if policy_steps is None or not policy_steps:
            raise ValueError("policy_steps should be a non-empty list")
        states = [policy_step.state for policy_step in policy_steps]
        min_limit = tf.reduce_min(states, axis=0)
        max_limit = tf.reduce_max(states, axis=0)
        return min_limit, max_limit

    def update_policy_step_limits(self, policy_steps, limits_min, limits_max):
        states = [policy_step.state for policy_step in policy_steps]
        min_limit = tf.reduce_min(states, axis=0)
        max_limit = tf.reduce_max(states, axis=0)
        limits_min = tf.minimum(limits_min, min_limit)
        limits_max = tf.maximum(limits_max, max_limit)
        return limits_min, limits_max

    def initialize_limits(self, time_step):
        action = self._model.agent.policy.action(time_step)
        limits_min, limits_max = self._policy_step_limits_from_set([action])
        return limits_min, limits_max

    def generate_policy_step_states_limits2(self):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        actions = []
        print("Generating policy step states limits.")
        generate_policy_step_states_limits_tf = tf.function(
            self._generate_policy_step_states_limits)
        for observation in tqdm(self.unique_observations[:10]):
            time_step.observation['obs'] = tf.constant(
                [[observation]], dtype=tf.int32)
            action = generate_policy_step_states_limits_tf(
                time_step, self._model.agent.policy, nr_runs=3)
            actions.extend(action)
        limits_min, limits_max = self._policy_step_limits_from_set(actions)
        print(limits_min, limits_max)
        return limits_min, limits_max

    def _generate_policy_step_states_limits2(self, time_step, policy, nr_runs=3):
        actions = [policy.action(time_step)]
        for _ in range(nr_runs):
            action = policy.action(time_step, actions[-1].state)
            actions.append(action)
        return actions

    def generate_policy_step_states_limits(self):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        print("Generating policy step states limits.")
        generate_policy_step_states_limits_tf = tf.function(
            self._generate_policy_step_states_limits)
        time_step.observation['obs'] = tf.constant([[0]], dtype=tf.int32)
        limits_min, limits_max = self.initialize_limits(time_step)
        # logger.debug(f"Initial limits: {limits_min}, {limits_max}")
        for observation in tqdm(self.unique_observations):
            time_step.observation['obs'] = tf.constant(
                [[observation]], dtype=tf.int32)
            action = generate_policy_step_states_limits_tf(
                time_step, self._model.agent.policy, nr_runs=3)
            limits_min, limits_max = self.update_policy_step_limits(
                action, limits_min, limits_max)
        # logger.debug(f"Final limits: {limits_min}, {limits_max}")
        return limits_min, limits_max

    def _generate_policy_step_states_limits(self, time_step, policy, nr_runs=3):
        actions = [policy.action(time_step)]
        for _ in range(nr_runs):
            action = policy.action(time_step, actions[-1].state)
            actions.append(action)
        return actions

    def monte_carlo_evaluation(self, time_step, nr_episodes, policy_limits_min=[0], policy_limits_max=[1], policy=None):
        unique_actions = np.empty(shape=(0,), dtype=np.int32)
        action = self._model.agent.policy.action(time_step)
        action_stats_dict = Counter()
        actions = []
        for _ in range(nr_episodes):
            new_state = [tf.random.uniform(shape=policy_limits_min[i].shape,
                                           minval=policy_limits_min[i],
                                           maxval=policy_limits_max[i],
                                           dtype=tf.float32) for i in range(len(action.state))]
            uniqor = policy(time_step, new_state)
            act = np.array(uniqor.action).item()
            action_stats_dict.update([act])
            actions.append(act)
        unique_actions = np.unique(actions)
        return unique_actions, action_stats_dict

    def make_actions_printable(self, actions):
        action_numbers = ""
        action_keywords = ""
        for action in actions:
            action_numbers += f"{', ' if len(action_numbers) > 0 else ''}{action}"
            action_keywords += f"{', ' if len(action_keywords) > 0 else ''}{self.tf_environment.act_keywords[action]}"
        return action_numbers, action_keywords

    def print_triplet(self, observation, actions, memory, file=sys.stdout):
        acts, act_keys = self.make_actions_printable(actions)
        print(f"Observation: {observation}. Actions: {acts}. " +
              f"Action Keywords: {act_keys}")
        print(f"Action statistics: {memory}", file=file)

    def memory_estimation(self, action_stats_dict, std_multiplier_threshold=0.5):
        numpy_values = np.array(list(action_stats_dict.values()))
        mean = np.mean(numpy_values)
        std = np.std(numpy_values)
        threshold = mean - std_multiplier_threshold * std
        action_stats_dict_filtered = {
            k: v for k, v in action_stats_dict.items() if v <= threshold}
        return action_stats_dict_filtered, len(action_stats_dict_filtered)

    def evaluate_model(self, file, nr_episodes=10, method="monte_carlo"):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        self.unique_observations = np.unique(self.storm_model.observations)
        limits_min, limits_max = self.generate_policy_step_states_limits()
        policy_function = tf.function(self._model.agent.policy.action)
        observation_dict = {}
        logger.debug("Evaluating model.")
        for observation in tqdm(self.unique_observations):
            time_step.observation['obs'] = tf.constant(
                [[observation]], dtype=tf.int32)
            actions, action_stats_dict = self.monte_carlo_evaluation(
                time_step, nr_episodes, limits_min, limits_max, policy=policy_function)
            _, memory = self.memory_estimation(action_stats_dict)
            # self.print_triplet(observation, actions, memory, file)
            observation_dict[observation] = actions
        logger.debug("Evaluation finished.")
        with open("obs_actions.pickle", "wb") as file_pickle:
            pickle.dump(observation_dict, file_pickle)

    def evaluate_model_to_file(self, path_to_file):
        if self._model is None:
            logger.error("Model not created.")
            return
        with open(path_to_file, "w") as file:
            self.evaluate_model(file)


if __name__ == "__main__":
    interface = RLInterface()
    clean_argv = [sys.argv[0]]
    args = interface.parse_args()
    sys.argv = clean_argv
    interface.create_model(args.learning_method,
                           args.eval_episodes, args.grid_model)
    # actions = interface.tf_environment._simulator.available_actions()
    interface.evaluate_model_to_file("obs_actions.txt")
    # for i in range(interface.storm_model.nr_observations):
    #     print(interface.storm_model.get_observation_labels(i))
