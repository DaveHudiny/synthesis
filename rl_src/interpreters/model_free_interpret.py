# File: rl_interface.py
# Author: David Hudák
# Purpose: Agent model-free interpreter.

import logging
import os
from collections import Counter
from tqdm import tqdm
import argparse
import stormpy
import tf_agents as tfa
import tensorflow as tf
import dill
import pickle
import sys

import numpy as np

from interpreters.interpret import Interpret
from tf_agents.agents import tf_agent
from agents.recurrent_dqn_agent import Recurrent_DQN_agent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelInfo:
    def __init__(self, nr_observations, observations, act_keywords, nr_actions):
        self.nr_observations = nr_observations
        self.act_keywords = act_keywords
        self.nr_actions = nr_actions
        self.observations = observations

class ModelFreeInterpret(Interpret):
    def __init__(self, nr_episodes, model_info : ModelInfo, file=None):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self.nr_episodes = nr_episodes
        self.model_info = model_info
        self.file = None

    def get_dictionary(self, agent):
        self.agent = agent
        obs_act_dict, memory_dict = self.evaluate_model()
        self.agent = None
        return obs_act_dict, memory_dict, self.model_info.act_keywords 

    def create_timestep(self, mask_size):
        discount = tf.constant([1.0], dtype=tf.float32)
        observation = {
            'mask': tf.constant([[True for _ in range(mask_size)]], dtype=tf.bool),
            'observation': tf.constant([[0]], dtype=tf.float32)
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
        print(states)
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
        action = self.agent.policy(time_step)
        limits_min, limits_max = self._policy_step_limits_from_set([action])
        return limits_min, limits_max

    def generate_policy_step_states_limits(self):
        time_step = self.create_timestep(self.model_info.nr_actions)
        print("Generating policy step states limits.")
        generate_policy_step_states_limits_tf = tf.function(
            self._generate_policy_step_states_limits)
        time_step.observation['obs'] = tf.constant([[0]], dtype=tf.float32)
        limits_min, limits_max = self.initialize_limits(time_step)
        for observation in tqdm(self.unique_observations):
            time_step.observation['obs'] = tf.constant(
                [[observation]], dtype=tf.float32)
            actions = generate_policy_step_states_limits_tf(
                time_step, self.agent.policy, nr_runs=3)
            limits_min, limits_max = self.update_policy_step_limits(
                actions, limits_min, limits_max)
        return limits_min, limits_max

    def _generate_policy_step_states_limits(self, time_step, policy, nr_runs=3):
        actions = [policy(time_step)]
        for _ in range(nr_runs):
            action = policy(time_step, actions[-1].state)
            actions.append(action)
        return actions

    def monte_carlo_evaluation(self, time_step, nr_episodes, policy_limits_min=[0], policy_limits_max=[1], policy=None):
        unique_actions = np.empty(shape=(0,), dtype=np.int32)
        action = self.agent.policy(time_step)
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
            action_keywords += f"{', ' if len(action_keywords) > 0 else ''}{self.model_info.act_keywords[action]}"
        return action_numbers, action_keywords

    def print_triplet(self, observation, actions, memory, file=sys.stdout):
        acts, act_keys = self.make_actions_printable(actions)
        print(f"Observation: {observation}. Actions: {acts}. " +
              f"Action Keywords: {act_keys}", file=file)
        print(f"Action statistics: {memory}", file=file)

    def memory_estimation(self, action_stats_dict, std_multiplier_threshold=0.15):
        numpy_values = np.array(list(action_stats_dict.values()))
        mean = np.mean(numpy_values)
        std = np.std(numpy_values)
        threshold = mean - std_multiplier_threshold * std
        action_stats_dict_filtered = {
            k: v for k, v in action_stats_dict.items() if v >= threshold}
        assert len(
            action_stats_dict_filtered) > 0, f"Overfiltered. action_stats_dict: {action_stats_dict}. action_stats_dict_filtered: {action_stats_dict_filtered}. mean: {mean}. std: {std}. threshold: {threshold}. std_multiplier_threshold: {std_multiplier_threshold}."
        return list(action_stats_dict_filtered.keys()), len(action_stats_dict_filtered)

    def evaluate_model(self, nr_episodes=10, method="monte_carlo"):
        time_step = self.create_timestep(self.model_info.nr_actions)
        self.unique_observations = np.unique(self.model_info.observations)
        limits_min, limits_max = self.generate_policy_step_states_limits()
        policy_function = tf.function(self.agent.policy)
        observation_dict = {}
        observation_memory_size = {}
        logger.debug("Evaluating model.")
        for observation in tqdm(self.unique_observations):
            time_step.observation['obs'] = tf.constant(
                [[observation]], dtype=tf.float32)
            actions, action_stats_dict = self.monte_carlo_evaluation(
                time_step, nr_episodes, limits_min, limits_max, policy=policy_function)
            actions_filtered, memory = self.memory_estimation(
                action_stats_dict)
            if self.file is not None:
                self.print_triplet(observation, actions_filtered, memory, file=self.file)
            observation_dict[observation] = actions_filtered
            observation_memory_size[observation] = memory
        return observation_dict, observation_memory_size

    def save_evaluation(self, observation_dict, observation_memory_size):
        with open("obs_actions.pickle", "wb") as file_pickle:
            pickle.dump(observation_dict, file_pickle)
        with open("obs_memory.pickle", "wb") as file_pickle:
            pickle.dump(observation_memory_size, file_pickle)
