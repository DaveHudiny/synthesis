from environment.environment_wrapper import Environment_Wrapper
from environment import tf_py_environment
from agents.father_agent import FatherAgent

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.trajectories.trajectory import Trajectory

import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

import numpy as np
from collections import defaultdict

class ResultInfo:
    def __init__(self):
        self.ending_list = ["goal_end", "trap_end",
                            "max_step_end", "unknown_end"]
        self.ending_stats_dict = {key: 0 for key in self.ending_list}

    def update_ending_stats(self, steps, max_steps, is_done, report_labels, is_goal_state : callable):
        if steps >= max_steps:
            self.ending_stats_dict["max_step_end"] += 1
        elif is_done and is_goal_state(report_labels):
            self.ending_stats_dict["goal_end"] += 1
        elif is_done and 'traps' in report_labels:
            self.ending_stats_dict["trap_end"] += 1
        elif is_done:
            self.ending_stats_dict["unknown_end"] += 1

    def __str__(self):
        return f"Ending stats: {self.ending_stats_dict}"


class TracingInterpret:
    """Interpretation class for tracing the environment and agent interaction.

    This class is used to trace the interaction between the environment and the agent.
    Result of the tracing is a dictionary of observations to actions, which is usable by the Paynt."""

    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 encoding_type="One-Hot"):
        """Initializes the TracingInterpret class.

        Args:
            environment (Environment_Wrapper): The environment to be traced.
            tf_environment (tf_py_environment.TFPyEnvironment): The environment to be traced in TensorFlow format.
            encoding_type (str, optional): The encoding type of the observations. Defaults to "One-Hot".
            possible_observations (list, optional): The list of possible observations. Important when working with One-Hot encoding. Defaults to None."
        """
        self.environment, _ = self.parse_environment(environment)
        self.tf_environment, _ = self.parse_tf_environment(tf_environment)
        self.obs_action_dict = {}
        self.encoding_type = encoding_type
        self.possible_observations = self.environment._possible_observations

    def parse_environment(self, environment) -> tuple[Environment_Wrapper, Environment_Wrapper]:
        if isinstance(environment, dict):
            return environment["eval_model"], environment["train_model"]
        else:
            return environment, environment

    def parse_tf_environment(self, tf_environment) -> tuple[tf_py_environment.TFPyEnvironment, tf_py_environment.TFPyEnvironment]:
        if isinstance(tf_environment, dict):
            return tf_environment["eval_sim"], tf_environment["train_sim"]
        else:
            return tf_environment, tf_environment

    def create_observation_prioritizer_by_sorting(self, obs_act_dict_counts):
        """Creates the observation prioritizer by sorting the observations.

        Args:
            obs_act_dict_counts (dict): The dictionary of observations to actions with counts.

        Returns:
            list: The list of indices of observations sorted by the variance of actions."""
        obs_variances = np.ones((len(self.possible_observations),))

        for obs in obs_act_dict_counts:
            numpy_values = np.array(list(obs_act_dict_counts[obs].values()))
            if len(numpy_values) == 0:
                continue
            std = np.std(numpy_values)
            obs_variances[obs] += std
        return np.argsort(obs_variances)

    def get_dictionary(self, agent=None, with_refusing=True, vectorized=True, randomize_illegal_actions=True):
        """Gets the dictionary of observations to actions, memory dictionary, action keywords and observation prioritizer.
            Used as main interface between RL algorithms and Paynt."""
        obs_act_stats_dict = self.compute_observation_action_dictionary(
            agent=agent, with_refusing=with_refusing, vectorized=vectorized, randomize_illegal_actions=randomize_illegal_actions)
        obs_act_dict, memory_dict = self.compute_cutted_dictionary(
            obs_act_stats_dict, cut_actions=False, memory_greediness=0.5)
        prioritizer = self.create_observation_prioritizer_by_sorting(
            obs_act_stats_dict)

        return obs_act_dict, memory_dict, self.environment.act_to_keywords, prioritizer

    def plot_histogram(self, obs_act_dict):
        nr_observations = self.environment.stormpy_model.nr_observations
        obs_action_sums_dict = {}
        import matplotlib.pyplot as plt
        for obs in range(nr_observations):
            obs_action_sums_dict[obs] = 0
        for obs in obs_act_dict:
            obs_action_sums_dict[obs] = sum(obs_act_dict[obs].values())
        logger.info("Number of observations after interpretation:", len(
            obs_act_dict), "Number of observations in total:", nr_observations)
        logger.info("Number of observations with more than 1 action:", len(
            [k for k, v in obs_action_sums_dict.items() if v > 1]))
        plt.hist(obs_action_sums_dict.values(),
                 bins=15, align='left', rwidth=0.8)
        plt.xticks(range(max(obs_action_sums_dict.keys()) + 1))
        plt.show()

    def compute_cutted_dictionary(self, obs_act_dict_counts, cut_actions=False, memory_greediness=0.5):
        """Computes the cutted dictionary of observations to actions.

        Args:
            obs_act_dict_counts (dict): The dictionary of observations to actions with counts.
            cut_actions (bool, optional): Whether to cut the actions. Defaults to False.
            memory_greediness (float, optional): The greediness of the memory, which determines how strict is cutting the memory. 
                                                 The lower number, the more strict cutting. Defaults to 0.5.

        Returns:
            dict: The dictionary of observations to actions.
            dict: The dictionary of observations to memory.
        """
        obs_act_dict = {}
        memory_dict = {}
        if not cut_actions:
            for obs in range(self.environment.stormpy_model.nr_observations):
                if obs in obs_act_dict_counts:
                    obs_act_dict[obs] = list(obs_act_dict_counts[obs].keys())
        for obs in range(self.environment.stormpy_model.nr_observations):
            memory_dict[obs] = 1
        for obs in obs_act_dict_counts:
            obs_act_dict_counts[obs] = {
                k: v for k, v in obs_act_dict_counts[obs].items() if v > 1}
            numpy_values = np.array(list(obs_act_dict_counts[obs].values()))
            if len(numpy_values) == 0:
                continue
            mean = np.mean(numpy_values)
            std = np.std(numpy_values)
            threshold = mean - memory_greediness * std
            action_stats_dict_filtered = {
                k: v for k, v in obs_act_dict_counts[obs].items() if v >= threshold}
            if cut_actions:
                obs_act_dict[obs] = list(action_stats_dict_filtered.keys())
            memory_dict[obs] = len(action_stats_dict_filtered.keys())
            if memory_dict[obs] <= 0:
                memory_dict[obs] = 1
        return obs_act_dict, memory_dict
    
    class ObsActUpdater:
        def __init__(self, randomize_illegal_actions=False):
            self.obs_act_dict = defaultdict(lambda: defaultdict(int))
            self.illegal_obs_act_dict = defaultdict(lambda: defaultdict(int))
            self.randomize_illegal_actions = randomize_illegal_actions

        def update_obs_act_dict(self, item: Trajectory):
            observations = item.observation["integer"].numpy()
            mask = item.observation["mask"].numpy()
            actions = item.action.numpy()
            for obs, act, sub_mask in zip(observations, actions, mask):
                # if not sub_mask[act] and self.randomize_illegal_actions:
                #     number_of_legal_actions = np.sum(sub_mask)
                #     for i in range(len(sub_mask)):
                #         if sub_mask[i]:
                #             self.obs_act_dict[obs[0]][i] += 1.0 / float(number_of_legal_actions)
                if not sub_mask[act]:
                    self.illegal_obs_act_dict[obs[0]][act] += 1
                else:
                    self.obs_act_dict[obs[0]][act] += 1
        
        def get_obs_act_dict(self):
            return self.obs_act_dict

    def init_observation_action_driver(self, vectorized = True, agent : FatherAgent = None, with_refusing=False, randomize_illegal_actions=False):
        self.obs_act_dict = {}
        self.aux_obs_act_dict = {}
        self.obs_act_updater = self.ObsActUpdater(randomize_illegal_actions=randomize_illegal_actions)
        num_steps = 1 if not vectorized else agent.args.num_environments * agent.args.trajectory_num_steps
        # logger.info(f"Number of steps for tracing is {num_steps}")
        self.obs_act_driver = DynamicStepDriver(
            self.tf_environment, agent.get_evaluation_policy(), observers=[self.obs_act_updater.update_obs_act_dict], num_steps=num_steps
        )

    def compute_observation_action_dictionary(self, num_episodes=50, agent: FatherAgent = None, with_refusing=False, vectorized=True, randomize_illegal_actions=True):
        """Computes the dictionary of observations to actions.

        Args:
            num_episodes (int, optional): The number of episodes to be traced. Defaults to 50.
            agent (FatherAgent, optional): The agent to be traced. Defaults to None.
            with_refusing (bool, optional): Whether to use refusing. Defaults to False.

        Returns:
            dict: The dictionary of observations to actions with counts of each action.
        """
        self.init_observation_action_driver(vectorized, with_refusing=with_refusing, agent=agent, randomize_illegal_actions=randomize_illegal_actions)
        for _ in range(16):
            self.obs_act_driver.run()
        return self.obs_act_updater.get_obs_act_dict()
