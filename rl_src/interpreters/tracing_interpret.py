from interpreters.interpret import Interpret
from environment.environment_wrapper import Environment_Wrapper
from tf_agents.environments import tf_py_environment
from agents.father_agent import FatherAgent

import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


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


class TracingInterpret(Interpret):
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

    def get_dictionary(self, agent=None, with_refusing=True):
        """Gets the dictionary of observations to actions, memory dictionary, action keywords and observation prioritizer.
            Used as main interface between RL algorithms and Paynt."""
        obs_act_stats_dict = self.compute_observation_action_dictionary(
            agent=agent, with_refusing=with_refusing)
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

    def update_obs_act_dict(self, observation, action):
        if observation not in self.obs_act_dict:
            self.obs_act_dict[observation] = {}

        if action not in self.obs_act_dict[observation]:
            self.obs_act_dict[observation][action] = 1
        else:
            self.obs_act_dict[observation][action] += 1

    def merge_dicts(self, dict1, dict2):
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]
            else:
                for key2 in dict2[key]:
                    if key2 not in dict1[key]:
                        dict1[key][key2] = dict2[key][key2]
                    else:
                        dict1[key][key2] += dict2[key][key2]
        return dict1

    def update_aux_obs_act_dict(self, observation, action):
        if observation not in self.aux_obs_act_dict:
            self.aux_obs_act_dict[observation] = {}

        if action not in self.aux_obs_act_dict[observation]:
            self.aux_obs_act_dict[observation][action] = 1
        else:
            self.aux_obs_act_dict[observation][action] += 1

    def compute_observation_action_dictionary(self, num_episodes=50, agent: FatherAgent = None, with_refusing=False):
        """Computes the dictionary of observations to actions.

        Args:
            num_episodes (int, optional): The number of episodes to be traced. Defaults to 50.
            agent (FatherAgent, optional): The agent to be traced. Defaults to None.
            with_refusing (bool, optional): Whether to use refusing. Defaults to False.

        Returns:
            dict: The dictionary of observations to actions with counts of each action.
        """
        if agent is None:
            raise ValueError("Agent must be provided.")
        self.obs_act_dict = {}
        policy = tf.function(agent.get_evaluation_policy().action)
        result_info = ResultInfo()
        step_rewards = []
        final_rewards = []
        for i in range(num_episodes):
            time_step = self.tf_environment.reset()
            policy_state = agent.get_initial_state(None)
            steps = 0
            self.aux_obs_act_dict = {}
            reward = 0

            while not time_step.is_last():
                action_step = policy(time_step, policy_state)
                policy_state = action_step.state
                observation = int(
                    self.environment.simulator._report_observation())
                time_step = self.tf_environment.step(action_step.action)
                action = int(self.environment.last_action)
                if with_refusing:
                    self.update_aux_obs_act_dict(observation, action)
                else:
                    self.update_obs_act_dict(observation, action)
                steps += 1
                reward += time_step.reward
            labels = self.environment.simulator._report_labels()
            if self.environment.is_goal_state(labels):
                if self.environment.normalize_simulator_rewards:
                    reward -= 1.0
                else:
                    reward -= self.environment.goal_value
            else:
                reward -= time_step.reward  # Removing the last reward (goal)
            goal_reward = time_step.reward
            step_rewards.append(reward.numpy())
            final_rewards.append(goal_reward.numpy())
            if with_refusing:
                if time_step.is_last() and self.environment.is_goal_state(labels):
                    self.obs_act_dict = self.merge_dicts(
                        self.obs_act_dict, self.aux_obs_act_dict)

            result_info.update_ending_stats(steps, self.environment._max_steps, time_step.is_last(),
                                            self.environment.simulator._report_labels(), self.environment.is_goal_state)

        logger.info(f"{result_info}")
        logger.info(f"Average reward without goal: {np.mean(step_rewards)}")
        logger.info(f"Average final reward: {np.mean(final_rewards)}")

        return self.obs_act_dict
