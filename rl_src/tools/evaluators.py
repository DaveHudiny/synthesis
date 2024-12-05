# Description: This file contains the function for evaluation of model.
# The function computes the average return of the policy over the given number of episodes.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: evaluators.py

import tensorflow as tf
import numpy as np

from environment.environment_wrapper import Environment_Wrapper
from environment.environment_wrapper_vec import Environment_Wrapper_Vec
from environment import tf_py_environment

from tf_agents.policies import TFPolicy

from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import Trajectory

import logging

logger = logging.getLogger(__name__)


class EvaluationResults:
    """Class for storing evaluation results."""

    def __init__(self, goal_value: tf.Tensor = tf.constant(0.0)):
        self.best_episode_return = tf.float32.min
        self.best_return = tf.float32.min
        self.goal_value = goal_value.numpy()
        self.returns_episodic = []
        self.returns = []
        self.reach_probs = []
        self.trap_reach_probs = []
        self.best_reach_prob = 0.0
        self.losses = []
        self.best_updated = False
        # self.each_episode_returns = []
        # self.each_episode_successes = []
        self.each_episode_variance = []
        self.each_episode_virtual_variance = []
        self.combined_variance = []
        self.num_episodes = []

    def set_experiment_settings(self, learning_algorithm: str = "", learning_rate: float = float("nan"),
                                nn_details: dict = {}, max_steps: int = float("nan")):
        self.learning_algorithm = learning_algorithm
        self.learning_rate = learning_rate
        self.nn_details = "Not implemented yet"
        self.max_steps = max_steps

    def save_to_json(self, filename, evaluation_time: float = float("nan")):
        import json
        with open(filename, "w") as file:
            _dict_ = self.__dict__.copy()
            del _dict_["best_updated"]
            for key in _dict_:
                _dict_[key] = str(_dict_[key])
            _dict_["evaluation_time"] = evaluation_time
            json.dump(_dict_, file)

    def load_from_json(self, filename):
        import json  # TODO probably not working with float32 conversion
        with open(filename, "r") as file:
            json_dict = json.load(file)
            self.__dict__.update(json_dict)

    def __str__(self):
        return str(self.__dict__)

    def update(self, avg_return, avg_episodic_return, reach_prob, episodes_variance=None, num_episodes=1, trap_reach_prob=0.0, virtual_variance=None, combined_variance=None):
        """Update the evaluation results in the object of EvaluationResults.

        Args:
            avg_episodic_return (tf.float32): Cumulative return of the policy virtual goal.
            avg_return (tf.float32): Cumulative return of the policy.
            reach_prob (tf.float32): Probability of reaching the goal.
        """
        self.best_updated = False
        self.returns_episodic.append(avg_episodic_return)
        self.returns.append(avg_return)
        self.reach_probs.append(reach_prob)
        self.each_episode_variance.append(episodes_variance)
        self.num_episodes.append(num_episodes)
        self.trap_reach_probs.append(trap_reach_prob)
        if avg_return > self.best_return:
            self.best_return = avg_return
            if avg_episodic_return >= self.best_episode_return:
                self.best_updated = True
        if avg_episodic_return > self.best_episode_return:
            self.best_episode_return = avg_episodic_return
            self.best_updated = True
        if reach_prob > self.best_reach_prob:
            self.best_reach_prob = reach_prob
        if virtual_variance is not None:
            self.each_episode_virtual_variance.append(virtual_variance)
        if combined_variance is not None:
            self.combined_variance.append(combined_variance)

    def add_loss(self, loss):
        """Add loss to the list of losses."""
        self.losses.append(loss)


def log_evaluation_info(evaluation_result: EvaluationResults = None):
    logger.info('Average Return = {0}'.format(
        evaluation_result.returns[-1]))
    logger.info('Average Virtual Goal Value = {0}'.format(
        evaluation_result.returns_episodic[-1]))
    logger.info('Goal Reach Probability = {0}'.format(
        evaluation_result.reach_probs[-1]))
    logger.info('Trap Reach Probability = {0}'.format(
        evaluation_result.trap_reach_probs[-1]))
    logger.info('Variance of Return = {0}'.format(
        evaluation_result.each_episode_variance[-1]))
    logger.info('Current Best Return = {0}'.format(
        evaluation_result.best_return))
    logger.info('Current Best Reach Probability = {0}'.format(
        evaluation_result.best_reach_prob))


class TrajectoryBuffer:
    class EpisodeOutcomes:
        def __init__(self, virtual_rewards: list = [], cumulative_rewards: list = [], goals_achieved: list = [], traps: list = []):
            assert type(virtual_rewards) == list and type(cumulative_rewards) == list and type(
                goals_achieved) == list and type(traps) == list, "All arguments must be lists."
            self.virtual_rewards = virtual_rewards
            self.cumulative_rewards = cumulative_rewards
            self.goals_achieved = goals_achieved
            self.traps_achieved = traps

        def add_episode_outcome(self, virtual_reward, cumulative_reward, goal_achieved, trap_achieved):
            self.virtual_rewards.append(virtual_reward)
            self.cumulative_rewards.append(cumulative_reward)
            self.goals_achieved.append(goal_achieved)
            self.traps_achieved.append(trap_achieved)

        def clear(self):
            self.virtual_rewards = []
            self.cumulative_rewards = []
            self.goals_achieved = []
            self.traps_achieved = []

    def __init__(self, environment: Environment_Wrapper_Vec = None):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.tf_step_types = []
        self.environment = environment
        self.episode_outcomes = self.EpisodeOutcomes([], [], [], [])

    def add_batched_step(self, traj: Trajectory):
        environment = self.environment
        self.virtual_rewards.append(environment.virtual_reward.numpy())
        self.real_rewards.append(environment.default_rewards.numpy())
        self.finished.append(environment.dones)
        self.finished_successfully.append(environment.goal_state_mask)
        self.finished_truncated.append(environment.truncated)
        self.finished_traps.append(environment.anti_goal_state_mask)

    def numpize_lists(self):
        self.virtual_rewards = np.array(self.virtual_rewards).T
        self.real_rewards = np.array(self.real_rewards).T
        self.finished = np.array(self.finished).T
        self.finished_successfully = np.array(self.finished_successfully).T
        self.finished_truncated = np.array(self.finished_truncated).T
        self.finished_traps = np.array(self.finished_traps).T

    def update_outcomes(self):
        self.numpize_lists()
        outcomes = self.episode_outcomes
        finished_true_indices = np.argwhere(self.finished == True)
        prev_index = np.array([0, -1])
        for index in finished_true_indices:
            if index[0] != prev_index[0]:
                prev_index = np.array([index[0], -1])
            in_episode_reward = np.sum(
                self.real_rewards[prev_index[0], prev_index[1]+1:index[1]+1])
            in_episode_virtual_reward = np.sum(
                self.virtual_rewards[prev_index[0], prev_index[1]+1:index[1]+1])
            goal_achieved = self.finished_successfully[index[0], index[1]]
            trap_achieved = self.finished_traps[index[0], index[1]]
            outcomes.add_episode_outcome(
                in_episode_virtual_reward, in_episode_reward, goal_achieved, trap_achieved)
            prev_index = index

    def final_update_of_results(self, updator: callable = None):
        self.update_outcomes()
        avg_return = np.mean(self.episode_outcomes.cumulative_rewards)
        avg_episode_return = np.mean(self.episode_outcomes.virtual_rewards)
        reach_prob = np.mean(self.episode_outcomes.goals_achieved)
        trap_prob = np.mean(self.episode_outcomes.traps_achieved)
        episode_variance = np.var(self.episode_outcomes.cumulative_rewards)
        virtual_variance = np.var(self.episode_outcomes.virtual_rewards)
        combined_variance = np.var(
            np.array(self.episode_outcomes.cumulative_rewards) + np.array(self.episode_outcomes.virtual_rewards))
        if updator:
            # updator(avg_return, avg_episode_return, reach_prob, returns, successes)
            updator(avg_return, avg_episode_return, reach_prob, episode_variance, num_episodes=len(self.episode_outcomes.cumulative_rewards),
                    trap_reach_prob=trap_prob, virtual_variance=virtual_variance, combined_variance=combined_variance)
        return avg_return, avg_episode_return, reach_prob

    def clear(self):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.episode_outcomes.clear()


def compute_average_return(policy: TFPolicy, tf_environment: tf_py_environment.TFPyEnvironment, num_episodes=30,
                           environment: Environment_Wrapper = None, updator: callable = None, custom_runner: callable = None):
    """Compute the average return of the policy over the given number of episodes."""
    total_return, episode_return, goals_visited, traps_visited = 0.0, 0.0, 0, 0
    returns = []
    episodic_returns = []
    if custom_runner is None:
        policy_function = tf.function(policy.action)

    for _ in range(num_episodes):
        if custom_runner is None:
            cumulative_return, episode_goal_visited, episode_trap_visited = run_single_episode(
                policy, policy_function, tf_environment, environment)
        else:
            cumulative_return, episode_goal_visited = custom_runner(
                tf_environment, environment)
        total_return, episode_return, goals_visited, traps_visited = process_episode_results(
            cumulative_return, total_return, episode_return, environment, returns, episodic_returns,
            episode_goal_visited, episode_trap_visited,  goals_visited, traps_visited)

    avg_return, avg_episode_return, reach_prob, episode_variance, virtual_variance, combined_variance, trap_reach_prob = calculate_statistics(
        total_return, episode_return, goals_visited, traps_visited, num_episodes, returns, episodic_returns)

    if updator:
        updator(avg_return, avg_episode_return, reach_prob, episode_variance,
                num_episodes=num_episodes, trap_reach_prob=trap_reach_prob,
                virtual_variance=virtual_variance, combined_variance=combined_variance)

    return avg_return, avg_episode_return, reach_prob


def run_single_episode(policy, policy_function, tf_environment: tf_py_environment.TFPyEnvironment, environment):
    """Run a single episode and return the cumulative reward and success flag."""
    time_step = tf_environment.reset()
    policy_state = policy.get_initial_state(None)
    cumulative_return = 0.0
    goal_visited, trap_visited = False, False

    while not time_step.is_last():
        action_step = policy_function(time_step, policy_state=policy_state)
        policy_state = action_step.state
        time_step = tf_environment.step(action_step.action)
        cumulative_return += time_step.reward / environment.normalizer

    if environment and environment.flag_goal:
        goal_visited = True
    if environment and environment.flag_trap:
        trap_visited = True

    return cumulative_return, goal_visited, trap_visited


def process_episode_results(cumulative_return: float, total_return, episode_return,
                            environment: Environment_Wrapper, returns: list, episodic_returns: list,
                            goal_visited: bool, trap_visited: bool, goals_visited: int, traps_visited: int):
    """Update the cumulative return, episode return and success data after an episode."""
    if environment:
        cumulative_return -= environment.virtual_value
        total_return += cumulative_return.numpy()[0]
        returns.append(cumulative_return.numpy()[0])
        episodic_returns.append(environment.virtual_value)
        episode_return += environment.virtual_value.numpy()
        if goal_visited:
            goals_visited += 1
        if trap_visited:
            traps_visited += 1
    else:
        returns.append(cumulative_return.numpy()[0])
        episodic_returns.append(cumulative_return.numpy()[0])
        episode_return += cumulative_return.numpy()[0]
        total_return += cumulative_return.numpy()[0]

    return total_return, episode_return, goals_visited, traps_visited


def calculate_statistics(total_return, episode_return, goal_visited, traps_visited, num_episodes, returns, episode_returns=None):
    """Calculate average return, episode return and success probability."""
    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    reach_prob = goal_visited / num_episodes
    episode_variance = np.var(returns)
    virtual_variance = np.var(episode_returns)
    combined_variance = np.var(np.array(returns) + np.array(episode_returns))
    trap_reach_prob = traps_visited / num_episodes
    return avg_return, avg_episode_return, reach_prob, episode_variance, virtual_variance, combined_variance, trap_reach_prob
