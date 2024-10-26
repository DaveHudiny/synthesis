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
from tf_agents.environments import tf_py_environment

from tf_agents.policies import TFPolicy

from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import Trajectory


class EvaluationResults:
    """Class for storing evaluation results."""

    def __init__(self, goal_value: tf.Tensor = tf.constant(0.0)):
        self.best_episode_return = tf.float32.min
        self.best_return = tf.float32.min
        self.goal_value = goal_value.numpy()
        self.returns_episodic = []
        self.returns = []
        self.reach_probs = []
        self.best_reach_prob = 0.0
        self.losses = []
        self.best_updated = False
        self.each_episode_returns = []
        self.each_episode_successes = []

    def set_experiment_settings(self, learning_algorithm: str = "", learning_rate: float = float("nan"),
                                nn_details: dict = {}, max_steps: int = float("nan")):
        self.learning_algorithm = learning_algorithm
        self.learning_rate = learning_rate
        self.nn_details = "Not implemented yet"
        self.max_steps = max_steps

    def save_to_json(self, filename):
        import json
        with open(filename, "w") as file:
            _dict_ = self.__dict__.copy()
            del _dict_["best_updated"]
            for key in _dict_:
                _dict_[key] = str(_dict_[key])
            json.dump(_dict_, file)

    def load_from_json(self, filename):
        import json  # TODO probably not working with float32 conversion
        with open(filename, "r") as file:
            json_dict = json.load(file)
            self.__dict__.update(json_dict)

    def __str__(self):
        return str(self.__dict__)

    def update(self, avg_return, avg_episodic_return, reach_prob, each_episode_return=None, each_episode_success=None):
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
        self.each_episode_returns.append(each_episode_return)
        self.each_episode_successes.append(each_episode_success)
        if avg_return > self.best_return:
            self.best_return = avg_return
            if avg_episodic_return >= self.best_episode_return:
                self.best_updated = True
        if avg_episodic_return > self.best_episode_return:
            self.best_episode_return = avg_episodic_return
            self.best_updated = True
        if reach_prob > self.best_reach_prob:
            self.best_reach_prob = reach_prob

    def add_loss(self, loss):
        """Add loss to the list of losses."""
        self.losses.append(loss)


def compute_average_return2(policy: TFPolicy, tf_environment: tf_py_environment.TFPyEnvironment, num_episodes=30,
                           environment: Environment_Wrapper = None, updator: callable = None):
    """Compute the average return of the policy over the given number of episodes.

    Args:
        policy: The policy to evaluate.
        tf_environment: The environment to evaluate the policy on.
        num_episodes: The number of episodes to run the evaluation for.
        environment: The environment wrapper object. Used for computing exact return without virtual goal.
        updator: The function for updating the evaluation results (e.g. update() from EvaluationResults class).
    """
    total_return = 0.0
    episode_return = 0.0
    goal_visited = 0
    policy_function = tf.function(policy.action)
    returns = []
    successes = []
    for _ in range(num_episodes):
        time_step = tf_environment._reset()
        policy_state = policy.get_initial_state(None)
        cumulative_return = 0.0
        while not time_step.is_last():
            action_step = policy_function(
                time_step, policy_state=policy_state)
            action = action_step.action
            policy_state = action_step.state
            time_step = tf_environment.step(action)
            total_return += time_step.reward / environment.normalizer
            cumulative_return += time_step.reward / environment.normalizer
        if environment is not None:
            total_return -= environment.virtual_value
            cumulative_return -= environment.virtual_value
            returns.append(cumulative_return.numpy()[0])
            episode_return += environment.virtual_value
            if environment.flag_goal:
                goal_visited += 1
                successes.append(True)
            else:
                successes.append(False)
        else:
            total_return -= time_step.reward
            episode_return += time_step.reward

    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    reach_prob = goal_visited / num_episodes

    if updator is not None:
        updator(avg_return.numpy()[0], avg_episode_return.numpy(
        ), reach_prob, returns, successes)
    return avg_return.numpy()[0], avg_episode_return.numpy(), reach_prob

class TrajectoryBuffer:
    class EpisodeOutcomes:
        def __init__(self, virtual_rewards : list = [], cumulative_rewards : list = [], goals_achieved : list = [], traps : list = []):
            assert type(virtual_rewards) == list and type(cumulative_rewards) == list and type(goals_achieved) == list and type(traps) == list, "All arguments must be lists."
            self.virtual_rewards = virtual_rewards
            self.cumulative_rewards = cumulative_rewards
            self.goals_achieved = goals_achieved
            self.traps_achieved = traps
        
        def add_episode_outcome(self, virtual_reward, cumulative_reward, goal_achieved, trap_achieved):
            self.virtual_rewards.append(virtual_reward)
            self.cumulative_rewards.append(cumulative_reward)
            self.goals_achieved.append(goal_achieved)
            self.traps_achieved.append(trap_achieved)

    def __init__(self, environment: Environment_Wrapper_Vec = None):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.tf_step_types = []
        self.environment = environment

    def add_batched_step(self, traj : Trajectory):
        environment = self.environment
        self.virtual_rewards.append(environment.virtual_reward.numpy())
        self.real_rewards.append(environment.default_rewards.numpy())
        self.finished.append(environment.dones)
        self.finished_successfully.append(environment.goal_state_mask)
        self.finished_truncated.append(environment.truncated)
        self.finished_traps.append(environment.anti_goal_state_mask)

    def numpize_lists(self):
        self.virtual_rewards = np.array(self.virtual_rewards)
        self.real_rewards = np.array(self.real_rewards)
        self.finished = np.array(self.finished)
        self.finished_successfully = np.array(self.finished_successfully)
        self.finished_truncated = np.array(self.finished_truncated)
        self.finished_traps = np.array(self.finished_traps)

    def compute_outcomes(self):
        self.numpize_lists()
        outcomes = self.EpisodeOutcomes()
        finished_true_indices = np.argwhere(self.finished == True)
        prev_index = np.array([0, -1])
        for index in finished_true_indices[1:]:
            if index[0] != prev_index[0]:
                prev_index = np.array([index[0], ])
            in_episode_reward = np.sum(self.real_rewards[prev_index[0], prev_index[1]+1:index[1]])
            in_episode_virtual_reward = np.sum(self.virtual_rewards[prev_index[0], prev_index[1]+1:index[1]])
            goal_achieved = np.any(self.finished_successfully[prev_index[0], prev_index[1]+1:index[1]])
            trap_achieved = np.any(self.finished_traps[prev_index[0], prev_index[1]+1:index[1]])
            outcomes.add_episode_outcome(in_episode_virtual_reward, in_episode_reward, goal_achieved, trap_achieved)
            prev_index = index
        return outcomes

    def final_update_of_results(self, updator: callable = None):
        outcomes = self.compute_outcomes()
        avg_return = np.mean(outcomes.cumulative_rewards)
        avg_episode_return = np.mean(outcomes.virtual_rewards)
        reach_prob = np.mean(self.finished_successfully)
        if updator:
            # updator(avg_return, avg_episode_return, reach_prob, returns, successes)
            updator(avg_return, avg_episode_return, reach_prob, outcomes.cumulative_rewards, outcomes.goals_achieved)
        return avg_return, avg_episode_return, reach_prob
    
    def clear(self):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []



def compute_vectorized_average_return(policy: TFPolicy, tf_environment: tf_py_environment.TFPyEnvironment, num_steps = 150, num_envs = 256, 
                                      environment: Environment_Wrapper_Vec = None, updator: callable = None):
    """Compute the average return of the policy over the given number of steps."""
    policy_function = tf.function(policy.action)
    buffer = TrajectoryBuffer()
    num_envs = num_envs
    num_steps = num_steps
    time_step = environment.reset()
    policy_state = policy.get_initial_state(num_envs)
    for _ in range(num_steps):
        print("Time step: ", time_step)
        # remove first dimension from time_step
        action_step = policy_function(time_step, policy_state=policy_state)
        policy_state = action_step.state
        print("Actions", action_step.action)
        action = tf.reshape(action_step.action, (num_envs, 1))
        print("Actions reshaped", action)
        time_step = tf_environment.step(action)
        buffer.add_batched_step(time_step, environment)

    return buffer.final_update_of_results(updator)
    
        
        

def compute_average_return(policy: TFPolicy, tf_environment: tf_py_environment.TFPyEnvironment, num_episodes=30,
                           environment: Environment_Wrapper = None, updator: callable = None, custom_runner : callable = None):
    """Compute the average return of the policy over the given number of episodes."""
    total_return, episode_return, goals_visited = 0.0, 0.0, 0
    returns, successes = [], []
    if custom_runner is None:
        policy_function = tf.function(policy.action)

    for _ in range(num_episodes):
        if custom_runner is None:
            cumulative_return, episode_goal_visited = run_single_episode(
                policy, policy_function, tf_environment, environment)
        else:
            cumulative_return, episode_goal_visited = custom_runner(tf_environment, environment)
        total_return, episode_return, goals_visited = process_episode_results(
            cumulative_return, total_return, episode_return, environment, returns, episode_goal_visited, successes, goals_visited)

    avg_return, avg_episode_return, reach_prob = calculate_statistics(
        total_return, episode_return, goals_visited, num_episodes)

    if updator:
        update_results(updator, avg_return, avg_episode_return, reach_prob, returns, successes)

    return avg_return, avg_episode_return, reach_prob

def run_single_episode(policy, policy_function, tf_environment, environment):
    """Run a single episode and return the cumulative reward and success flag."""
    time_step = tf_environment._reset()
    policy_state = policy.get_initial_state(None)
    cumulative_return = 0.0
    goal_visited = False

    while not time_step.is_last():
        action_step = policy_function(time_step, policy_state=policy_state)
        policy_state = action_step.state
        time_step = tf_environment.step(action_step.action)
        cumulative_return += time_step.reward / environment.normalizer

    if environment and environment.flag_goal:
        goal_visited = True

    return cumulative_return, goal_visited


def process_episode_results(cumulative_return, total_return, episode_return, environment, returns, goal_visited, successes, goals_visited):
    """Update the cumulative return, episode return and success data after an episode."""
    if environment:
        cumulative_return -= environment.virtual_value
        total_return += cumulative_return.numpy()[0]
        returns.append(cumulative_return.numpy()[0])
        episode_return += environment.virtual_value.numpy()
        if goal_visited:
            successes.append(True)
            goals_visited += 1
        else:
            successes.append(False)
    else:
        returns.append(cumulative_return.numpy()[0])
        episode_return += cumulative_return.numpy()[0]
        total_return += cumulative_return.numpy()[0]
        successes.append(False)
        

    return total_return, episode_return, goals_visited


def calculate_statistics(total_return, episode_return, goal_visited, num_episodes):
    """Calculate average return, episode return and success probability."""
    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    reach_prob = goal_visited / num_episodes
    return avg_return, avg_episode_return, reach_prob


def update_results(updator, avg_return, avg_episode_return, reach_prob, returns, successes):
    """Update the evaluation results using the provided updater function."""
    updator(avg_return, avg_episode_return, reach_prob, returns, successes)
