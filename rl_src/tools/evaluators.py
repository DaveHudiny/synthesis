# Description: This file contains the function for evaluation of model.
# The function computes the average return of the policy over the given number of episodes.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: evaluators.py

import tensorflow as tf

from environment.environment_wrapper import Environment_Wrapper
from tf_agents.environments import tf_py_environment


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
        import json # TODO probably not working with float32 conversion
        with open(filename, "r") as file:
            json_dict = json.load(file)
            self.__dict__.update(json_dict)

    def __str__(self):
        return str(self.__dict__)

    def update(self, avg_return, avg_episodic_return, reach_prob):
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


def compute_average_return(policy, tf_environment : tf_py_environment.TFPyEnvironment, num_episodes=30, environment: Environment_Wrapper = None, updator: callable=None):
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
    for _ in range(num_episodes):
        time_step = tf_environment._reset()
        policy_state = policy.get_initial_state(None)
        while not time_step.is_last():
            action_step = policy_function(
                time_step, policy_state=policy_state)
            action = action_step.action
            policy_state = action_step.state
            time_step = tf_environment.step(action)
            total_return += time_step.reward / environment.normalizer
        if environment is not None:
            total_return -= environment.virtual_value
            episode_return += environment.virtual_value
            if environment.flag_goal:
                goal_visited += 1
        else:
            total_return -= time_step.reward
            episode_return += time_step.reward

    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    reach_prob = goal_visited / num_episodes
    
    if updator is not None:
        updator(avg_return.numpy()[0], avg_episode_return.numpy(), reach_prob)
    return avg_return.numpy()[0], avg_episode_return.numpy(), reach_prob
