import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

import numpy as np

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
        self.paynt_bounds = [] # Shape (n, 2), where n is the number of iterations of PAYNT<->RL loop and 2 is the bound and number of iteration of each bound given the current iteration of RL.
        self.last_from_interpretation = False

        self.extracted_fsc_episode_return = -1.0
        self.extracted_fsc_return = -1.0
        self.extracted_fsc_reach_prob = -1.0
        self.extracted_fsc_variance = -1.0
        self.extracted_fsc_num_episodes = -1
        self.extracted_fsc_virtual_variance = -1.0
        self.extracted_fsc_combined_variance = -1.0

        self.artificial_reward_means = []
        self.artificial_reward_stds = []

    def add_artificial_reward(self, artificial_rewards_buffer : list[np.ndarray]):
        """Add artificial rewards to the evaluation results."""
        if len(artificial_rewards_buffer) > 0:
            self.artificial_reward_means.append(np.mean(artificial_rewards_buffer))
            self.artificial_reward_stds.append(np.std(artificial_rewards_buffer))
        else:
            self.artificial_reward_means.append(float("nan"))
            self.artificial_reward_stds.append(float("nan"))

    def set_experiment_settings(self, learning_algorithm: str = "", learning_rate: float = float("nan"),
                                nn_details: dict = {}, max_steps: int = float("nan")):
        self.learning_algorithm = learning_algorithm
        self.learning_rate = learning_rate
        self.nn_details = "Not implemented yet"
        self.max_steps = max_steps

    def add_paynt_bound(self, bound: float):
        number_of_iterations = len(self.returns)
        self.paynt_bounds.append([bound, number_of_iterations])

    def save_to_json(self, filename, evaluation_time: float = float("nan"), split_iteration = -1):
        import json
        with open(filename, "w") as file:
            _dict_ = self.__dict__.copy()
            del _dict_["best_updated"]
            for key in _dict_:
                _dict_[key] = str(_dict_[key])
            _dict_["evaluation_time"] = evaluation_time
            _dict_["split_iteration"] = split_iteration
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


def set_fsc_values_to_evaluation_result(external_evaluation_result : EvaluationResults, evaluation_result : EvaluationResults):
    external_evaluation_result.last_from_interpretation = True
    external_evaluation_result.extracted_fsc_episode_return = evaluation_result.returns_episodic[-1]
    external_evaluation_result.extracted_fsc_return = evaluation_result.returns[-1]
    external_evaluation_result.extracted_fsc_reach_prob = evaluation_result.reach_probs[-1]
    external_evaluation_result.extracted_fsc_num_episodes = evaluation_result.num_episodes[-1]
    external_evaluation_result.extracted_fsc_variance = evaluation_result.each_episode_variance[-1]
    external_evaluation_result.extracted_fsc_virtual_variance = evaluation_result.each_episode_virtual_variance[-1]
    external_evaluation_result.extracted_fsc_combined_variance = evaluation_result.combined_variance[-1]


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