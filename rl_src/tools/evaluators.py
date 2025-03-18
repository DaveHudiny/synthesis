# Description: This file contains the function for evaluation of model.
# The function computes the average return of the policy over the given number of episodes.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: evaluators.py

import tensorflow as tf
import numpy as np

from environment.environment_wrapper import Environment_Wrapper
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment import tf_py_environment
# from agents.father_agent import FatherAgent

from tf_agents.policies import TFPolicy

from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import Trajectory
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from interpreters.fsc_based_interpreter import NaiveFSCPolicyExtraction

from tools.args_emulator import ArgsEmulator

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
        self.paynt_bounds = [] # Shape (n, 2), where n is the number of iterations of PAYNT<->RL loop and 2 is the bound and number of iteration of each bound given the current iteration of RL.
        self.last_from_interpretation = False

        self.extracted_fsc_episode_return = -1.0
        self.extracted_fsc_return = -1.0
        self.extracted_fsc_reach_prob = -1.0
        self.extracted_fsc_variance = -1.0
        self.extracted_fsc_num_episodes = -1
        self.extracted_fsc_virtual_variance = -1.0
        self.extracted_fsc_combined_variance = -1.0

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

    def __init__(self, environment: EnvironmentWrapperVec = None):
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
        prev_index = np.array([0, 0])
        for index in finished_true_indices:
            if index[0] != prev_index[0]:
                prev_index = np.array([index[0], 0])
            in_episode_reward = np.sum(
                self.real_rewards[prev_index[0], prev_index[1]:index[1]+1])
            in_episode_virtual_reward = np.sum(
                self.virtual_rewards[prev_index[0], prev_index[1]:index[1]+1])
            goal_achieved = self.finished_successfully[index[0], index[1]]
            trap_achieved = self.finished_traps[index[0], index[1]]
            outcomes.add_episode_outcome(
                in_episode_virtual_reward, in_episode_reward, goal_achieved, trap_achieved)
            prev_index = index
            prev_index[1] += 1

    def final_update_of_results(self, updator: callable = None):
        self.update_outcomes()
        # print(self.episode_outcomes.cumulative_rewards)
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

def set_fsc_values_to_evaluation_result(external_evaluation_result : EvaluationResults, evaluation_result : EvaluationResults):
    external_evaluation_result.last_from_interpretation = True
    external_evaluation_result.extracted_fsc_episode_return = evaluation_result.returns_episodic[-1]
    external_evaluation_result.extracted_fsc_return = evaluation_result.returns[-1]
    external_evaluation_result.extracted_fsc_reach_prob = evaluation_result.reach_probs[-1]
    external_evaluation_result.extracted_fsc_num_episodes = evaluation_result.num_episodes[-1]
    external_evaluation_result.extracted_fsc_variance = evaluation_result.each_episode_variance[-1]
    external_evaluation_result.extracted_fsc_virtual_variance = evaluation_result.each_episode_virtual_variance[-1]
    external_evaluation_result.extracted_fsc_combined_variance = evaluation_result.combined_variance[-1]

def get_new_vectorized_evaluation_driver(tf_environment : tf_py_environment.TFPyEnvironment, environment : EnvironmentWrapperVec, 
                                         custom_policy=None, num_steps=1000) -> tuple[DynamicStepDriver, TrajectoryBuffer]:
    """Create a new vectorized evaluation driver and buffer."""
    trajectory_buffer = TrajectoryBuffer(environment)
    eager = PyTFEagerPolicy(
        policy=custom_policy, use_tf_function=True, batch_time_steps=False)
    vec_driver = DynamicStepDriver(
        tf_environment,
        eager,
        observers=[trajectory_buffer.add_batched_step],
        num_steps=(1 + num_steps) * tf_environment.batch_size
    )
    return vec_driver, trajectory_buffer


def evaluate_extracted_fsc(external_evaluation_result : EvaluationResults, model : str = "", agent = None, 
                           extracted_fsc_policy : NaiveFSCPolicyExtraction = None):
        """Evaluates the extracted FSC. The result is saved to the external_evaluation_result.
        
        Args:
            external_evaluation_result (EvaluationResults): Evaluation results to be updated.
            model (str): Path to the model to be evaluated
            agent (FatherAgent): Agent to be used for evaluation. TODO: Remove cyrcular dependency with some data structure.
            extracted_fsc_policy (ExtractedFSCPolicy): Extracted FSC policy to be evaluated. If None, the policy is created from the agent.
        """
        
        evaluation_result = EvaluationResults()
        if extracted_fsc_policy is None:
            extracted_fsc_policy = NaiveFSCPolicyExtraction(agent.wrapper, agent.environment, agent.tf_environment, agent.args, model = model)
        driver, buffer = get_new_vectorized_evaluation_driver(
            agent.tf_environment, agent.environment, custom_policy=extracted_fsc_policy, num_steps=agent.args.max_steps)
        agent.tf_environment.reset()
        driver.run()
        buffer.final_update_of_results(
            evaluation_result.update)
        log_evaluation_info(evaluation_result)
        set_fsc_values_to_evaluation_result(external_evaluation_result, evaluation_result)
        buffer.clear()

def evaluate_policy_in_model(policy : TFPolicy, args : ArgsEmulator, environment : EnvironmentWrapperVec, tf_environment, max_steps = None,
                             evaluation_result : EvaluationResults = None) -> EvaluationResults:
    """Evaluate the policy in the given environment and return the evaluation results."""
    if max_steps is None:
        max_steps = args.max_steps
    if evaluation_result is None:
        evaluation_result = EvaluationResults()
    driver, buffer = get_new_vectorized_evaluation_driver(
        tf_environment, environment, custom_policy=policy, num_steps=max_steps)
    environment.set_random_starts_simulation(False)
    tf_environment.reset()
    driver.run()
    buffer.final_update_of_results(evaluation_result.update)
    log_evaluation_info(evaluation_result)
    buffer.clear()
    return evaluation_result