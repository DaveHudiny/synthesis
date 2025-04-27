# Description: This file contains the function for evaluation of model.
# The function computes the average return of the policy over the given number of episodes.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: evaluators.py

import tensorflow as tf

from environment.environment_wrapper import Environment_Wrapper
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment import tf_py_environment

from tf_agents.policies import TFPolicy

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from tools.evaluators_non_vectorized import calculate_statistics, process_episode_results, run_single_episode
from tools.trajectory_buffer import TrajectoryBuffer
from tools.args_emulator import ArgsEmulator

import logging

from tools.evaluation_results_class import EvaluationResults, log_evaluation_info

logger = logging.getLogger(__name__)


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


def get_new_vectorized_evaluation_driver(tf_environment: tf_py_environment.TFPyEnvironment, environment: EnvironmentWrapperVec,
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


def evaluate_policy_in_model(policy: TFPolicy, args: ArgsEmulator = None,
                             environment: EnvironmentWrapperVec = None,
                             tf_environment=None, max_steps=None,
                             evaluation_result: EvaluationResults = None) -> EvaluationResults:
    """Evaluate the policy in the given environment and return the evaluation results."""
    if max_steps is None and args is not None:
        max_steps = args.max_steps
    elif max_steps is None:
        max_steps = 1000
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
