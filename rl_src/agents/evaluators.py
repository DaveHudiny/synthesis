# Description: This file contains the function for evaluation of model.
# The function computes the average return of the policy over the given number of episodes.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: evaluators.py

import tensorflow as tf

from environment.environment_wrapper import Environment_Wrapper




def compute_average_return(policy, tf_environment, num_episodes=10, using_logits=False, environment : Environment_Wrapper = None):
    """Compute the average return of the policy over the given number of episodes.
    
    Args:
        policy: The policy to evaluate.
        tf_environment: The environment to evaluate the policy on.
        num_episodes: The number of episodes to run the evaluation for.
        using_logits: Whether to use logits for action selection.
        environment: The environment wrapper object. Used for computing exact return without virtual goal.
    """
    total_return = 0.0
    episode_return = 0.0
    policy_function = tf.function(policy.action)
    # policy_function = common.function(policy.action)
    policy_distribution = tf.function(policy.distribution)
    for _ in range(num_episodes):
        time_step = tf_environment._reset()
        policy_state = policy.get_initial_state(None)
        while not time_step.is_last():
            action_step = policy_function(
                time_step, policy_state=policy_state)
            if using_logits:
                logits = policy_distribution(
                    time_step, policy_state).action.parameters["logits"]
                action = {"action": action_step.action, "logits": logits}
            else:
                action = action_step.action
            policy_state = action_step.state
            time_step = tf_environment.step(action)
            total_return += time_step.reward
        if environment is not None:
            total_return -= environment.virtual_value
            episode_return += environment.virtual_value
        else:
            total_return -= time_step.reward
            episode_return += time_step.reward

    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    return avg_return.numpy()[0], avg_episode_return.numpy()
