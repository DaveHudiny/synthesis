# Description: This file contains the functions for evaluation of model.
import tensorflow as tf
from tf_agents.policies import py_tf_eager_policy

from tf_agents.utils import common


def compute_average_return(policy, tf_environment, num_episodes=10, using_logits=False):
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
        total_return -= time_step.reward
        episode_return += time_step.reward

    avg_return = total_return / num_episodes
    avg_episode_return = episode_return / num_episodes
    return avg_return.numpy()[0], avg_episode_return.numpy()[0]
