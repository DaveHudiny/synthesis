from environment import tf_py_environment
from environment.environment_wrapper import Environment_Wrapper
import numpy as np


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