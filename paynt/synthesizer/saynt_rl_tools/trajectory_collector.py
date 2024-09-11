
import tensorflow as tf


from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import TFPolicy
from stormpy.simulator import SparseSimulator

from rl_src.environment.environment_wrapper import Environment_Wrapper

class Step:
    def __init__(self, state=None, observation=None, action=None, success=False):
        self.state = state
        self.observation = observation
        self.action = action
        self.success = success

def report_state_and_observation(simulator : SparseSimulator) -> tuple[int, int]:
    state = simulator._report_state()
    observation = simulator._report_observation()
    return state, observation

def collect_trajectories(num_of_episodes : int = 10, policy : TFPolicy = None, environment : Environment_Wrapper = None, tf_environment : TFPyEnvironment = None) -> list[list[Step]]:
        take_action = tf.function(policy.action)
        episodes = []
        for _ in range(num_of_episodes):
            episode = []
            time_step = tf_environment.reset()
            policy_state = policy.get_initial_state(None)
            state, observation = report_state_and_observation(environment.simulator)
            while not time_step.is_last():
                policy_step = take_action(time_step, policy_state)
                policy_state = policy_step.state
                action = int(policy_step.action.numpy()[0])
                episode_step = Step(state, observation, action)
                episode.append(episode_step)
                time_step = tf_environment.step(policy_step.action)
                state, observation = report_state_and_observation(environment.simulator)
            if environment.is_goal_state(environment.labels):
                success = True
            else:
                success = False
            episode_step = Step(state, observation, action=None, success=success)
            episode.append(episode_step)
            episodes.append(episode)

        return episodes