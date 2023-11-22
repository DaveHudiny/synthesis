# File: rl_interface.py
# Author: David Hudák
# Purpose: Interface between paynt and safe reinforcement learning algorithm

import sys

import numpy as np
sys.path.append('/home/david/Plocha/paynt/synthesis/rl_approach/safe_rl')

import pickle
import dill

import shield_v2
import rl_simulator_v2 as rl_simulator
import tensorflow as tf
import tf_agents as tfa
import stormpy

import logging
logger = logging.getLogger(__name__)




class RLInterface:
    def __init__(self):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self._model = None
        self.unique_observations = None

    def add_to_sys_argv_obstacle():
        sys.argv.append("-m")
        sys.argv.append("obstacle")
        sys.argv.append("-c")
        sys.argv.append("N=6")
    
    def add_to_sys_argv_refuel():
        sys.argv.append("-m")
        sys.argv.append("refuel")
        sys.argv.append("-c")
        sys.argv.append("N=6,ENERGY=10")
    
    def add_to_sys_argv_intercept():
        sys.argv.append("-m")
        sys.argv.append("intercept")
        sys.argv.append("-c")
        sys.argv.append("N=5,RADIUS=2")

    def add_to_sys_argv_evade():
        sys.argv.append("-m")
        sys.argv.append("evade")
        sys.argv.append("-c")
        sys.argv.append("N=6,RADIUS=2")

    def add_to_sys_argv_sac_strategy():
        sys.argv.append("--learning_method")
        sys.argv.append("SAC")

    def add_to_sys_argv_ddqn_strategy():
        sys.argv.append("--learning_method")
        sys.argv.append("DDQN")

    

    # def convert_state_storm_tensor(self, storm_state):
    #     discount = ts.tensor(storm_state.discount)
    #     tfa.trajectories.time_step.TimeStep(discount=discount, reward=reward, observation=observation, step_type=step_type)
    #     pass

    def convert_state_tensor_storm_action(self, tensor_state):
        pass

    def save_model(self, path_to_model):
        with open(path_to_model, "wb") as file:
            logger.debug(f"Saving model to {path_to_model}")
            dill.dump(self._model, file)
            logger.debug(f"Saved model to {path_to_model}")

    def load_model(self, path_to_model):
        with open(path_to_model, "rb") as file:
            logger.debug(f"Loading model from {path_to_model}")
            self._model = dill.load(file)
            logger.debug(f"Loaded model from {path_to_model}")

    def ask_model(self, storm_state):
        tensor_state = None
        self._model.agent.policy.action(tensor_state)

    def create_model(self):
        RLInterface.add_to_sys_argv_evade()
        RLInterface.add_to_sys_argv_sac_strategy()
        self._model, self.storm_model, self.tf_environment = shield_v2.improved_main()
        

    def train_model(self):
        pass

    def create_timestep(self, mask_size):
        discount = tf.constant([1.0], dtype=tf.float32)
        observation = {
            'mask': tf.constant([[False for i in range(mask_size)]], dtype=tf.bool),
            'obs': tf.constant([[0]], dtype=tf.int32)
        }
        reward = tf.constant([-1.0], dtype=tf.float32)
        step_type = tf.constant([1], dtype=tf.int32)

        time_step = tfa.trajectories.time_step.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation
        )
        return time_step
    
    def _policy_step_limits_from_set(self, policy_steps = None):
        min_limit = policy_steps[0].state
        max_limit = policy_steps[0].state
        for policy_step in policy_steps:
            min_limit = tf.minimum(min_limit, policy_step.state)
            max_limit = tf.maximum(max_limit, policy_step.state)
        return min_limit, max_limit
    
    def generate_policy_step_states_limits(self):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        actions = []
        for observation in self.unique_observations:
            time_step.observation['obs'] = tf.constant([[observation]], dtype=tf.int32)
            action = self._model.agent.policy.action(time_step)
            for i in range(10):
                action = self._model.agent.policy.action(time_step, action.state)
                actions.append(action)
            action = self._model.agent.policy.action(time_step)
            actions.append(action)
        limits_min, limits_max = self._policy_step_limits_from_set(actions)
        return limits_min, limits_max
    
    def monte_carlo_evaluation(self, time_step, nr_episodes, policy_limits_min=[0], policy_limits_max=[1]):
        unique_actions = np.empty(shape=(0,), dtype=np.int32)
        action = self._model.agent.policy.action(time_step)
        for _ in range(nr_episodes):
            for state in action.state:
                state = tf.random.uniform(shape=state.shape, 
                                                    minval=policy_limits_min, 
                                                    maxval=policy_limits_max, 
                                                    dtype=tf.float32)
            uniqor = self._model.agent.policy.action(time_step, action.state)
            unique_actions = np.unique(np.append(unique_actions, uniqor.action))
        return unique_actions
    
    def make_actions_printable(self, actions):
        action_numbers = ""
        action_keywords = ""
        for action in actions:
            action_numbers += f"{', ' if len(action_numbers) > 0 else ''}{action}"
            action_keywords += f"{', ' if len(action_keywords) > 0 else ''}{self.tf_environment.act_keywords[action]}"
        return action_numbers, action_keywords
    
    
    def evaluate_model(self, file, nr_episodes=10, method="monte_carlo"):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        self.unique_observations = np.unique(self.storm_model.observations)
        limits_min, limits_max = self.generate_policy_step_states_limits()
        observation_dict = {}
        for observation in self.unique_observations:
            time_step.observation['obs'] = tf.constant([[observation]], dtype=tf.int32)
            if method == "monte_carlo":
                action = self._model.agent.policy.action(time_step)
                actions = self.monte_carlo_evaluation(time_step, nr_episodes, limits_min, limits_max)
                acts, act_keys = self.make_actions_printable(actions)
                print(f"Observation: {observation}. Actions: {acts}. " +
                      f"Action Keywords: {act_keys}")
                file.write(f"Observation: {observation}. Action: {acts}. " +
                           f"Action Keyword: {act_keys}\n")
                observation_dict[observation] = actions
            else:
                print("No method specified.")
        with open("obs_actions.pickle", "wb") as file_pickle:
            pickle.dump(observation_dict, file_pickle)
    
    def evaluate_model_to_file(self, path_to_file):
        with open(path_to_file, "w") as file:
            self.evaluate_model(file)



if __name__ == "__main__":
    interface = RLInterface()
    interface.create_model()
    actions = interface.tf_environment._simulator.available_actions()
    interface.evaluate_model_to_file("obs_actions.txt")
    # for i in range(interface.storm_model.nr_observations):
    #     print(interface.storm_model.get_observation_labels(i))
    