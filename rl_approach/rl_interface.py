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

    def add_to_sys_argv_obstacle():
        sys.argv.append("-m")
        sys.argv.append("obstacle")
        sys.argv.append("-c")
        sys.argv.append("N=6")
    
    def add_to_sys_argv_refuel():
        sys.argv.append("-m")
        sys.argv.append("refuel")
        sys.argv.append("-c")
        sys.argv.append("N=6,ENERGY=7")

    def add_to_sys_argv_sac_strategy():
        sys.argv.append("--learning_method")
        sys.argv.append("SAC")

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
        RLInterface.add_to_sys_argv_obstacle()
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
    
    def evaluate_model(self, file):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        unique_observations = np.unique(self.storm_model.observations)
        for observation in unique_observations:
            time_step.observation['obs'] = tf.constant([[observation]], dtype=tf.int32)
            action = self._model.agent.policy.action(time_step)
            print(f"Observation: {observation}. Action: {action.action.numpy()[0]}. " +
                  f"Action Keyword: {self.tf_environment.act_keywords[action.action.numpy()[0]]}")
            file.write(f"Observation: {observation}. Action: {action.action.numpy()[0]}. " +
                       f"Action Keyword: {self.tf_environment.act_keywords[action.action.numpy()[0]]}\n")
            print(action)
    
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
    