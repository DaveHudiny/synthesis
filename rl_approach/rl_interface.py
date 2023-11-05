# File: rl_interface.py
# Author: David Hudák
# Purpose: Interface between paynt and safe reinforcement learning algorithm

import sys
sys.path.append('/home/david/Plocha/paynt/synthesis/rl_approach/safe_rl')

import pickle
import dill

import shield_v2
import rl_simulator_v2 as rl_simulator
import tensorflow as tf
import tf_agents as tfa

import logging
logger = logging.getLogger(__name__)

def add_to_sys_argv_obstacle():
    sys.argv.append("-m")
    sys.argv.append("obstacle")
    sys.argv.append("-c")
    sys.argv.append("N=4")

class StormState:
    def __init__(self, discount, reward, observation, step_type):
        self.discount = discount
        self.reward = reward
        self.observation = observation
        self.step_type = step_type


class RLInterface:
    def __init__(self):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self._model = None

    def convert_state_storm_tensor(self, storm_state):
        discount = ts.tensor(storm_state.discount)
        tfa.trajectories.time_step.TimeStep(discount=discount, reward=reward, observation=observation, step_type=step_type)
        pass

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
        add_to_sys_argv_obstacle()
        self._model, _ = shield_v2.improved_main()
        self.save_model(path_to_model="./models_rl/obstacle_model_ppo_500runs.pkl")
        

    def train_model(self):
        pass



if __name__ == "__main__":
    interface = RLInterface()
    interface.create_model()
    interface.load_model("./models_rl/obstacle_model_ppo_500runs.pkl")
    print(interface._model)