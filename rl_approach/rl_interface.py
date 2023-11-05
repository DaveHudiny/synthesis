# File: rl_interface.py
# Author: David Hudák
# Purpose: Interface between paynt and safe reinforcement learning algorithm

import sys
sys.path.append('/home/david/Plocha/paynt/synthesis/rl_approach/safe_rl')

import pickle

import shield_v2
import rl_simulator_v2 as rl_simulator




import logging
logger = logging.getLogger(__name__)

def add_to_sys_argv_obstacle():
    sys.argv.append("-m")
    sys.argv.append("obstacle")
    sys.argv.append("-c")
    sys.argv.append("N=4")


class RLInterface:
    def __init__(self):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self._model = None

    def convert_state_storm_tensor():
        pass

    def convert_state_tensor_storm():
        pass

    def save_model(self, path_to_model):
        with open(path_to_model, "wb") as file:
            logger.debug(f"Saving model to {path_to_model}")
            pickle.dump(self._model, file)
            logger.debug(f"Saved model to {path_to_model}")

    def load_model(self, path_to_model):
        with open(path_to_model, "rb") as file:
            logger.debug(f"Loading model from {path_to_model}")
            self._model = pickle.load(file)
            logger.debug(f"Loaded model from {path_to_model}")

    def ask_model(self, storm_state):
        pass

    def create_model(self):
        add_to_sys_argv_obstacle()
        self._model = shield_v2.improved_main()
        self.save_model(path_to_model="./models/obstacle_model_ppo_500runs")
        

    def train_model(self):
        pass



if __name__ == "__main__":
    interface = RLInterface()
    interface.create_model()