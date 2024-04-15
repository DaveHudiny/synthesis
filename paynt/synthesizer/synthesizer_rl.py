from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_initializer import ArgsEmulator, Initializer

import tensorflow as tf

from tf_agents.environments import tf_py_environment

import logging

logger = logging.getLogger(__name__)


class Synthesizer_RL:
    def __init__(self, stormpy_model, args : ArgsEmulator):
        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(self.initializer.environment)
        logger.info("RL Environment initialized")
        self.initializer.initialize_agent()
    
    def train_agent(self, iterations : int):
        self.initializer.agent.train_agent(iterations)
