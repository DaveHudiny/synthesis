# This file contains the Synthesizer_RL class, which creates interface between RL and PAYNT.
# Author: David Hudák
# Login: xhudak03
# File: synthesizer_rl.py

from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_main import ArgsEmulator, Initializer
from rl_src.interpreters.tracing_interpret import TracingInterpret
from rl_src.agents.policies.fsc_policy import FSC_Policy


import tensorflow as tf

from tf_agents.environments import tf_py_environment

from paynt.quotient.fsc import FSC

import logging

logger = logging.getLogger(__name__)

class Synthesizer_RL:
    """Class for the interface between RL and PAYNT.
    """
    def __init__(self, stormpy_model, args : ArgsEmulator, fsc_pre_init : bool = False, 
                 initial_fsc_multiplier : float = 1.0):
        """Initialization of the interface.
        Args:
            stormpy_model: Model of the environment.
            args (ArgsEmulator): Arguments for the initialization.
            fsc_pre_init (bool, optional): Whether to initialize FSC policy. Defaults to False.
            initial_fsc_multiplier (float, optional): Initial soft FSC multiplier. Defaults to 1.0.
        """
        
        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(self.initializer.environment)
        logger.info("RL Environment initialized")
        self.initializer.initialize_agent(fsc_pre_init)
        self.interpret = TracingInterpret(self.initializer.environment, self.initializer.tf_environment, self.initializer.args.encoding_method)
        self.fsc_multiplier = initial_fsc_multiplier

    def train_agent(self, iterations : int):
        """Train the agent.
        Args:
            iterations (int): Number of iterations.
        """
        self.initializer.agent.train_agent_off_policy(iterations)
        self.initializer.agent.save_agent()

    def interpret_agent(self, best : bool = False, with_refusing : bool = False, greedy : bool = False):
        """Interpret the agent.
        Args:
            best (bool, optional): Whether to use the best, or the last trained agent. Defaults to False.
            with_refusing (bool, optional): Whether to use refusing. Defaults to False.
            greedy (bool, optional): Whether to use greedy policy evaluation (in case of PPO). Defaults to False.
        Returns:
            dict: Dictionary of the interpretation.
        """
        self.initializer.agent.load_agent(best)
        if greedy: # Works only with agents which use policy wrapping (in our case only PPO)
            self.initializer.agent.set_agent_stochastic()
        else:
            self.initializer.agent.set_agent_greedy()
        return self.interpret.get_dictionary(self.initializer.agent, with_refusing)
    
    def update_fsc_multiplier(self, multiplier : float):
        """Multiply the FSC multiplier.
        Args:
            multiplier (float): Multiplier for multiplication.
        """
        self.fsc_multiplier *= multiplier
    
    def train_agent_with_fsc_data(self, iterations : int, fsc : FSC, soft_decision : bool = False):
        """Train the agent with FSC data.
        Args:
            iterations (int): Number of iterations.
            fsc (FSC): FSC data.
            soft_decision (bool, optional): Whether to use soft decision. Defaults to False.
        """
        self.initializer.agent.load_agent()
        self.initializer.agent.init_fsc_policy_driver(self.initializer.tf_environment, fsc, soft_decision, self.fsc_multiplier)
        self.initializer.agent.train_agent_off_policy(iterations)


    def train_agent_combined_with_fsc(self, iterations : int, fsc : FSC):
        """Train the agent combined with FSC.
        Args:
            iterations (int): Number of iterations.
            fsc (FSC): FSC data.
        """
        assert self.initializer.args.learning_method == "PPO", "FSC policy can be created only for PPO agent"
        self.initializer.agent.wrapper._set_fsc_oracle(fsc, self.initializer.environment.action_keywords)
        self.initializer.agent.init_collector_driver(self.initializer.tf_environment)
        self.initializer.agent.train_agent_off_policy(iterations)


