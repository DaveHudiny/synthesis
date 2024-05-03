from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_initializer import ArgsEmulator, Initializer
from rl_src.interpreters.tracing_interpret import TracingInterpret
from rl_src.agents.policies.fsc_policy import FSC_Policy


import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy

import tf_agents

from paynt.quotient.fsc import FSC

import logging

logger = logging.getLogger(__name__)

class Synthesizer_RL:
    def __init__(self, stormpy_model, args : ArgsEmulator, fsc_pre_init : bool = False, fsc_multiplier : float = 2.0):
        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(self.initializer.environment)
        logger.info("RL Environment initialized")
        self.initializer.initialize_agent(fsc_pre_init)
        self.interpret = TracingInterpret(self.initializer.environment, self.initializer.tf_environment, self.initializer.args.encoding_method)
        self.fsc_multiplier = fsc_multiplier

    def train_agent(self, iterations : int):
        self.initializer.agent.train_agent_off_policy(iterations)
        self.initializer.agent.save_agent()

    def interpret_agent(self, best : bool = False, with_refusing : bool = False, greedy : bool = False):
        self.initializer.agent.load_agent(best)
        if greedy: # Works only with agents which use policy wrapping (in our case only PPO)
            self.initializer.agent.set_agent_training()
        else:
            self.initializer.agent.set_agent_evaluation()
        return self.interpret.get_dictionary(self.initializer.agent, with_refusing)
    
    def update_fsc_multiplier(self, multiplier : float):
        self.fsc_multiplier *= multiplier
    
    def train_agent_with_fsc_data(self, iterations : int, fsc : FSC, soft_decision : bool = False):
        self.initializer.agent.load_agent()
        self.initializer.agent.init_fsc_policy_driver(self.initializer.tf_environment, fsc, soft_decision, self.fsc_multiplier)
        self.initializer.agent.train_agent_off_policy(iterations)


    def train_agent_combined_with_fsc(self, iterations : int, fsc : FSC):
        assert self.initializer.args.learning_method == "PPO", "FSC policy can be created only for PPO agent"
        self.initializer.agent.wrapper._set_fsc_oracle(fsc, self.initializer.environment.action_keywords)
        self.initializer.agent.init_collector_driver(self.initializer.tf_environment)
        self.initializer.agent.train_agent_off_policy(iterations)


