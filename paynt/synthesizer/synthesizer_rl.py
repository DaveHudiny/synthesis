from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_initializer import ArgsEmulator, Initializer
from rl_src.interpreters.tracing_interpret import TracingInterpret

import tensorflow as tf

from tf_agents.environments import tf_py_environment

from paynt.quotient.fsc import FSC

import logging

logger = logging.getLogger(__name__)


class Synthesizer_RL:
    def __init__(self, stormpy_model, args : ArgsEmulator):
        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(self.initializer.environment)
        logger.info("RL Environment initialized")
        self.initializer.initialize_agent()
        self.interpret = TracingInterpret(self.initializer.environment, self.initializer.tf_environment, self.initializer.args.encoding_method)
    
    def train_agent(self, iterations : int):
        self.initializer.agent.train_agent(iterations)

    def interpret_agent(self, best : bool = False, with_refusing : bool = False, greedy : bool = True):
        self.initializer.agent.load_agent(best)
        if greedy: # Works only with agents which use policy wrapping (in our case only PPO)
            self.initializer.agent.set_agent_evaluation()
        else:
            self.initializer.agent.set_agent_evaluation(epsilon_greedy = True)
        return self.interpret.get_dictionary(self.initializer.agent, with_refusing)
    
    def train_agent_with_fsc(self, iterations : int, fsc : FSC):
        pass



