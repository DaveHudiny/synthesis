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
    def __init__(self, stormpy_model, args : ArgsEmulator):
        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(self.initializer.environment)
        logger.info("RL Environment initialized")
        self.initializer.initialize_agent()
        self.interpret = TracingInterpret(self.initializer.environment, self.initializer.tf_environment, self.initializer.args.encoding_method)

    def create_fsc_policy(self, fsc : FSC):
        assert self.initializer.args.learning_method == "PPO", "FSC policy can be created only for PPO agent"
        fsc_policy = FSC_Policy(self.initializer.tf_environment, fsc,
                                     observation_and_action_constraint_splitter=self.initializer.agent.observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.initializer.environment.action_keywords,
                                     info_spec=self.initializer.agent.agent.policy.info_spec)
        return fsc_policy
    
    def train_agent(self, iterations : int):
        self.initializer.agent.train_agent_off_policy(iterations)

    def interpret_agent(self, best : bool = False, with_refusing : bool = False, greedy : bool = False):
        self.initializer.agent.load_agent(best)
        if greedy: # Works only with agents which use policy wrapping (in our case only PPO)
            self.initializer.agent.set_agent_training()
        else:
            self.initializer.agent.set_agent_evaluation()
        return self.interpret.get_dictionary(self.initializer.agent, with_refusing)
    
    def train_agent_with_fsc_data(self, iterations : int, fsc : FSC):
        self.initializer.agent.init_fsc_policy_driver(self.initializer.tf_environment, fsc)
        self.initializer.agent.train_agent_off_policy(iterations)

    def train_agent_combined_with_fsc(self, iterations : int, fsc : FSC):
        fsc_policy = self.create_fsc_policy(fsc)
        self.initializer.agent.wrapper._set_fsc_oracle(fsc_policy)
        self.initializer.agent.init_collector_driver(self.initializer.tf_environment)
        self.initializer.agent.train_agent_off_policy(iterations)


