# Implementation of PPO with critic represetned by Q-values obtained from FSC.
# Author: David Hudák
# Login: xhudak03
# File: ppo_with_qvalues_fsc.py

from agents.father_agent import FatherAgent
from environment.environment_wrapper import Environment_Wrapper


from tf_agents.networks import network
from tf_agents.environments import tf_py_environment

import sys
sys.path.append("../")



class FSC_Critic(network.Network):
    def __init__(self, input_tensor_spec, name="FSC_QValue_Estimator", qvalues_function=None):
        super(FSC_Critic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        
        
    

class PPO_with_QValues_FSC(FatherAgent):
    def __init__(self, environment, tf_environment, args, load=False, agent_folder=None, qvalues_function=None):
        self.common_init(environment, tf_environment, args, load, agent_folder, )
        self.agent = None
        self.policy_state = None
        
        assert qvalues_function is not None # Q-values function must be provided
        self.qvalues_function = qvalues_function

    def get_evaluation_policy(self):
        return self.agent.collect_policy

    def get_initial_state(self, batch_size=None):
        return self.agent.collect_policy.get_initial_state(batch_size)

    def save_agent(self, best=False):
        self.agent.save()

    def load_agent(self, best=False):
        self.agent.load()
        
