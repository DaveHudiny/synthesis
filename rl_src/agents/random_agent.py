from rl_src.agents.father_agent import FatherAgent

from tf_agents.policies.random_tf_policy import RandomTFPolicy

import logging

logger = logging.getLogger(__name__)

class RandomTFPAgent(FatherAgent):
    def __init__(self, environment, tf_environment, args, load=False, agent_folder=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        self.agent = RandomTFPolicy(tf_environment.time_step_spec(), tf_environment.action_spec(), 
                                    observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        self.policy_state = None

    def get_evaluated_policy(self):
        return self.agent
    
    def get_initial_state(self, batch_size=None):
        return ()
    
    def save_agent(self, best=False):
        logging.info("Random agent does not have any weights to save.")

    def load_agent(self, best=False):
        logging.info("Random agent does not have any weights to load.")