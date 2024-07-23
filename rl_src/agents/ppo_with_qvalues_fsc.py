# Implementation of PPO with critic represetned by Q-values obtained from FSC.
# Author: David Hudák
# Login: xhudak03
# File: ppo_with_qvalues_fsc.py

from agents.father_agent import FatherAgent
from environment.environment_wrapper import Environment_Wrapper

from agents.networks.value_networks import FSC_Critic
from agents.networks.actor_networks import create_recurrent_actor_net_demasked
from agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper

from tools.encoding_methods import observation_and_action_constraint_splitter


from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import py_tf_eager_policy

import tensorflow as tf
import tf_agents

import logging

logger = logging.getLogger(__name__)


class PPO_with_QValues_FSC(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args, load=False, agent_folder=None, qvalues_table=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        self.agent = None
        self.policy_state = None
        print(qvalues_table)
        assert qvalues_table is not None  # Q-values function must be provided
        self.qvalues_function = qvalues_table

        self.actor_net = create_recurrent_actor_net_demasked(
            tf_environment, tf_environment.action_spec())
        self.critic_net = FSC_Critic(
            tf_environment.observation_spec()["observation"], 
            qvalues_table=self.qvalues_function, nr_observations=environment.nr_obs)
        
        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(observation=tf_environment.observation_spec()["observation"])
        
        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            tf_environment.action_spec(),
            actor_net=self.actor_net,
            value_net=self.critic_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            normalize_observations=False,
            normalize_rewards=False,
            use_gae=True,
            num_epochs=2,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=tf.Variable(0),
            lambda_value=0.5,
            name='PPO_with_QValues_FSC',
            greedy_eval=False
        )
        self.agent.initialize()
        
        self.agent.initialize()
        logging.info("Agent initialized")
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")
        self.init_collector_driver(tf_environment)
        self.wrapper = Policy_Mask_Wrapper(self.agent.policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                           is_greedy=False)
        if load:
            self.load_agent()
            
    def init_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment):
        self.collect_policy_wrapper = Policy_Mask_Wrapper(
            self.agent.collect_policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec())
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.collect_policy_wrapper, use_tf_function=True, batch_time_steps=False)
        observer = self.demasked_observer()
        self.driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[observer],
            num_steps=self.traj_num_steps)