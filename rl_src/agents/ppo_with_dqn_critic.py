# Implementation of PPO with critic represetned by Q-values obtained from FSC.
# Author: David Hudák
# Login: xhudak03
# File: ppo_with_qvalues_fsc.py

from agents.father_agent import FatherAgent
from environment.environment_wrapper import Environment_Wrapper

from agents.networks.value_networks import get_alternative_call_of_qnet, create_recurrent_value_net_demasked, Value_DQNet
from agents.networks.actor_networks import create_recurrent_actor_net_demasked
from agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper

from tools.encoding_methods import observation_and_action_constraint_splitter


from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.policies import py_tf_eager_policy

import tensorflow as tf
import tf_agents

import logging

logger = logging.getLogger(__name__)


class PPO_with_DQN_Critic(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args, load=False, agent_folder=None, dqn_agent : DqnAgent = None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        tf_environment = self.tf_environment
        self.agent = None
        self.policy_state = None

        assert dqn_agent != None

        self.actor_net = create_recurrent_actor_net_demasked(
            tf_environment, tf_environment.action_spec())
        self.dqn_critic_net = dqn_agent._q_network
        # self.dqn_critic_net.__call__ = get_alternative_call_of_qnet(self.critic_net) # TODO: Fix this
        
        # self.critic_net = create_recurrent_value_net_demasked(tf_environment)
        # self.critic_net.__call__ = get_alternative_call_of_qnet(self.dqn_critic_net)
        self.critic_net = Value_DQNet(self.dqn_critic_net)

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
            lambda_value=0.95,
            name='PPO_with_QValues_FSC',
            greedy_eval=False,
            discount_factor=0.99
        )
        self.agent.initialize()
        logging.info("Agent initialized")
        
        self.args.prefer_stochastic = True
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")
        self.init_collector_driver(self.tf_environment_train)
        self.wrapper = Policy_Mask_Wrapper(self.agent.policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                           is_greedy=False)
        # self.wrapper = self.agent.policy
        if load:
            self.load_agent()
            
    def init_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment):
        self.collect_policy_wrapper = Policy_Mask_Wrapper(
            self.agent.collect_policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec())
        # self.collect_policy_wrapper = self.agent.collect_policy
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.collect_policy_wrapper, use_tf_function=True, batch_time_steps=False)
        # eager = self.collect_policy_wrapper
        observer = self.demasked_observer()
        # observer = self.replay_buffer.add_batch
        self.driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[observer],
            num_steps=self.traj_num_steps)