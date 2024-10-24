# Implementation of PPO agent with recurrent neural networks.
# Author: David Hudák
# Login: xhudak03
# File: recurrent_ppo_agent.py

from agents.father_agent import FatherAgent
from tools.encoding_methods import *

import tensorflow as tf
import tf_agents

from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_agent

from tf_agents.utils import common

from tf_agents.policies import py_tf_eager_policy


from environment.environment_wrapper import Environment_Wrapper

from tf_agents.trajectories import trajectory
from tf_agents.trajectories import Trajectory
from tf_agents.trajectories import policy_step


from agents.policies.stochastic_ppo_collector_policy import Stochastic_PPO_Collector_Policy
from agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper
from agents.policies.fsc_policy import FSC_Policy

from agents.networks.value_networks import create_recurrent_value_net_demasked
from agents.networks.actor_networks import create_recurrent_actor_net_demasked


import sys
sys.path.append("../")
from paynt.quotient.fsc import FSC



import logging

logger = logging.getLogger(__name__)


class Recurrent_PPO_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, 
                 args, load=False, agent_folder=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0)

        action_spec = tf_environment.action_spec()

        self.actor_net = create_recurrent_actor_net_demasked(
            tf_environment, action_spec)
        
        self.value_net = create_recurrent_value_net_demasked(
                tf_environment)
        
        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(observation=tf_environment.observation_spec()["observation"])

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=self.actor_net,
            value_net=self.value_net,
            num_epochs=3,
            train_step_counter=train_step_counter,
            greedy_eval=False,
            discount_factor=0.9,
            use_gae=True,
            lambda_value=0.5,
        )
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
        
    def init_fsc_policy_driver(self, tf_environment: tf_py_environment.TFPyEnvironment, fsc: FSC = None, soft_decision: bool = False, 
                               fsc_multiplier: float = 2.0):
        """Initializes the driver for the FSC policy. Used for hard and soft FSC advices."""
        parallel_policy = self.wrapper
        self.fsc_policy = FSC_Policy(tf_environment, fsc,
                                     observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.environment.action_keywords,
                                     info_spec=self.agent.collect_policy.info_spec,
                                     parallel_policy=parallel_policy, soft_decision=soft_decision,
                                     soft_decision_multiplier=fsc_multiplier)
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        observer = self.demasked_observer()
        self.driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
            tf_environment,
            eager,
            observers=[observer],
            num_episodes=1
        )
    
    def reset_weights(self):
        for layer in self.agent._actor_net.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                for w in layer.trainable_weights:
                    w.assign(tf.random.normal(w.shape, stddev=0.05))
            elif isinstance(layer, tf.keras.layers.Dense):
                for w in layer.trainable_weights:
                    w.assign(tf.random.normal(w.shape, stddev=0.05))
        for layer in self.agent._value_net.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                for w in layer.trainable_weights:
                    w.assign(tf.random.normal(w.shape, stddev=0.05))
            elif isinstance(layer, tf.keras.layers.Dense):
                for w in layer.trainable_weights:
                    w.assign(tf.random.normal(w.shape, stddev=0.05))
        self.agent._actor_net.built = False
        self.agent._value_net.built = False
        self.agent._actor_net.build(self.tf_environment.observation_spec())
        self.agent._value_net.build(self.tf_environment.observation_spec())

    #######################################################################
    # Legacy Code -- Mostly used for dynamic action space.               #
    #######################################################################
        
    def create_recurrent_actor_net(self, tf_environment: tf_py_environment.TFPyEnvironment, action_spec):
        preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
        layer_params = (50, 50)
        actor_net = tf_agents.networks.actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            tf_environment.observation_spec(),
            action_spec,
            preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=layer_params,
            output_fc_layer_params=None,
            lstm_size=(64,),
            conv_layer_params=None,
        )
        return actor_net
        
    def create_recurrent_value_net(self, tf_environment: tf_py_environment.TFPyEnvironment, action_spec):
        preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
        layer_params = (50, 50)
        value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
            tf_environment.observation_spec(),
            preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=layer_params,
            output_fc_layer_params=None,
            lstm_size=(64,),
            conv_layer_params=None
        )
        return value_net
