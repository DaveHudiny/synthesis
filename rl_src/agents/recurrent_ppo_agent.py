# Implementation of PPO agent with recurrent neural networks.
# Author: David Hudák
# Login: xhudak03
# File: recurrent_ppo_agent.py

import logging
from paynt.quotient.fsc import FSC
from agents.father_agent import FatherAgent
from tools.encoding_methods import *

import tensorflow as tf
import tf_agents

from environment import tf_py_environment
from tf_agents.agents.ppo import ppo_agent

from tf_agents.utils import common

from tf_agents.policies import py_tf_eager_policy


from environment.environment_wrapper import Environment_Wrapper

from agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper
from rl_src.agents.policies.parallel_fsc_policy import FSC_Policy

from agents.networks.value_networks import create_recurrent_value_net_demasked
from agents.networks.actor_networks import create_recurrent_actor_net_demasked

from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

import sys
sys.path.append("../")


logger = logging.getLogger(__name__)


class Recurrent_PPO_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args, load=False, agent_folder=None, actor_net: ActorDistributionRnnNetwork = None,
                 critic_net: ValueRnnNetwork = None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0)
        tf_environment = self.tf_environment
        action_spec = tf_environment.action_spec()
        if actor_net is not None:
            self.actor_net = actor_net
        else:
            self.actor_net = create_recurrent_actor_net_demasked(
                tf_environment, action_spec)
        if critic_net is not None:
            self.value_net = critic_net
        else:
            self.value_net = create_recurrent_value_net_demasked(
                tf_environment)

        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(
            observation=tf_environment.observation_spec()["observation"])

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=self.actor_net,
            value_net=self.value_net,
            num_epochs=2,
            train_step_counter=train_step_counter,
            greedy_eval=False,
            discount_factor=0.99,
            use_gae=True,
            lambda_value=0.95,
            gradient_clipping=0.5,
            policy_l2_reg=0.0001,
            value_function_l2_reg=0.0001,
            value_pred_loss_coef=0.45,
            entropy_regularization=0.1,
            normalize_rewards=True,
        )
        self.agent.initialize()
        logging.info("Agent initialized")
        self.init_replay_buffer()
        logging.info("Replay buffer initialized")

        self.init_collector_driver(self.tf_environment, demasked=True)
        self.wrapper = Policy_Mask_Wrapper(self.agent.collect_policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                           is_greedy=False)
        if load:
            self.load_agent()
        self.init_vec_evaluation_driver(
            self.tf_environment, self.environment, num_steps=self.args.max_steps)

    def reset_weights(self):
        for net_type in [self.agent._value_net, self.agent._actor_net]:
            for layer in net_type.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    for w in layer.trainable_weights:
                        w.assign(tf.random.normal(w.shape, stddev=0.05))
                        # glorlot_uniform
                        w.assign(tf.random.normal(w.shape, stddev=0.05))
                elif isinstance(layer, tf.keras.layers.Dense):
                    for w in layer.trainable_weights:
                        w.assign(tf.random.normal(w.shape, stddev=0.05))
            net_type.built = False
            net_type.build(self.tf_environment.observation_spec())
