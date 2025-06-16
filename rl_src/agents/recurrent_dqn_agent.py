# Implementation of recurrent DQN agent.
# Author: David Hudák
# Login: xhudak03
# File: recurrent_dqn_agent.py

import tensorflow as tf

from environment import tf_py_environment

from environment.environment_wrapper import Environment_Wrapper
from tools.encoding_methods import *

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential

from agents.father_agent import *
from tools.args_emulator import *

import logging


class Recurrent_DQN_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args: ArgsEmulator, load=False, agent_folder=None, agent_settings: AgentSettings = None,
                 single_value_qnet: bool = False):
        single_value_qnet = False
        self.common_init(environment, tf_environment, args, load, agent_folder)
        tf_environment = self.tf_environment
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        self.q_net = tf_agents.networks.q_rnn_network.QRnnNetwork(
            input_tensor_spec=tf_environment.observation_spec()["observation"],
            action_spec=tf_environment.action_spec(),
                lstm_size=(64, 64))

        logging.info("Creating agent")
        self.agent = dqn_agent.DqnAgent(
            tf_environment._time_step_spec,
            # tf_environment._action_spec,
            tf_environment.action_spec(),
            q_network=self.q_net,
            optimizer=optimizer,
            # td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
            epsilon_greedy=0.1,
            # gradient_clipping=0.7,
            gamma=0.99
        )
        self.policy_state = self.agent.policy.get_initial_state(None)
        self.agent.initialize()
        print(self.q_net.summary())
        logging.info("Agent initialized")
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")

        if single_value_qnet:
            alternative_observer = self.get_action_handicapped_observer()
            self.init_collector_driver(
                self.tf_environment, alternative_observer)
        else:
            self.init_collector_driver(self.tf_environment)

        logging.info("Collector driver initialized")
        self.init_random_collector_driver(self.tf_environment)
        if load:
            self.load_agent()

    def reset_weights(self):
        """Reset weights of the agent's Q-network."""
        for layer in self.agent._q_network.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                # For LSTM layers, reset both kernel and recurrent kernel weights
                for weight in layer.weights:
                    if 'kernel' in weight.name or 'recurrent_kernel' in weight.name:
                        weight.assign(tf.keras.initializers.RandomUniform(
                            minval=-0.03, maxval=0.03)(weight.shape))
            else:
                # For other layers, reset kernel and bias weights
                for weight in layer.weights:
                    if 'kernel' in weight.name or 'bias' in weight.name:
                        weight.assign(tf.keras.initializers.RandomUniform(
                            minval=-0.3, maxval=0.3)(weight.shape))
