# Implementation of recurrent DQN agent.
# Author: David Hudák
# Login: xhudak03
# File: recurrent_dqn_agent.py

import tensorflow as tf

from tf_agents.environments import tf_py_environment

from environment.environment_wrapper import Environment_Wrapper
from tools.encoding_methods import *

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential

from agents.father_agent import *

import logging


class Recurrent_DQN_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args, load=False, agent_folder=None, agent_settings: AgentSettings = None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        tf_environment = self.tf_environment
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        if agent_settings is None: # Default settings
            postprocessing_layers = [tf.keras.layers.Dense(
                100, activation='relu') for _ in range(2)]
            q_values_layer = tf.keras.layers.Dense(
                units=len(environment.action_keywords),
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2)
            )
            q_net = sequential.Sequential([q_values_layer])
            lstm1 = tf.keras.layers.LSTM(
                100, return_sequences=True, return_state=True, activation='relu', dtype=tf.float32)
            self.q_net = sequential.Sequential(
                [lstm1] + postprocessing_layers + [q_net])
        else: # Custom settings. Not used in this project currently.
            preprocessing_layers = [tf.keras.layers.Dense(
                num_units, activation='relu') for num_units in agent_settings.preprocessing_layers]
            lstm_units = [tf.keras.layers.LSTM(num_units, return_sequences=True, return_state=True,
                                               activation='tanh', dtype=tf.float32) for num_units in agent_settings.lstm_units]
            postprocessing_layers = [tf.keras.layers.Dense(
                num_units, activation='relu') for num_units in agent_settings.postprocessing_layers]
            q_values_layer = tf.keras.layers.Dense(
                units=len(environment.action_keywords),
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(tf_environment = self.tf_environment,
                    minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2)
            )
            self.q_net = sequential.Sequential(
                preprocessing_layers + lstm_units + postprocessing_layers + [q_values_layer])

        logging.info("Creating agent")
        self.agent = dqn_agent.DqnAgent(
            tf_environment._time_step_spec,
            tf_environment._action_spec,
            q_network=self.q_net,
            optimizer=optimizer,
            # td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
            epsilon_greedy=0.14,
            gradient_clipping=0.7,
            gamma=0.99
        )
        self.policy_state = self.agent.policy.get_initial_state(None)
        self.agent.initialize()
        print(self.q_net.summary())
        logging.info("Agent initialized")
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")
        self.init_collector_driver(self.tf_environment_train)
        logging.info("Collector driver initialized")
        self.init_random_collector_driver(self.tf_environment_train)
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
