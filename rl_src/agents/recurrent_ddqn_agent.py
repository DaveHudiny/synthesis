import tensorflow as tf

from tf_agents.environments import tf_py_environment

from environment.environment_wrapper import Environment_Wrapper
from agents.tools import *

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential

from agents.father_agent import *

import logging


class Recurrent_DDQN_agent(FatherAgent):
    def __init__(self, environment : Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
        # preprocessing_layer = tf.keras.layers.Dense(32, activation='relu')
        layer_params = (50, 50, )
        self.fc_layer_params = layer_params


        dense_layers = [tf.keras.layers.Dense(num_units, activation='relu') for num_units in self.fc_layer_params]

        q_values_layer = tf.keras.layers.Dense(
            units=len(environment.action_keywords),
            activation=None,
            # kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            # bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True, activation='relu', dtype=tf.float32)
        q_net = sequential.Sequential([lstm1, q_net])
        logging.info("Creating agent")
        self.agent = dqn_agent.DdqnAgent(
            tf_environment.time_step_spec(),
            tf_environment.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
            epsilon_greedy=0.1
        )
        
        self.policy_state = self.agent.policy.get_initial_state(None)
        self.agent.initialize()
        logging.info("Agent initialized")
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")
        self.init_collector_driver(tf_environment)
        logging.info("Collector driver initialized")
        if load:
            self.load_agent()
        self.init_random_collector_driver(tf_environment)


    def reset_weights(self):
        for layer in self.agent._q_network.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                # For LSTM layers, reset both kernel and recurrent kernel weights
                for weight in layer.weights:
                    if 'kernel' in weight.name or 'recurrent_kernel' in weight.name:
                        weight.assign(tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)(weight.shape))
            else:
                # For other layers, reset kernel and bias weights
                for weight in layer.weights:
                    if 'kernel' in weight.name or 'bias' in weight.name:
                        weight.assign(tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3)(weight.shape))