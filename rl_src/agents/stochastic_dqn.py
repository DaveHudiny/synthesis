from agents.father_agent import FatherAgent

import tensorflow as tf
import tf_agents

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.networks import sequential, actor_distribution_network, actor_distribution_rnn_network, value_rnn_network, value_network
from tf_agents.networks import categorical_projection_network

from tf_agents.utils import common

from environment.environment_wrapper import Environment_Wrapper

import logging


class Stochastic_DQN(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0)

        preprocessing_layers = [tf.keras.layers.Dense(
            200, activation='relu') for _ in range(2)]
        q_values_layer = tf.keras.layers.Dense(
            units=len(environment.action_keywords),
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )
        categorical_projection_net = categorical_projection_network.CategoricalProjectionNetwork(
            tf_environment.action_spec())
        q_net = sequential.Sequential([categorical_projection_net])
        lstm1 = tf.keras.layers.LSTM(
            100, return_sequences=True, return_state=True, activation='tanh', dtype=tf.float32)
        q_net = sequential.Sequential(preprocessing_layers + [lstm1, q_net])

        logging.info("Creating agent")
        self.agent = dqn_agent.DqnAgent(
            tf_environment._time_step_spec,
            tf_environment._action_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
            epsilon_greedy=0.14
        )
        self.policy_state = self.agent.policy.get_initial_state(None)
        self.agent.initialize()
        print(q_net.summary())
        logging.info("Agent initialized")
        self.init_replay_buffer(tf_environment)
        logging.info("Replay buffer initialized")
        self.init_collector_driver(tf_environment)
        logging.info("Collector driver initialized")
        self.init_random_collector_driver(tf_environment)
        if load:
            self.load_agent()
