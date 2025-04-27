# Description: Network creators for critic-based agents.
# Author: David Hudák

from tf_agents.networks import network
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment

from stormpy.storage import SparsePomdp

import numpy as np

from enum import Enum


def create_recurrent_value_net_demasked_tuned(tf_environment: tf_py_environment.TFPyEnvironment):
    # preprocessing_layer = tf.keras.layers.Dense(32, activation='relu')
    input_layer_params = (32,)
    output_layer_params = (32,)
    value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
        tf_environment.observation_spec()["observation"],
        # preprocessing_layers=preprocessing_layer,
        input_fc_layer_params=input_layer_params,
        output_fc_layer_params=output_layer_params,
        lstm_size=(32,),
        conv_layer_params=None
    )
    return value_net


def create_recurrent_value_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment, rnn_less=False):
    preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
    layer_params = (64, )
    if rnn_less:
        lstm_size = None
        preprocessing_layers = [preprocessing_layer]
        value_net = tf_agents.networks.value_network.ValueNetwork(
            tf_environment.observation_spec()["observation"],
            # preprocessing_layers=preprocessing_layers
        )
    else:
        lstm_size = (32,)
        value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
            tf_environment.observation_spec()["observation"],
            # preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=layer_params,
            lstm_size=lstm_size,
            # output_fc_layer_params=(64,),
            conv_layer_params=None
        )
    return value_net


class Periodic_FSC_Neural_Critic(tf_agents.networks.value_rnn_network.ValueRnnNetwork):
    class Periodic_Modes(Enum):
        PURE_VALUE_NET = True
        COMBINED_VALUE = False

    def __init__(self, input_tensor_spec, name="Periodic_FSC_Neural_Critic", qvalues_table=None,
                 observation_and_action_constraint_splitter: callable = None, nr_observations: int = 1,
                 stormpy_model: SparsePomdp = None, periode_length: int = 0,
                 tf_environment: tf_py_environment.TFPyEnvironment = None):
        # Original qvalues_table is a list of lists of floats with None values for unreachable states
        qvalues_table = self.__make_qvalues_table_tensorable(qvalues_table)

        # reward_multiplier is only used to change the sign of expected rewards
        # If we want to minimize the number (e.g. steps), we use negative multiplier
        # If we want to maximize the number of collected rewards, we use positive multiplier
        self.qvalues_table = tf.constant(
            qvalues_table, dtype=tf.float32)  # * reward_multiplier

        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self.nr_observations = nr_observations
        self.nr_states = stormpy_model.nr_states
        # self.value_net = create_recurrent_value_net_demasked(tf_environment)
        preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
        layer_params = (64, 64)
        super(Periodic_FSC_Neural_Critic, self).__init__(
            tf_environment.observation_spec()["observation"],
            preprocessing_layers=preprocessing_layer,
            input_fc_layer_params=layer_params,
            output_fc_layer_params=None,
            lstm_size=(64,),
            conv_layer_params=None
        )

        self.periode_length = periode_length
        self.current_step_index = 0
        self.current_mode = Periodic_FSC_Neural_Critic.Periodic_Modes.COMBINED_VALUE

    def __make_qvalues_table_tensorable(self, qvalues_table):
        nr_states = len(qvalues_table)
        for state in range(nr_states):
            memory_size = len(qvalues_table[state])
            for memory in range(memory_size):
                if qvalues_table[state][memory] == None:
                    not_none_values = [qvalues_table[state][i] for i in range(
                        memory_size) if qvalues_table[state][i] is not None]
                    if len(not_none_values) == 0:
                        qvalues_table[state][memory] = 0.0
                    else:
                        qvalues_table[state][memory] = np.min(not_none_values)
        return qvalues_table

    def update_current_mode(self):
        self.current_step_index = (
            self.current_step_index + 1) % self.periode_length
        if self.current_step_index == 0:
            if self.current_mode == self.Periodic_Modes.COMBINED_VALUE:
                self.current_mode = self.Periodic_Modes.PURE_VALUE_NET
            else:
                self.current_mode = self.Periodic_Modes.COMBINED_VALUE

    def q_values_function_simplified(self, observations, step_type, network_state):
        if len(observations.shape) == 2:  # Unbatched observation
            observations = tf.expand_dims(observations, axis=0)
        clipped_observations = tf.clip_by_value(
            observations[:, :, -1], 0.0, 1.0)
        indices = tf.cast(
            tf.round(clipped_observations * self.nr_states), dtype=tf.int32)
        indices = tf.clip_by_value(indices, 0, self.nr_states - 1)
        if indices.shape == (1, 1):  # Single observation
            values = tf.gather(self.qvalues_table, indices)
            values = tf.reduce_max(values, axis=-1)
        else:
            q_values_table = self.qvalues_table
            qvalues = tf.gather(q_values_table, indices)
            values = tf.reduce_max(qvalues, axis=-1)
        return values

    def call(self, observations, step_type, network_state, training=False):
        # values, network_state = self.qvalues_function(
        #     observations, step_type, network_state)
        if self.current_mode == self.Periodic_Modes.COMBINED_VALUE:
            training = False
        values, network_state = super().call(
            observations, step_type, network_state, training)
        # print("Origo hodnoty:", values.numpy())
        if self.current_mode == self.Periodic_Modes.COMBINED_VALUE:
            values_fsc = self.q_values_function_simplified(
                observations, step_type, network_state)
            values = tf.maximum(values, values_fsc)
        if step_type.shape == (1,):
            values = tf.constant(values, shape=(1, 1))
        self.update_current_mode()
        return values, network_state


class Value_DQNet(network.Network):
    def __init__(self, q_net: network.Network, trainable=False):
        self._network_output_spec = tf_agents.specs.ArraySpec(
            (1,), dtype=np.float32)
        super(Value_DQNet, self).__init__(
            input_tensor_spec=q_net.input_tensor_spec,
            state_spec=q_net.state_spec
        )
        self.q_net = q_net
        for layer in self.q_net.layers:
            layer.trainable = trainable
        self.get_initial_state = q_net.get_initial_state

    def call(self, observation, step_type=None, network_state=(), training=False):
        training = False
        values, network_state = self.q_net(inputs=observation, step_type=step_type,
                                           network_state=network_state, training=training)
        value = tf.math.reduce_max(values, axis=-1, keepdims=True)
        return tf.squeeze(value, -1), network_state