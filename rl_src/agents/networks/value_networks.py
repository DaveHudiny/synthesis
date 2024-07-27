# Description: Network creators for critic-based agents.
# Author: David Hudák

from tf_agents.networks import network
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment

import numpy as np

def create_recurrent_value_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment):
    preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
    layer_params = (64, 64)
    value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
        tf_environment.observation_spec()["observation"],
        preprocessing_layers=preprocessing_layer,
        input_fc_layer_params=layer_params,
        output_fc_layer_params=None,
        lstm_size=(64,),
        conv_layer_params=None
    )
    return value_net

class FSC_Critic(network.Network):
    def __init__(self, input_tensor_spec, name="FSC_QValue_Estimator", qvalues_table=None, 
                 observation_and_action_constraint_splitter=None, nr_observations=1,
                 reward_multiplier = 1.0):
        
        # Original qvalues_table is a list of lists of floats with None values for unreachable states
        qvalues_table = self.__make_qvalues_table_tensorable(qvalues_table)
        # reward_multiplier is only used to change the sign of expected rewards
        # If we want to minimize the number (e.g. steps), we use negative multiplier
        # If we want to maximize the number of collected rewards, we use positive multiplier
        self.qvalues_table = tf.constant(qvalues_table) * reward_multiplier
        
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self.nr_observations = nr_observations
        
        super(FSC_Critic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(), # If we would like to use memory-based version of FSC estimation, we should specify the state specification here
            name=name)
        
    def __make_qvalues_table_tensorable(self, qvalues_table):
        nr_states = len(qvalues_table)
        for state in range(nr_states):
            memory_size = len(qvalues_table[state])
            for memory in range(memory_size):
                if qvalues_table[state][memory] == None:
                    not_none_values = [qvalues_table[state][i] for i in range(memory_size) if qvalues_table[state][i] is not None]
                    if len(not_none_values) == 0:
                        qvalues_table[state][memory] = 0.0
                    else:
                        qvalues_table[state][memory] = np.min(not_none_values)
        return qvalues_table
        
    def qvalues_function(self, observations, step_type, network_state):
        if len(observations.shape) == 2: # Unbatchet observation
            observations = tf.expand_dims(observations, axis=0)
            
        # Conversion of observation to integer indices -- currently implemented as additional normalised feature
        indices = tf.zeros_like(observations[:, :, -1] * self.nr_observations, dtype=tf.int32)
        
        if self.observation_and_action_constraint_splitter is not None:
            observations, _ = self.observation_and_action_constraint_splitter(observations)
        if network_state == (): # unknown memory node, return average
            values_rows = tf.gather(self.qvalues_table, indices)
            values = tf.reduce_max(values_rows, axis=-1)
            network_state = ()
        else:
            values = self.qvalues_table[observations][network_state]
            network_state = () # TODO: Implement memory-based version
        return values, network_state
        
    def call(self, observations, step_type, network_state, training=False):
        values, network_state = self.qvalues_function(observations, step_type, network_state)
        if step_type.shape == (1,):
            values = tf.constant(values, shape=(1, 1))
        return values, network_state
