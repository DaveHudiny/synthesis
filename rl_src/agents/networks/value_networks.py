# Description: Network creators for critic-based agents.
# Author: David Hudák

from tf_agents.networks import network
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment

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
                 observation_and_action_constraint_splitter=None):
        
        self.qvalues_table = tf.constant(qvalues_table)
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        
        super(FSC_Critic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(), # If we would like to use memory-based version of FSC estimation, we would need to specify the state specification here
            name=name)
        
    def qvalues_function(self, observations, step_type, network_state):
        if self.observation_and_action_constraint_splitter is not None:
            observations, _ = self.observation_and_action_constraint_splitter(observations)
        if network_state == (): # unknown memory node, return average
            value = tf.reduce_mean(self.qvalues_table[observations], 1)
        else:
            value = self.qvalues_table[observations][network_state]
        return value
        
    def call(self, observations, step_type, network_state):
        value = self.qvalues_function(observations, step_type, network_state)
        tf.squeeze(value, axis=-1)
