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
                 observation_and_action_constraint_splitter=None, nr_observations=1):
        
        self.qvalues_table = tf.constant(qvalues_table)
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self.nr_observations = nr_observations
        
        super(FSC_Critic, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(), # If we would like to use memory-based version of FSC estimation, we would need to specify the state specification here
            name=name)
        
    def qvalues_function(self, observations, step_type, network_state):
        if len(observations.shape) == 2: # Unbatchet observation
            observations = tf.expand_dims(observations, axis=0)
            
        # Conversion of observation to integer indices -- currently implemented as additional normalised feature
        indices = tf.zeros_like(observations[:, :, -1] * self.nr_observations, dtype=tf.int32)
        
        if self.observation_and_action_constraint_splitter is not None:
            observations, _ = self.observation_and_action_constraint_splitter(observations)
        if network_state == (): # unknown memory node, return average
            values_rows = tf.gather(self.qvalues_table, indices)
            values = tf.reduce_mean(values_rows, axis=-1)
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
