# Description: This file contains the function to create the actor network for the recurrent agent.
# Author: David Hudák

from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.environments import tf_py_environment

import tensorflow as tf

def create_recurrent_actor_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment, action_spec):
    preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
    layer_params = (64, 64)
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_environment.observation_spec()["observation"],
        action_spec,
        preprocessing_layers=preprocessing_layer,
        input_fc_layer_params=layer_params,
        output_fc_layer_params=None,
        lstm_size=(64,),
        conv_layer_params=None
    )
    return actor_net
