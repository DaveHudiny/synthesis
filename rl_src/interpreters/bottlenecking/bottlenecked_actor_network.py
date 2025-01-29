import tensorflow as tf
import tf_agents.networks as Network

import gin

from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from interpreters.bottlenecking.bottleneck_autoencoder import Autoencoder


@gin.configurable
class BottleneckedActor(Network.network.DistributionNetwork):
    def __init__(self, original_actor_rnn_network : ActorDistributionRnnNetwork, bottleneck_autoencoder : Autoencoder,
                 name='BottleneckedActor'):
        super(BottleneckedActor, self).__init__(
            input_tensor_spec=original_actor_rnn_network.input_tensor_spec,
            state_spec=original_actor_rnn_network._lstm_encoder.state_spec,
            output_spec=original_actor_rnn_network.output_spec,
            name=name
        )
        self.original_actor_rnn_network = original_actor_rnn_network
        self.bottleneck_autoencoder = bottleneck_autoencoder

        self._lstm_encoder = self.original_actor_rnn_network._lstm_encoder
        self._projection_networks = self.original_actor_rnn_network._projection_networks
        self._output_tensor_spec = self.original_actor_rnn_network.output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec
    
    def get_initial_state(self, batch_size=None):
        initial_state = self.original_actor_rnn_network.get_initial_state(batch_size)
        substates = tf.concat(initial_state, axis=-1)
        substates = self.bottleneck_autoencoder(substates)
        new_state = tf.split(substates, num_or_size_splits=2, axis=-1)
        initial_state = new_state
        return initial_state
    
    def call(self, observation, step_type, network_state=(), training=False):
        output_actions, network_state = self.original_actor_rnn_network(
            observation,
            step_type=step_type,
            network_state=network_state,
            training=training
            )
        if isinstance(network_state, tuple):
            keys = list(network_state.keys())
            substates = tf.concat(network_state[keys[0]], axis=-1)
            substates = self.bottleneck_autoencoder(substates)
            new_state = tf.nest.map_structure(
                lambda x: tf.split(x, num_or_size_splits=2, axis=-1),
                network_state[keys[0]],
            )
            network_state[keys[0]] = new_state
        else:
            substates = tf.concat(network_state, axis=-1)
            substates = self.bottleneck_autoencoder(substates)
            new_state = tf.split(substates, num_or_size_splits=2, axis=-1)
            network_state = new_state
        return output_actions, network_state

