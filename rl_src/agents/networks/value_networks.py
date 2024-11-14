# Description: Network creators for critic-based agents.
# Author: David Hudák

from tf_agents.networks import network
import tensorflow as tf

import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.specs import BoundedArraySpec

from tools.belief_updater import Belief_Updater

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

def create_recurrent_value_net_demasked(tf_environment: tf_py_environment.TFPyEnvironment):
    preprocessing_layer = tf.keras.layers.Dense(64, activation='relu')
    layer_params = (64, 64)
    value_net = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
        tf_environment.observation_spec()["observation"],
        preprocessing_layers=preprocessing_layer,
        input_fc_layer_params=layer_params,
        lstm_size=(32,),
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
                 stormpy_model : SparsePomdp = None, periode_length : int = 0,
                 tf_environment : tf_py_environment.TFPyEnvironment = None):
        # Original qvalues_table is a list of lists of floats with None values for unreachable states
        qvalues_table = self.__make_qvalues_table_tensorable(qvalues_table)
        
        # reward_multiplier is only used to change the sign of expected rewards
        # If we want to minimize the number (e.g. steps), we use negative multiplier
        # If we want to maximize the number of collected rewards, we use positive multiplier
        self.qvalues_table = tf.constant(qvalues_table, dtype=tf.float32)  # * reward_multiplier

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
        self.current_step_index = (self.current_step_index + 1) % self.periode_length
        if self.current_step_index == 0:
            if self.current_mode == self.Periodic_Modes.COMBINED_VALUE:
                self.current_mode = self.Periodic_Modes.PURE_VALUE_NET
            else:
                self.current_mode = self.Periodic_Modes.COMBINED_VALUE

        
    def q_values_function_simplified(self, observations, step_type, network_state):
        if len(observations.shape) == 2:  # Unbatched observation
            observations = tf.expand_dims(observations, axis=0)
        clipped_observations = tf.clip_by_value(observations[:, :, -1], 0.0, 1.0)
        indices = tf.cast(
            tf.round(clipped_observations * self.nr_states), dtype=tf.int32)
        indices = tf.clip_by_value(indices, 0, self.nr_states - 1)
        if indices.shape == (1, 1): # Single observation
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
        values, network_state = super().call(observations, step_type, network_state, training)
        # print("Origo hodnoty:", values.numpy())
        if self.current_mode == self.Periodic_Modes.COMBINED_VALUE:
            values_fsc = self.q_values_function_simplified(observations, step_type, network_state)
            values = tf.maximum(values, values_fsc)
        if step_type.shape == (1,):
            values = tf.constant(values, shape=(1, 1))
        self.update_current_mode()
        return values, network_state
    

class Value_DQNet(network.Network):
    def __init__(self, q_net : network.Network, trainable = False):
        self._network_output_spec = tf_agents.specs.ArraySpec((1,), dtype=np.float32)
        super(Value_DQNet, self).__init__(
            input_tensor_spec=q_net.input_tensor_spec,
            state_spec=q_net.state_spec
        )
        self.q_net = q_net
        for layer in self.q_net.layers:
            layer.trainable = trainable
        self.get_initial_state = q_net.get_initial_state
    
    def call(self, observation, step_type = None, network_state=(), training=False):
        training = False
        values, network_state = self.q_net(inputs=observation, step_type=step_type, 
                                           network_state=network_state, training=training)
        value = tf.math.reduce_max(values, axis=-1, keepdims=True)
        return tf.squeeze(value, -1), network_state
    
    

def get_alternative_call_of_qnet(q_net : network.Network):
    func = q_net.__call__
    def call(observation, step_type = None, network_state=(), training=False):
        values, network_state = func(observation, step_type, network_state, training)
        value = tf.math.reduce_max(values, axis=-1, keepdims=True)
        return value, network_state
    return call



class FSC_Critic(network.Network):
    def __init__(self, input_tensor_spec, name="FSC_QValue_Estimator", qvalues_table=None,
                 observation_and_action_constraint_splitter: callable = None, nr_observations: int = 1,
                 reward_multiplier=1.0, stormpy_model: SparsePomdp = None,
                 action_labels_at_observation: dict = None):
        """Initializes the FSC critic pseudo-network.

        Args:

        input_tensor_spec: A nest of tf.TypeSpec representing the input observations.
        name: A string representing the name of the network.
        qvalues_table: A list of lists of floats representing the Q-values of the states and memories.
        observation_and_action_constraint_splitter: A callable that takes observations and returns (observations, mask).
        nr_observations: An integer representing the number of all possible observations.
        reward_multiplier: A float representing the multiplier of the rewards.
        stormpy_model: A SparsePOMDP representing the StormPy model.
        action_labels_at_observation: A dictionary representing the action labels at the observation.
        """

        # Original qvalues_table is a list of lists of floats with None values for unreachable states
        qvalues_table = self.__make_qvalues_table_tensorable(qvalues_table)
        
        # reward_multiplier is only used to change the sign of expected rewards
        # If we want to minimize the number (e.g. steps), we use negative multiplier
        # If we want to maximize the number of collected rewards, we use positive multiplier
        self.qvalues_table = tf.constant(qvalues_table, dtype=tf.float32)  # * reward_multiplier

        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self.nr_observations = nr_observations
        self.nr_states = stormpy_model.nr_states
        state_spec = ()

        super(FSC_Critic, self).__init__(
                input_tensor_spec=input_tensor_spec,
                state_spec=state_spec,
                name=name)

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
    
    @tf.function
    def get_initial_state(self, batch_size=None):
        return ()
    
    def unbatched_belief_computation(self, beliefs):
        qvalues_table = tf.expand_dims(self.qvalues_table, axis=0)
        beliefs = tf.expand_dims(beliefs, axis=0)
        beliefs = tf.transpose(beliefs, perm=[1, 2, 0])
        values = tf.multiply(qvalues_table, beliefs)
        values = tf.reduce_mean(values, axis=1)
        values = tf.reduce_max(values, axis=-1)
        values = tf.expand_dims(values, axis=0)
        return values
    
    def batched_belief_computation(self, beliefs): 
        # B - batch size, S - number of steps, N - number of states, M - number of memories
        qvalues_table = tf.expand_dims(self.qvalues_table, axis=0)  # (1, N, M)
        expanded_beliefs = tf.expand_dims(beliefs, axis=2)  # (B, S, 1, N)
        expanded_beliefs = tf.transpose(expanded_beliefs, perm=[0, 1, 3, 2])  # (B, S, N, 1)
        values = tf.multiply(qvalues_table, expanded_beliefs)  # (B, S, N, M)
        values = tf.reduce_mean(values, axis=-2)  # (B, S, M)
        values = tf.reduce_max(values, axis=-1)  # (B, S)
        return values
    
    def qvalues_function(self, observations, step_type, belief):
        # if self.observation_and_action_constraint_splitter is not None:
        #     observations, _ = self.observation_and_action_constraint_splitter(
        #         observations)
        if len(observations.shape) == 2:  # Unbatched observation
            observations = tf.expand_dims(observations, axis=0)

        # Conversion of observation to integer indices -- currently implemented as additional normalised feature
        indices = tf.zeros_like(
            observations[:, :, -1] * self.nr_observations, dtype=tf.int32)
        # if belief == ():  # unknown memory node, return max
        #     values_rows = tf.gather(self.qvalues_table, indices)
        #     values = tf.reduce_max(values_rows, axis=-1)
        #     belief = ()
        # else:
            # belief = self.belief_updater.next_belief_without_known_action(belief, indices)
        if indices.shape == (1, 1): # single observation
            values = tf.multiply(self.qvalues_table, tf.transpose(belief))
            values = tf.reduce_sum(values, axis=0)
            values = tf.reduce_max(values, axis=-1)
        else:
            beliefs = self.belief_updater.compute_beliefs_for_consequent_steps(belief, indices)
            if len(tf.squeeze(indices).shape) == 1:
                values = self.unbatched_belief_computation(beliefs)
                belief = beliefs[-1, :]
            else:
                values = self.batched_belief_computation(beliefs)
                belief = beliefs[:, -1, :]
        return values, belief

    def q_values_function_simplified(self, observations, step_type, network_state):
        if len(observations.shape) == 2:  # Unbatched observation
            observations = tf.expand_dims(observations, axis=0)
        clipped_observations = tf.clip_by_value(observations[:, :, -1], 0.0, 1.0)
        indices = tf.cast(
            tf.round(clipped_observations * self.nr_states), dtype=tf.int32)
        indices = tf.clip_by_value(indices, 0, self.nr_states - 1)
        if indices.shape == (1, 1): # Single observation
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
        values = self.q_values_function_simplified(observations, step_type, network_state)
        if step_type.shape == (1,):
            values = tf.constant(values, shape=(1, 1))

        return values, network_state
