import collections

from stormpy.storage import storage

import tensorflow as tf
import numpy as np

class Belief_Updater:
    """Class for storing parts of StormPy model for computing belief with supplement belief update function"""
    def __init__(self, stormpy_model : storage.SparsePomdp, action_labels_at_observation : dict):
        self.pomdp_observations = stormpy_model.observations
        self.stormpy_model = stormpy_model
        self.action_labels_at_observation = action_labels_at_observation
        self.ndi = stormpy_model.nondeterministic_choice_indices.copy()
        self.nr_states = stormpy_model.nr_states
        self.nr_observations = stormpy_model.nr_observations
        self.nr_choices = stormpy_model.nr_choices
        self.pomdp_get_choice_index = stormpy_model.get_choice_index
        self.pomdp_transition_matrix = stormpy_model.transition_matrix
        self.init_observation_state_mapping()
        self.init_observation_action_lengths()
        self.init_choice_index_matrix(self.max_number_of_actions)
        self.init_transition_matrix()
        
        
    def init_observation_action_lengths(self):
        self.observation_action_lengths = np.zeros((self.nr_observations,), dtype=np.int32)
        for obs in range(len(self.action_labels_at_observation)):
            self.observation_action_lengths[obs] = len(self.action_labels_at_observation[obs])
        self.max_number_of_actions = np.max(self.observation_action_lengths)
        self.observation_action_lengths = tf.convert_to_tensor(self.observation_action_lengths, dtype=tf.int32)
        
    def init_observation_state_mapping(self):
        observation_state_mapping = np.zeros((self.nr_observations, self.nr_states))
        observation_mask = self.pomdp_observations == np.arange(self.nr_observations)[:, None]
        observation_state_mapping[observation_mask] = 1
        observation_state_mapping /= np.sum(observation_state_mapping, axis=1, keepdims=True)
        self.observation_state_mapping = tf.convert_to_tensor(observation_state_mapping, dtype=tf.float32)
        
    def init_choice_index_matrix(self, max_number_of_actions):
        choice_index_matrix = np.zeros((self.nr_states, max_number_of_actions), dtype=np.int32)
        for state in range(self.nr_states):
            for action in range(max_number_of_actions):
                choice = self.get_choice_index(state, action)
                if choice >= self.nr_choices:
                    continue
                choice_index_matrix[state, action] = self.get_choice_index(state, action)
        self.choice_index_matrix = tf.convert_to_tensor(choice_index_matrix, dtype=tf.int32)
    
    def init_transition_matrix(self):
        choice_transition_matrix = np.zeros((self.nr_choices, self.nr_states,), dtype=np.float32)
        for choice in range(self.nr_choices):
            for entry in self.pomdp_transition_matrix.get_row(choice):
                choice_transition_matrix[choice, entry.column] = entry.value()
        self.transition_matrix = tf.convert_to_tensor(choice_transition_matrix, dtype=tf.float32)
        
    def get_choice_index(self, state, action):
        choice = self.pomdp_get_choice_index(state, action)
        return choice
    
    def next_belief(self, belief, action_label, next_obs):
        any_belief_state = list(belief.keys())[0]
        obs = self.pomdp_observations[any_belief_state]
        action = self.action_labels_at_observation[obs].index(action_label)
        new_belief = collections.defaultdict(float)
        for state, state_prob in belief.items():
            choice = self.get_choice_index(state, action)
            for entry in self.transition_matrix.get_row(choice):
                next_state = entry.column
                if self.pomdp_observations[next_state] == next_obs:
                    new_belief[next_state] += state_prob * entry.value()
        prob_sum = sum(new_belief.values())
        new_belief = {state:prob/prob_sum for state,prob in new_belief.items()}
        return new_belief
    
    @tf.function
    def initial_belief(self, observations):
        if len(observations.shape) > 1: # Batched input
            indices = observations[:, 0]
            belief = tf.gather(self.observation_state_mapping, indices)
        else:
            indices = observations[0]
            belief = self.observation_state_mapping[indices]
        return belief    
    
    @tf.function
    def nullify_illegal_states(self, belief, next_obs):
        mask = tf.cast(self.observation_state_mapping[next_obs] > 0.0, tf.float32)
        belief = belief * mask
        return belief
    
    @tf.function
    def update_belief_old(self, belief, obs, next_obs): # Update single belief state
        new_belief = tf.zeros((self.nr_states,), dtype=tf.float32)
        for state in range(self.nr_states): # Iterate over states
            for action in range(self.observation_action_lengths[obs]): # Iterate over actions
                choice = self.choice_index_matrix[state, action]
                row = self.transition_matrix[choice]
                new_belief = tf.add(new_belief, row * belief[state])
        new_belief = self.nullify_illegal_states(new_belief, next_obs) # Zero out all states that does not emit next_obs
        new_belief = new_belief / tf.reduce_sum(new_belief) # Normalize
        return new_belief 
    
    def update_belief(self, belief, obs, next_obs):  # Update single belief state
        choices = self.choice_index_matrix[:, :self.observation_action_lengths[obs]]
        selected_rows = tf.gather(self.transition_matrix, choices)
        weighted_rows = selected_rows * tf.reshape(belief, (-1, 1, 1))
        new_belief = tf.reduce_sum(weighted_rows, axis=[0, 1])
        new_belief = self.nullify_illegal_states(new_belief, next_obs)  # Zero out all states that do not emit next_obs
        belief_sum = tf.reduce_sum(new_belief)
        new_belief = tf.cond(belief_sum > 0, lambda: new_belief / belief_sum, lambda: new_belief)  # Normalize only if sum > 0
        return new_belief
        
    @tf.function
    def compute_beliefs_for_consequent_steps(self, belief, observations, actions = None):
        beliefs = []
        observations = tf.squeeze(observations)
        sum = tf.reduce_sum(belief)
        if sum < 1e-6: # if belief is zero, create pure belief based on first observation
            belief = self.initial_belief(observations)
        if len(observations.shape) > 1:
            for i in range(len(observations)): # Iterate over batches
                sub_beliefs = []
                sub_belief = belief[i]
                sub_beliefs.append(sub_belief)
                for j in range(len(observations[i]) - 1): # Iterate over time steps
                    sub_belief = self.update_belief(sub_belief, observations[i, j], observations[i, j+1])
                    sub_beliefs.append(sub_belief)
                # belief = self.next_belief(belief, actions[i], observations[i])
                beliefs.append(sub_beliefs)
        else:
            beliefs.append(belief)
            for i in range(len(observations) - 1):
                belief = self.update_belief(belief, observations[i], observations[i+1])
                beliefs.append(belief)
        belief_tensor = tf.convert_to_tensor(beliefs)
        return belief_tensor