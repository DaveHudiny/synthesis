import collections

from stormpy.storage import storage

import tensorflow as tf
import numpy as np

class Belief_Updater:
    """Class for storing parts of StormPy model for computing belief with supplement belief update function"""
    def __init__(self, stormpy_model : storage.SparsePomdp, action_labels_at_observation : dict):
        self.pomdp_observations = stormpy_model.observations
        self.action_labels_at_observation = action_labels_at_observation
        self.ndi = stormpy_model.nondeterministic_choice_indices.copy()
        self.nr_states = stormpy_model.nr_states
        self.nr_observations = stormpy_model.nr_observations
        self.pomdp_get_choice_index = stormpy_model.get_choice_index
        self.pomdp_transition_matrix = stormpy_model.transition_matrix
        self.init_observation_state_mapping()
        
    def init_observation_state_mapping(self):
        observation_state_mapping = np.zeros((self.nr_observations, self.nr_states))
        observation_mask = self.pomdp_observations == np.arange(self.nr_observations)[:, None]
        observation_state_mapping[observation_mask] = 1
        observation_state_mapping /= np.sum(observation_state_mapping, axis=1, keepdims=True)
        self.observation_state_mapping = tf.convert_to_tensor(observation_state_mapping, dtype=tf.float32)
        
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
            indices = observations[:][0]
            belief = tf.gather(self.observation_state_mapping, indices)
        else:
            indices = observations[0]
            belief = self.observation_state_mapping[indices]
        return belief    
        
    @tf.function
    def compute_beliefs_for_consequent_steps(self, belief, observations, actions = None):
        beliefs = []
        observations = tf.squeeze(observations)
        sum = tf.reduce_sum(belief)
        if sum < 1e-6: # if belief is zero, create pure belief based on first observation
            belief = self.initial_belief(observations)
        if len(observations.shape) > 1:
            for i in range(len(observations)):
                sub_beliefs = []
                for j in range(len(observations[i])):
                    # belief[i] = self.next_belief(belief, actions[i][j], observations[i][j])
                    sub_beliefs.append(belief[j])
                # belief = self.next_belief(belief, actions[i], observations[i])
                beliefs.append(sub_beliefs)
        else:
            for i in range(len(observations)):
                beliefs.append(belief)
        return tf.convert_to_tensor(beliefs)
        
    
    # @tf.function
    def next_belief_without_known_action(self, belief_dict, next_obs):
        belief = belief_dict["belief"]
        non_zero = tf.math.count_nonzero(belief)
        if non_zero == 0:
            for state in range(self.nr_states):
                squeezed = tf.squeeze(next_obs)
                for i in range(len(squeezed)):
                    if self.pomdp_observations[state] == squeezed[i]:
                        belief[state][i] = 1.0
            sum = tf.reduce_sum(belief)
            belief = belief / sum
            return belief_dict
        belief = belief_dict["belief"]
        flag = belief_dict["init_flag"]
        if flag:
            return belief_dict
        any_belief_state = list(belief.keys())[0]
        obs = self.pomdp_observations[any_belief_state]
        actions = self.action_labels_at_observation[obs]
        new_belief = collections.defaultdict(float)
        for state, state_prob in belief.items():
            for action in actions:
                choice = self.get_choice_index(state, action)
                for entry in self.transition_matrix.get_row(choice):
                    next_state = entry.column
                    if self.pomdp_observations[next_state] == next_obs:
                        new_belief[next_state] += state_prob * entry.value()
        prob_sum = sum(new_belief.values())
        new_belief = {state:prob/prob_sum for state,prob in new_belief.items()}
        return new_belief