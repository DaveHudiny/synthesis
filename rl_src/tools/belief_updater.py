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
        self.pomdp_get_choice_index = stormpy_model.get_choice_index
        self.pomdp_transition_matrix = stormpy_model.transition_matrix
        
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
    
    def initial_belief(self, observations):
        first_observation = observations[0]
        belief = np.zeros(self.nr_states)
        for state in range(self.nr_states):
            if self.pomdp_observations[state] == first_observation:
                belief[state] = 1.0
        belief = belief / np.sum(belief)
        return belief    
        
    def compute_beliefs_for_consequent_steps(self, belief, observations, actions = None):
        beliefs = []
        observations = tf.squeeze(observations)
        non_zero = tf.math.count_nonzero(belief)
        if non_zero == 0:
            belief = self.initial_belief(observations)
        
        for i in range(len(observations)):
            beliefs.append(belief)
        return tf.convert_to_tensor(beliefs)
        
    
    # @tf.function
    def next_belief_without_known_action(self, belief_dict, next_obs):
        belief = belief_dict["belief"]
        non_zero = tf.math.count_nonzero(belief)
        
        print(non_zero)
        if non_zero == 0:
            print("Creating pure belief for given observation")
            for state in range(self.nr_states):
                print(self.pomdp_observations[state])
                print(next_obs)
                squeezed = tf.squeeze(next_obs)
                for i in range(len(squeezed)):
                    if self.pomdp_observations[state] == squeezed[i]:
                        print(belief)
                        belief[state][i] = 1.0
                        print("Prirazeno")
            print(belief["belief"])
            sum = tf.reduce_sum(belief)
            belief = belief / sum
            print(belief)
            return belief_dict
        belief = belief_dict["belief"]
        flag = belief_dict["init_flag"]
        if flag:
            return belief_dict
        print("Inside:", belief)
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