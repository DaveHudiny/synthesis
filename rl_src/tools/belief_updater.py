import collections

from stormpy.storage import storage

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