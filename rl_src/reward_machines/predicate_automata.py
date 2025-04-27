import tensorflow as tf


class PredicateAutomata:
    def __init__(self, states : list[int], state_labels : list[str], transition_matrix : list[list[int]], predicates_set_labels : list[str], initial_state = 0,
                 observation_labels : list[str] = None, predicate_based_rewards = False):
        """
        Initialize the RewardAutomata object.

        Args:
            states (list[int]): List of states in the automata.
            state_labels (list[str]): List of labels for each state.
            transition_matrix (list[list[int]]): Transition matrix of the automata.
            predicates_set_labels (list[str]): List of labels for each predicate.
            initial_state (int): Initial state of the automata.
            observation_labels (list[str]): List of labels for each observation.
            predicate_based_rewards (bool): Whether the rewards are based on observing predicates or observing new states.
        """
        self.states = states
        self.state_labels = state_labels
        self.transition_matrix = tf.constant(transition_matrix)
        self.predicate_set_labels = predicates_set_labels
        self.num_states = len(states)
        self.num_actions = len(transition_matrix[0])
        assert self.num_actions == 2**len(predicates_set_labels), "Number of actions should be equal to 2^number of predicates."
        assert len(predicates_set_labels) > 0, "There should be at least a single observable predicate."
        self.initial_state = initial_state
        self.predicate_based_rewards = predicate_based_rewards
        self.observation_labels = observation_labels

        # Search predicates in the observation_labels
        self.boolean_predicate_labels = []
        self.integer_predicate_labels = []
        self.integer_predicate_values = []
        for predicate in predicates_set_labels:
            if "=" in predicate:
                predicate = predicate.split("=")
                self.integer_predicate_labels.append(predicate[0])
                self.integer_predicate_values.append(int(predicate[1]))
            else:
                self.boolean_predicate_labels.append(predicate)
        if observation_labels is not None:
            self.boolean_predicate_indices = [label_i[0] for label_i in enumerate(observation_labels) if observation_labels[label_i[0]] in self.boolean_predicate_labels]
            self.integer_predicate_indices = [label_i[0] for label_i in enumerate(observation_labels) if observation_labels[label_i[0]] in self.integer_predicate_labels]


    @tf.function
    def _convert_binary_predicates_to_action_number(self, predicates : tf.Tensor) -> tf.Tensor:
        """
        Convert predicates to binary representation.

        Args:
            predicates (tf.Tensor): Batch of lists of boolean predicates (shape == [batch_size, len(self.precidate_set_labels)]).

        Returns:
            tf.Tensor: Int representation of the predicates (shape == [batch_size, 1]).
        """
        # Convert boolean predicates to binary representation

        mask_int = tf.cast(predicates, tf.int32)
        mask_int = tf.reshape(mask_int, (tf.shape(mask_int)[0], -1))
        mask_int = tf.reduce_sum(mask_int * (2**tf.range(tf.shape(mask_int)[1], dtype=tf.int32)), axis=1)
        mask_int = tf.reshape(mask_int, (-1, 1))
        return mask_int

    def evaluate_observation(self, observation : tf.Tensor) -> list[bool]:
        """Returns the evaluated predicates for the given observations.
        Args:
            observation (tf.Tensor): Observation to evaluate (shape == [batch_size, len(self.observation_labels)]).

        Returns: 
            tf.Tensor: List of boolean evaluations of each predicate (shape == [batch_size, len(self.predicate_set_labels)]).
        """
        # Evaluate the boolean predicates (vectorized)
        boolean_predicates = tf.gather(observation, self.boolean_predicate_indices, axis=-1)
        boolean_predicates = tf.cast(boolean_predicates, tf.bool)
        # Evaluate the integer predicates (vectorized)
        if len(self.integer_predicate_indices) > 0:  
            integer_predicates = tf.gather(observation, self.integer_predicate_indices, axis=-1)
            integer_predicates = tf.equal(integer_predicates, self.integer_predicate_values[-1])
            integer_predicates = tf.cast(integer_predicates, tf.bool)
        # Concatenate both predicates
            predicates = tf.concat([boolean_predicates, integer_predicates], axis=-1)
        else:
            predicates = boolean_predicates
        predicates = tf.cast(predicates, tf.bool)
        # Convert to binary representation
        predicates = tf.reshape(predicates, (tf.shape(predicates)[0], -1))
        return predicates

    @tf.function
    def step(self, current_state : tf.Tensor, observation : tf.Tensor) -> tf.Tensor:
        """
        Perform a step in the automata given the current state and observation.
        Args:
            current_state (tf.Tensor): Current state of the automata (shape == [batch_size, 1]).
            observation (tf.Tensor): Observation from the environment (shape == [batch_size, len(self.observation_labels)]).
        Returns:
            tf.Tensor: Next state of the automata (shape == [batch_size, 1]).
        """

        predicates = self.evaluate_observation(observation)
        # Convert predicates to binary representation
        action = self._convert_binary_predicates_to_action_number(predicates)
        # next_state = self.transition_matrix[current_state][action] batch-less version
        indices = tf.stack([current_state, action], axis=-1)
        next_state = tf.gather_nd(self.transition_matrix, indices)
        next_state = tf.cast(next_state, tf.int32)
        predicates = tf.cast(predicates, tf.bool)
        return next_state, predicates
    
    def get_reward_state_spec_len(self, predicate_based = False):
        """
        Get the state specification of the regarding visited states.

        Returns:
            list: List of state specifications.
        """
        if predicate_based:
            return len(self.predicate_set_labels)
        else:
            return len(self.states)
        
    def get_initial_state(self, batch_size = None):
        """
        Get the initial state of the automata.

        Returns:
            int: Initial state of the automata.
        """
        if batch_size is not None:
            return tf.fill((batch_size, 1), self.initial_state)
        else:
            return self.iniptial_state
        
    def get_initial_visited_states(self, batch_size = None):
        """
        Get the initial visited states of the automata.

        Returns:
            list: List of initial visited states.
        """
        number_of_visited_states = self.get_reward_state_spec_len(predicate_based=batch_size)
        if batch_size is not None:
            return tf.fill((batch_size, number_of_visited_states), False)
        else:
            return [False] * number_of_visited_states

def create_dummy_predicate_automata(observation_labels : list[str] = None) -> PredicateAutomata:
    states = [0, 1, 2, 3, 4, 5]
    state_labels = ['u0', 'u1', 'u2', 'u3', 'u4', 'u5']
    # transition_matrix = [
    #     [1, 2, 1, 2],  # Transitions from state 0 given actions 0, 1, 2 and 3
    #     [0, 2, 0, 1],  # Transitions from state 1...
    #     [0, 2, 2, 2]   # Transitions from state 2...
    # ]
    transition_matrix = [[0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 2, 4, 1, 1, 1], [2, 2, 2, 2, 4, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4, 4, 4], [5, 5, 5, 2, 4, 5, 5, 5]]
    # predicate_labels = ['amdone', 'refuelAllowed', 'hascrash']
    # maybe the predicates are reversed
    predicate_labels = ["hascrash", "amdone", "refuelAllowed"]
    automata = PredicateAutomata(states = states, state_labels = state_labels, transition_matrix = transition_matrix, predicates_set_labels = predicate_labels,
                                 observation_labels=observation_labels, predicate_based_rewards=False, initial_state=0)
    return automata

if __name__ == "__main__":
    # Example usage of RewardAutomata

    automata = create_dummy_predicate_automata()
    current_state = [[0], [0]]
    next_state = automata.step(current_state, [[True, False], [False, True]])
    for _ in range(100): # Play random actions
        current_state = next_state
        next_state = automata.step(current_state, [[True, False], [False, True]])
        print("Current state:", current_state)
    # print("Next state:", next_state)  # Output: Next state: 1
    