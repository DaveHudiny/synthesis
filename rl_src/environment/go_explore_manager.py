import numpy as np

from reward_machines.predicate_automata import PredicateAutomata, create_dummy_predicate_automata


class GoExploreManager:
    class Records:
        """
        Class to store the records of the Go-Explore manager.
        """

        def __init__(self, num_of_states: int = 1, buffer_size: int = 1000, original_initial_state=0):
            self.state_database = np.full(
                (num_of_states, buffer_size), original_initial_state, dtype=np.int32)
            self.current_buffer_heads = np.zeros(num_of_states, dtype=np.int32)
            self.number_of_state_visitations = np.zeros(
                num_of_states, dtype=np.int32)
            self.number_of_samplings_from_state = np.zeros(
                num_of_states, dtype=np.int32)

        def restart_info(self):
            """
            Restart the Go-Explore manager records sampling information.
            """
            self.number_of_samplings_from_state = np.zeros(
                self.number_of_samplings_from_state.shape, dtype=np.int32)
            self.number_of_state_visitations = np.zeros(
                self.number_of_state_visitations.shape, dtype=np.int32)

        def restart_database(self):
            """
            Restart the Go-Explore manager records database.
            """
            self.state_database = np.full(
                self.state_database.shape, self.state_database[0, 0], dtype=np.int32)
            self.current_buffer_heads = np.zeros(
                self.current_buffer_heads.shape, dtype=np.int32)
            self.number_of_state_visitations = np.zeros(
                self.number_of_state_visitations.shape, dtype=np.int32)
            self.number_of_samplings_from_state = np.zeros(
                self.number_of_samplings_from_state.shape, dtype=np.int32)

    def __init__(self, automata: PredicateAutomata = None, buffer_size: int = 1000, original_initial_state=0):
        if automata:
            self.automata = automata
            self.states = automata.get_states()
            self.state_labels = automata.get_state_labels()
            self.num_of_states = len(self.states)
        else:
            self.automata = None
            self.states = []
            self.state_labels = []

        self.original_initial_state = original_initial_state
        self.num_of_stored_states = 0
        self.buffer_size = buffer_size
        self.records = self.Records(num_of_states=len(
            self.states), buffer_size=buffer_size, original_initial_state=original_initial_state)

    def __circular_buffer_write(self, buffer: np.ndarray, buffer_index: int, current_buffer_head: int, data: np.ndarray):
        """
        Write to the circular buffer of the Go-Explore manager Records.
        """
        if len(data) > len(buffer[buffer_index]) - current_buffer_head:  # Buffer overflow, start overwriting from the beginning
            from_start_end = len(data) - (len(buffer[buffer_index]) - current_buffer_head)

            buffer[buffer_index][current_buffer_head:] = data[:-from_start_end]
            buffer[buffer_index][:from_start_end] = data[-from_start_end:]
        else:
            buffer[buffer_index][current_buffer_head: current_buffer_head +
                                 len(data)] = data

    def add_state(self, automata_states: np.ndarray, mdp_states: np.ndarray):
        """
        Add a state to the Go-Explore manager Records.
        """
        automata_state_counts = np.bincount(
            automata_states, minlength=self.num_of_states)
        for automata_state in np.arange(self.num_of_states):
            if automata_state_counts[automata_state] > 0:
                automata_state_indices = np.where(
                    automata_states == automata_state)
                # self.records.state_database[automata_state, self.records.current_buffer_heads[automata_state]:] = mdp_states[mdp_state_indices]
                self.__circular_buffer_write(self.records.state_database, automata_state,
                                             self.records.current_buffer_heads[automata_state], mdp_states[automata_state_indices])
                self.records.current_buffer_heads[automata_state] += automata_state_counts[automata_state]
                self.records.current_buffer_heads[automata_state] %= self.buffer_size
                self.records.number_of_state_visitations[automata_state] += automata_state_counts[automata_state]
        self.num_of_stored_states += len(automata_states) % (
            self.buffer_size * self.num_of_states)
        
    def add_state_vec_mine(self, automata_states: np.ndarray, mdp_states: np.ndarray):
        """
        Vectorized add state to the Go-Explore manager Records.
        """
        current_heads = self.records.current_buffer_heads
        num_bins = self.num_of_states
        matches = (automata_states == np.arange(num_bins)[:, None]).astype(int)
        cumsums_for_states = np.cumsum(matches, axis=1) - 1

        cumsums_for_states += current_heads[:, None]
        cumsums_for_states %= self.buffer_size
        indices_from_cumsums = cumsums_for_states[automata_states, np.arange(len(automata_states))]
        buffer_indices = np.column_stack((automata_states, indices_from_cumsums))
        self.records.state_database[buffer_indices[:, 0], buffer_indices[:, 1]] = mdp_states
        self.records.current_buffer_heads += np.bincount(automata_states, minlength=num_bins)
        self.records.current_buffer_heads %= self.buffer_size
        self.records.number_of_state_visitations += np.bincount(automata_states, minlength=num_bins)
        self.num_of_stored_states += len(automata_states) % (self.buffer_size * self.num_of_states)

    def sample_states(self, num_samples: int) -> np.ndarray[np.int32]:
        """
        Sample states for Go-Explore learning.
        """
        if self.num_of_stored_states == 0:
            new_states = np.full((num_samples, ), self.original_initial_state)
        else:  # Inverse frequency sampling
            number_of_visitations = self.records.number_of_state_visitations
            number_of_samplings_from_state = self.records.number_of_samplings_from_state
            epsilon = 1e-6
            inverse_visitations = 1 / \
                (number_of_visitations + epsilon + number_of_samplings_from_state)
            probs = inverse_visitations / np.sum(inverse_visitations)
            # where the number of visitations is 0, we set the probability to 0, since we don't know these states
            probs[number_of_visitations == 0] = 1e-6
            probs /= np.sum(probs) # Normalize the probabilities
            automata_states = np.random.choice(
                self.num_of_states, size=num_samples, p=probs)
            # Get random states from the circular buffer
            cols = np.random.randint(0, self.buffer_size, size=num_samples)
            rows = automata_states
            new_states = self.records.state_database[rows, cols]
            np.add.at(self.records.number_of_samplings_from_state, rows, 1)

        # Sample from bernoulli distribution to replace some of the sampled states with the original initial state
        bernoulli_probs = np.random.uniform(size=num_samples)
        bernoulli_mask = bernoulli_probs < 0.15
        new_states[bernoulli_mask] = self.original_initial_state
        return new_states
    
    def restart(self):
        """
        Restart the Go-Explore manager.
        """
        self.records.restart_info()
        self.records.restart_database()
        self.num_of_stored_states = 0

def test_go_explore_manager():
    """
    Test the Go-Explore manager.
    """
    dummy = create_dummy_predicate_automata(
        ["hascrash", "amdone", "refuelAllowed"]) # Nr of states is 6
    go_explore_manager = GoExploreManager(
        automata=dummy, buffer_size=3, original_initial_state=0) 
    go_explore_manager.add_state_vec_mine(
        np.array([0, 1, 2, 3, 4, 0]), np.array([0, 1, 2, 3, 4, 7]))
    assert np.array_equal(
        go_explore_manager.records.state_database, 
        np.array([[0, 7, 0],
                  [1, 0, 0],
                  [2, 0, 0],
                  [3, 0, 0],
                  [4, 0, 0],
                  [0, 0, 0]])), "State database is not correct after adding state 1"
    go_explore_manager.add_state_vec_mine(
        np.array([0, 1, 2, 3, 4, 0]), np.array([0, 1, 2, 3, 4, 7]))
    assert np.array_equal(
        go_explore_manager.records.state_database, 
        np.array([[7, 7, 0],
                  [1, 1, 0],
                  [2, 2, 0],
                  [3, 3, 0],
                  [4, 4, 0],
                  [0, 0, 0]])), "State database is not correct after adding state 2"
    go_explore_manager.add_state_vec_mine(
        np.array([0, 1, 2, 3, 4, 0, 0]), np.array([0, 1, 2, 3, 4, 7, 8]))
    assert np.array_equal(
        go_explore_manager.records.state_database, 
        np.array([[8, 0, 7],
                  [1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4],
                  [0, 0, 0]])), "State database is not correct after adding state 3"
    
    print("State database is correct after adding state")
    
    # Test sampling states
    sampled_states = go_explore_manager.sample_states(10)
    assert len(sampled_states) == 10, "Sampled states length is not correct"
    assert np.all(sampled_states >= 0), "Sampled states are not all non-negative"
    # Check that all the sampled states are in the state database
    assert np.all(np.isin(sampled_states, go_explore_manager.records.state_database)), "Sampled states are not in the state database"
    print("Sampled states are correct")

if __name__ == "__main__":
    print("Testing Go-Explore manager...")
    test_go_explore_manager()
    print("All tests passed.")
