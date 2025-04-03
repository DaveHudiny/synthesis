# Description: This file contains the implementation of the SparseRewardShaper class, which implements the reward shaping based on external policy.
# The class supports the following shaping methods: given by demonstrations and given by policy.

import enum

from tf_agents.trajectories import Trajectory

from agents.policies.fsc_copy import FSC

import numpy as np

import logging

logger = logging.getLogger(__name__)

import time


class RewardShaperMethods(enum.IntEnum):
    DEMONSTRATION = 1
    FSC_POLICY = 2


class ObservationLevel(enum.IntEnum):
    STATE_ACTION = 1
    OBSERVATION_ACTION = 2

class EpisodeOutcome(enum.IntEnum):
    RUNNING = 0
    SUCCESS_FINAL_STATE = 1
    FAILURE_FINAL_STATE = 2


class DemonstrationBuffer:
    def __init__(self, observation_length: int = 1, action_length: int = 1,
                 batch_size: int = 4, buffer_length: int = 1000, cyclic_buffer: bool = False):
        """Initializes the demonstration buffer. The buffer is used to store the demonstrations for the reward shaping."""
        self.actions_buffer = np.zeros(
            (batch_size, buffer_length, action_length))
        self.observations_buffer = np.zeros(
            (batch_size, buffer_length, observation_length))

        self.counter = 0
        self.observation_length = observation_length
        self.action_length = action_length
        self.buffer_length = buffer_length
        self.batch_size = batch_size
        self.cyclic_buffer = cyclic_buffer
        self.tupelized_buffer = None
        self.buffer_filled = False

    def add_pair(self, observation, actions) -> bool:
        """Adds the pair of observation and action to the buffer. If the buffer is full, the method returns True, otherwise False."""
        if self.counter >= self.buffer_length:
            logger.debug("The buffer is full.")
            self.buffer_filled = True
            if self.cyclic_buffer:
                self.counter = 0
            else:
                return True
        self.observations_buffer[:, self.counter, :] = observation
        actions = np.reshape(actions, (self.batch_size, self.action_length))
        self.actions_buffer[:, self.counter, :] = actions
        self.counter += 1
        return False

    def is_buffer_filled(self):
        return self.buffer_filled

    def get_buffer(self):
        return self.observations_buffer, self.actions_buffer

    def get_tupelized_buffer(self):
        return self.tupelized_buffer

    def clear_buffer(self):
        self.observations_buffer = np.zeros(
            (self.batch_size, self.buffer_length, self.observations_buffer.shape[2]))
        self.actions_buffer = np.zeros(
            (self.batch_size, self.buffer_length, self.actions_buffer.shape[2]))
        self.tupelized_buffer = None
        self.counter = 0
        self.buffer_filled = False

    def tupelize_demonstration_steps(self):
        concatenated = np.concatenate(
            (self.observations_buffer, self.actions_buffer), axis=2)
        self.tupelized_buffer = concatenated.reshape(
            (self.batch_size * self.buffer_length, self.observation_length + self.action_length))
        self.uniquize_demonstration_steps()
        
    def uniquize_demonstration_steps(self):
        if self.tupelized_buffer is None:
            self.tupelize_demonstration_steps()
        self.tupelized_buffer = np.unique(self.tupelized_buffer, axis=0)

    def are_pairs_in_buffer(self, observations, actions):
        """Checks whether the pairs of observations and actions are in the buffer.
        Args:
            observations (np.array): The observations with the same batch size as actions. The shape is (batch_size, observation_length).
            actions (np.array): The actions. The batch size must be the same as observations. The shape is (batch_size, action_length).

        Returns:
            np.array: The boolean array of the same batch size as the observations and actions. If the pair is in the buffer, the value is True, otherwise False.
                      The output is in the shape (batch_size, 1).
        """
        # assert observations.shape[0] == actions.shape[0], "The batch size of observations and actions must be the same."
        # if self.tupelized_buffer is None:
        #     self.tupelize_demonstration_steps()
        if self.tupelized_buffer is None:
            self.tupelize_demonstration_steps()
            # self.uniquize_demonstration_steps()
        
        actions = np.reshape(actions, (-1, self.action_length))
        query = np.concatenate((observations, actions), axis=1)
        matches = np.any(np.all(self.tupelized_buffer[:, None] == query, axis=2), axis=0)
        return matches

    # TODO: Implement the following method to allow for counting the number of the corresponding pairs in the buffer.


class SparseRewardShaper:
    def __init__(self, shaper_method: RewardShaperMethods, observation_level: ObservationLevel = ObservationLevel.OBSERVATION_ACTION,
                 maximum_reward: float = 1.0, batch_size: int = 256, buffer_length: int = 5, cyclic_buffer: bool = False,
                 observation_length: int = 1, action_length: int = 1):
        self.shaper_method = shaper_method
        self.observation_level = observation_level
        self.demonstration_buffer = DemonstrationBuffer(
            observation_length, action_length, batch_size, buffer_length, cyclic_buffer)
        self.fsc_set = False
        self.maximum_reward = maximum_reward

    def create_reward_function(self) -> callable:
        """Creates the reward function based on the current reward shaping method."""
        if self.shaper_method == RewardShaperMethods.DEMONSTRATION:
            def reward_function(observation, action):
                self.demonstration_buffer.tupelize_demonstration_steps()
                matches = self.demonstration_buffer.are_pairs_in_buffer(
                    observation, action)
                matches = np.reshape(matches, (-1))
                return np.where(matches, self.maximum_reward, 0)
            return reward_function
        elif self.shaper_method == RewardShaperMethods.FSC_POLICY:
            assert self.fsc_set, "FSC policy was not set."
            return lambda _: 0

    def add_demonstration(self, action, observation=None, states=None, ):
        """Adds a demonstration to the list of demonstrations. Used for demonstration based reward shaping."""

        assert self.shaper_method == RewardShaperMethods.DEMONSTRATION, "This method is only available for demonstration based reward shaping."
        assert observation is not None or states is not None, "Either observation or states must be provided."
        if self.observation_level == ObservationLevel.OBSERVATION_ACTION: 
            assert observation is not None, "Observation must be provided with OBSERVATION_ACTION level."
        if self.observation_level == ObservationLevel.STATE_ACTION:
            assert states is not None, "States must be provided with STATE_ACTION level."
        if self.observation_level == ObservationLevel.OBSERVATION_ACTION:
            self.demonstration_buffer.add_pair(observation, action)
        else:
            self.demonstration_buffer.add_pair(states, action)

    def set_fsc_policy(self, fsc: FSC):
        """Sets the FSC policy. Used for policy based reward shaping."""
        pass

if __name__ == "__main__":
    # Test the SparseRewardShaper class
    shaper = SparseRewardShaper(RewardShaperMethods.DEMONSTRATION, ObservationLevel.OBSERVATION_ACTION, batch_size=3)

    # Add demonstrations
    shaper.add_demonstration(observation=np.array([[1], [2], [3]]), action=np.array([4, 5, 6]))
    shaper.add_demonstration(observation=np.array([[2], [3], [4]]), action=np.array([5, 6, 7]))
    shaper.add_demonstration(observation=np.array([[3], [4], [5]]), action=np.array([6, 7, 8]))

    shaper.demonstration_buffer.uniquize_demonstration_steps()
    # Create the reward function
    reward_function = shaper.create_reward_function()

    # Test the reward function with the corrent and incorrect pairs
    assert reward_function(np.array([[1]]), np.array([4]))[0] > 0.0, "The reward function is not correct."
    assert reward_function(np.array([[1]]), np.array([5]))[0] == 0.0, "The reward function is not correct."
    assert reward_function(np.array([[1], [1]]), np.array([4, 5]))[0] > 0.0, "The reward function is not correct."
    assert reward_function(np.array([[1], [1]]), np.array([4, 5]))[1] == 0.0, "The reward function is not correct."
    
    # Clear the buffer and test the reward function again
    shaper.demonstration_buffer.clear_buffer()
    assert not shaper.demonstration_buffer.is_buffer_filled(), "The buffer should not be filled."
    assert reward_function(np.array([[1]]), np.array([4])) == [0.0], "The reward function is not correct."
    assert reward_function(np.array([[1]]), np.array([5])) == [0.0], "The reward function is not correct."

    print("All tests passed.")