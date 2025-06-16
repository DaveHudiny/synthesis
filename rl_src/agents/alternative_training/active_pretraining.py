# Author: David Hudák
# Description: Active pretraining for RL agents using a replay buffer and a policy network.

import numpy as np

import tensorflow as tf

from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment


from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

class EntropyRewardGenerator:
    """Generates entropy rewards for the agent based on the observations."""

    def __init__(self, binary_flag : bool = False, full_observability_flag : bool = False, max_reward : float = 1.0, decreaser : str = 'halve'):
        self.binary_flag = binary_flag
        self.full_observability_flag = full_observability_flag
        self.max_reward = max_reward
        self.reward = max_reward
        self.running_mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.experience_buffer = ExperienceBuffer(feature_spec_size=0)  # Initialize with a dummy size, will be set later

    @tf.function
    def find_k_nearest_neigbhors(observations: tf.Tensor, data : tf.Tensor, k: int = 1) -> tf.Tensor:
        """Finds the k-nearest neighbors for each observation (observations has shape [batch_size, F], where N is size of feature vector)."""
        # Compute pairwise distances between observations
        pairwise_distances = tf.norm(tf.expand_dims(observations, 1) - tf.expand_dims(data, 0), axis=-1)

        # Get the indices of the k-nearest neighbors for each observation
        _, indices = tf.nn.top_k(-pairwise_distances, k=k+1)

        # Exclude the observation itself (the first neighbor is always the observation itself)
        indices = indices[:, 1:]
        return indices
    
    @tf.function
    def find_k_nearest_neighbors_binary(observations: tf.Tensor, data: tf.Tensor, k: int = 1) -> tf.Tensor:
        """Finds the k-nearest neighbors by number of different features for each observation (observations has shape [batch_size, F], where N is size of feature vector)."""
        # Compute pairwise distances between observations
        pairwise_distances = tf.reduce_sum(tf.cast(tf.not_equal(tf.expand_dims(observations, 1), tf.expand_dims(data, 0)), tf.float32), axis=-1)

        # Get the indices of the k-nearest neighbors for each observation
        _, indices = tf.nn.top_k(-pairwise_distances, k=k+1)

        # Exclude the observation itself (the first neighbor is always the observation itself)
        indices = indices[:, 1:]
        return indices  

    @tf.function
    def compute_observation_entropy(observations: tf.Tensor, data: tf.Tensor, k = 3) -> tf.Tensor:
        """Computes the entropy of the observation.
        
        Source: https://proceedings.nips.cc/paper/2021/file/99bf3d153d4bf67d640051a1af322505-Paper.pdf

        Args:
            observations: A tensor of shape [batch_size, F] where F is the size of the feature vector.
            k: The number of nearest neighbors to consider for the entropy calculation.
        Returns:
            A tensor of shape [batch_size] containing the entropy for each observation.
        1. Find the k-nearest neighbors for each observation.
        2. Compute the entropy based on the distances to the k-nearest neighbors.
        """
        # Find the most different observation vector to each observation vector.
        neighbors = EntropyRewardGenerator.find_k_nearest_neigbhors(observations, data, k=k)

        # Compute the distances to the k-nearest neighbors
        distances = tf.norm(tf.expand_dims(observations, 1) - tf.gather(data, neighbors), axis=-1) ** observations.shape[-1]  # Raise to the power of the number of features

        # Compute the entropy based on the distances
        reward_entropy = tf.math.log(tf.reduce_mean(distances, axis=-1) + 1)  # Add a small constant to avoid log(0)
        return reward_entropy

    @tf.function
    def compute_observation_entropy_binary(observations: tf.Tensor, data: tf.Tensor, k = 3) -> tf.Tensor:
        """Computes the binary entropy of the observation."""
        # Find the most different observation vector to each observation vector.
        neighbors = EntropyRewardGenerator.find_k_nearest_neighbors_binary(observations, data, k=k)

        # Compute the distances to the k-nearest neighbors
        distances = tf.reduce_sum(
                        tf.cast(
                            tf.not_equal(tf.expand_dims(observations, 1), tf.gather(data, neighbors)), 
                            tf.float32
                        ), 
                        axis=-1
                        ) ** observations.shape[-1]  # Raise to the power of the number of features
        # Compute the entropy based on the distances
        reward_entropy = tf.math.log(tf.reduce_mean(distances, axis=-1) + 1)

        return reward_entropy

    # @tf.function
    def compute_entropy_reward(self, observation : tf.Tensor = None, state : tf.Tensor = None) -> tf.Tensor:
        """Computes the entropy reward for the environment."""
        if self.full_observability_flag:
            observations = state
        else:
            observations = observation
        self.experience_buffer.add_experience(observations)
        data = self.experience_buffer.get_buffer()
        
        if self.binary_flag:
            entropy = EntropyRewardGenerator.compute_observation_entropy_binary(observations, data)
        else:
            entropy = EntropyRewardGenerator.compute_observation_entropy(observations, data)
        self.running_mean.assign(0.9 * self.running_mean + 0.1 * tf.reduce_mean(entropy))
        

        return entropy / (tf.abs(self.running_mean) + 1e-8)  # Add a small constant to avoid division by zero

    def halve_entropy_reward(self) -> tf.Tensor:
        """Halves the entropy reward."""
        self.reward /= 2.0

    def decrease_entropy_reward_linear(self) -> tf.Tensor:
        """Decreases the entropy reward linearly."""

        self.reward -= self.max_reward / 100.0

    def decrease_entropy_reward_continuous(self) -> tf.Tensor:
        """Decreases the entropy reward continuously."""
        self.reward *= 0.95
        

class ExperienceBuffer:
    """A buffer for storing experiences."""
    
    def __init__(self, feature_spec_size : int, max_size: int = 10000):
        self.buffer = np.zeros((max_size, feature_spec_size), dtype=np.float32)
        self.max_size = max_size
        self.current_head = 0

    def resize(self, feature_spec_size: int):
        """Resizes the buffer to accommodate a new feature specification size."""
        self.buffer = np.zeros((self.max_size, feature_spec_size), dtype=np.float32)
    
    def add_experience(self, experience: tf.Tensor):
        """Adds an experience to the buffer."""
        if self.buffer.shape[1] != experience.shape[-1]:
            self.resize(experience.shape[-1])
        buffer_indices = np.arange(self.current_head, self.current_head + experience.shape[0]) % self.max_size
        self.buffer[buffer_indices] = experience.numpy()  # Convert tf.Tensor to numpy array
        self.current_head = (self.current_head + experience.shape[0]) % self.max_size

    def get_buffer(self) -> tf.Tensor:
        """Returns the buffer as a tf.Tensor."""
        return tf.convert_to_tensor(self.buffer, dtype=tf.float32)
                    


        
        
        