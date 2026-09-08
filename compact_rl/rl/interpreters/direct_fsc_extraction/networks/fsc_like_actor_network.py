import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from keras import layers, models, activations
from tf_agents.keras_layers import dynamic_unroll_layer

import tensorflow_probability as tfp
import math


class FSCLikeActorNetwork(models.Model):

    def __init__(self, observation_shape: tf.TensorShape,
                 action_range: int,
                 memory_len: int,
                 use_one_hot: bool = False,
                 use_residual_connection: bool = True,
                 gumbel_softmax_one_hot: bool = True,
                 stochastic_updates: bool = True,
                 seed: int = 42,
                 use_matrices : bool = False,
                 use_vq_vae: bool = False,
                 vq_commitment_cost: float = 0.25
                ):
        super(FSCLikeActorNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.memory_len = memory_len
        self.dense1 = layers.Dense(128, activation='relu')
        self.grumender = layers.Dense(64, activation="relu")
        self.grumender_2 = layers.Dense(64, activation=None)
        self.simple_rnn_for_memory = layers.GRU(
            64,
            return_sequences=True,
            return_state=True,
            reset_after=True,
            recurrent_initializer='orthogonal',
            dropout=0.2,
            recurrent_dropout=0.2)
        self.memory_pre_dense = layers.Dense(64, activation='relu')
        self.memory_dense = layers.Dense(memory_len, activation=None)
        self.pre_action_dense = layers.Dense(64, activation='relu')
        self.pre_action_dense_2 = layers.Dense(32, activation='relu')

        self.action = layers.Dense(self.action_range, activation=None)
        self.return_probs = True
        self.use_one_hot = use_one_hot
        self.gumbel_softmax_one_hot = gumbel_softmax_one_hot
        self.temperature = 0.5
        self.seed = seed
        self.use_matrices = use_matrices
        self.use_vq_vae = use_vq_vae
        self.vq_commitment_cost = vq_commitment_cost
        if gumbel_softmax_one_hot and use_matrices:
            assert use_one_hot, "Gumbel softmax requires one-hot encoding."
        if use_vq_vae:
            self._set_vq_vae(use_one_hot)
        elif gumbel_softmax_one_hot:
            self._set_gumbel_softmax_one_hot(use_one_hot, stochastic_updates, use_matrices)
        elif not use_one_hot:

            self.memory_function = layers.Lambda(
                lambda x: 1.5 * tf.tanh(x) + 0.5 * tf.tanh(-3 * x))
            self.quantization_layer = layers.Lambda(lambda x: tf.round(x))
            self.one_hot_constant = 0
        else:

            self.memory_function = layers.Lambda(
                lambda x: activations.sigmoid(x))
            self.quantization_layer = layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1),
                                                                         depth=self.memory_len, dtype=tf.float32))
            self.one_hot_constant = 1
        
        self.noise_level = 0.35

    def _set_vq_vae(self, use_one_hot: bool):
        """Replaces the Gumbel-Softmax relaxation with a VQ-VAE-style vector-quantized
        bottleneck: a learned codebook of |N| embeddings, a nearest-neighbour (L2)
        hard assignment, and a straight-through gradient estimator (van den Oord et al.,
        2017). Unlike Gumbel-Softmax, this bottleneck has no temperature schedule and is
        always hard (there is no continuous relaxation), so memory updates are
        deterministic rather than stochastic.
        """
        assert use_one_hot, "VQ-VAE requires one-hot encoding of memory nodes."
        self.vq_codebook = self.add_weight(
            name="vq_codebook",
            shape=(self.memory_len, self.memory_len),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-1.0 / self.memory_len, maxval=1.0 / self.memory_len),
            trainable=True,
        )
        self.one_hot_constant = 1

    def vq_nearest_codes(self, z_e):
        """Finds the nearest codebook entry (by squared L2 distance) for each row of
        z_e (shape (batch, memory_len)). Returns the one-hot assignment (shape
        (batch, memory_len)) and the corresponding codebook vectors z_q (same shape).
        """
        z_e_sq = tf.reduce_sum(tf.square(z_e), axis=-1, keepdims=True)  # (batch, 1)
        codebook_sq = tf.reduce_sum(tf.square(self.vq_codebook), axis=-1)  # (memory_len,)
        dot = tf.matmul(z_e, self.vq_codebook, transpose_b=True)  # (batch, memory_len)
        distances = z_e_sq + codebook_sq[tf.newaxis, :] - 2.0 * dot
        indices = tf.argmin(distances, axis=-1)
        one_hot = tf.one_hot(indices, depth=self.memory_len, dtype=tf.float32)
        z_q = tf.matmul(one_hot, self.vq_codebook)
        return one_hot, z_q

    def _set_single_vector_gumbel_softmax(self, use_one_hot: bool, stochastic_updates: bool):
        assert use_one_hot, "Gumbel softmax requires one-hot encoding."
        
        self.projection_network = layers.Dense(self.memory_len, activation='relu')
        self.memory_function = layers.Lambda(
            lambda x: self.gumbel_softmax(x, temperature=self.temperature, seed=self.seed))
        self.one_hot_constant = 1
        if stochastic_updates:
            self.quantization_layer = layers.Lambda(lambda x:
                                                    tf.one_hot(
                                                        tf.reshape(tf.random.categorical(
                                                            logits=tf.math.log(x), num_samples=1, dtype=tf.int32, seed=self.seed),
                                                            shape=(x.shape[0], -1)),
                                                        depth=self.memory_len,
                                                        dtype=tf.float32, axis=-1))
        else:
            self.quantization_layer = layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1),
                                                                         depth=self.memory_len, dtype=tf.float32))

    def _set_matrix_gumbel_softmax(self, use_one_hot: bool, stochastic_updates: bool):
        assert use_one_hot, "Gumbel softmax requires one-hot encoding."
        # Assure that memory_len is a perfect square for matrix reshaping.
        assert self.nr_classes_per_row ** 2 == self.memory_len, "Memory length must be a perfect square for matrix reshaping."
        self.projection_network = layers.Dense(self.memory_len, activation='relu') # Project to the matrix size.
        self.memory_function = layers.Lambda(
            lambda x: tf.reshape(self.gumbel_softmax(tf.reshape(x, (
                                                                    tf.shape(x)[0], 
                                                                    self.nr_classes_per_row, 
                                                                    self.nr_classes_per_row
                                                                   )
                                                                ), 
                                                     temperature=self.temperature, 
                                                     seed=self.seed), 
                                 (tf.shape(x)[0], self.memory_len)
                                )
            )
        self.one_hot_constant = 1
        if stochastic_updates:
            self.quantization_layer = layers.Lambda(lambda x:
                                                    tf.reshape(tf.one_hot(
                                                        tf.random.categorical(
                                                            logits=tf.math.log(tf.reshape(x, (
                                                                tf.shape(x)[0], 
                                                                self.nr_classes_per_row, 
                                                                self.nr_classes_per_row
                                                            )), axis=-1), num_samples=1, dtype=tf.int32, seed=self.seed),
                                                        depth=self.nr_classes_per_row,
                                                        dtype=tf.float32, axis=-1),
                                 (tf.shape(x)[0], self.memory_len)
                                ))
        else:
            self.quantization_layer = layers.Lambda(lambda x: tf.reshape(tf.one_hot(
                                                        tf.argmax(tf.reshape(x, (
                                                            tf.shape(x)[0], 
                                                            self.nr_classes_per_row, 
                                                            self.nr_classes_per_row
                                                        ), axis=-1), axis=-1),
                                                        depth=self.nr_classes_per_row,
                                                        dtype=tf.float32, axis=-1),
                                 (tf.shape(x)[0], self.memory_len)
                                ))
        

    def _set_gumbel_softmax_one_hot(self, use_one_hot: bool, stochastic_updates: bool, use_matrices: bool):
        if use_matrices:
            self.nr_classes_per_row = int(math.sqrt(self.memory_len))
            self._set_matrix_gumbel_softmax(use_one_hot, stochastic_updates)
        else:
            self._set_single_vector_gumbel_softmax(use_one_hot, stochastic_updates)
        
                

    def set_return_probs(self, return_probs: bool):
        """
        Set whether the network should return probabilities or not.
        """
        self.return_probs = return_probs

    def set_gumbel_temperature(self, temperature: float):
        """
        Set the temperature for the Gumbel-Softmax distribution.
        """
        self.temperature = temperature

    def get_initial_state(self, batch_size):
        if self.use_matrices:
            zeros_like = tf.zeros((batch_size, 
                                   self.nr_classes_per_row, 
                                   self.nr_classes_per_row - self.one_hot_constant)
                                 )
            init_memory_matrix = tf.concat([tf.ones((batch_size, self.nr_classes_per_row, self.one_hot_constant)), zeros_like], axis=-1)
            return tf.reshape(init_memory_matrix, (batch_size, self.memory_len))
        else:
            zeros_like = tf.zeros(
                (batch_size, self.memory_len - self.one_hot_constant))
        
            return tf.concat([tf.ones((batch_size, self.one_hot_constant)), zeros_like], axis=-1)
    
    def sample_gumbel(self, shape, eps = 1e-20, seed=None):
        """
        Sample from the Gumbel-Softmax distribution.
        """
        U = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax(self, logits, temperature=1.0, seed=None):
        """
        Sample from the Gumbel-Softmax distribution and return one-hot encoding.
        """
        gumbel_noise = self.sample_gumbel(tf.shape(logits), seed=seed)
        y = logits + gumbel_noise
        y = tf.nn.softmax(y / self.temperature, axis=-1)
        return y
    
    
    def set_noise_level(self, noise_level):
        """
        Set the noise level for the memory function output.
        """
        self.noise_level = noise_level

    def add_noise_to_neural_weights(self, seed):
        """
        Add noise to the neural network weights.
        """
        
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                noise = tf.random.normal(shape=tf.shape(layer.kernel), mean=0.0, stddev=0.01, seed=seed)
                layer.kernel.assign_add(noise)
            if hasattr(layer, 'bias'):
                layer.bias.assign_add(tf.random.normal(shape=tf.shape(layer.bias), mean=0.0, stddev=0.1, seed=seed))

    @tf.function
    def call(self, inputs, step_type: StepType, old_memory, seed):
        # inputs = tf.concat([inputs, tf.cast(old_memory, tf.float32)], axis=-1)
        self.seed = seed
        x = self.dense1(inputs)
        if old_memory is None:
            old_memory = self.get_initial_state(tf.shape(x)[0])
        old_memory = tf.where(tf.reshape(step_type, (-1, 1)) == StepType.FIRST, 
                              tf.stop_gradient(self.get_initial_state(tf.shape(x)[0])), old_memory)
        old_memory = self.grumender(old_memory)
        old_memory = self.grumender_2(old_memory)
        x, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)
        x = self.pre_action_dense(x)
        x = self.pre_action_dense_2(x)
        # x = layers.concatenate(x1, axis=-1)
        memory = self.memory_pre_dense(memory)
        memory = self.memory_dense(memory)
        if self.use_vq_vae:
            # VQ-VAE bottleneck: always a hard nearest-code assignment (no soft/temperature
            # regime), with a straight-through estimator and the standard codebook +
            # commitment losses (van den Oord et al., 2017) returned alongside the action.
            one_hot, z_q = self.vq_nearest_codes(memory)
            codebook_loss = tf.reduce_mean(tf.square(tf.stop_gradient(memory) - z_q))
            commitment_loss = tf.reduce_mean(tf.square(memory - tf.stop_gradient(z_q)))
            vq_loss = codebook_loss + self.vq_commitment_cost * commitment_loss
            memory = memory + tf.stop_gradient(one_hot - memory)
        else:
            vq_loss = tf.constant(0.0, dtype=tf.float32)
            memory = self.memory_function(memory)
            # memory += self.generate_noise(memory)
            if not self.return_probs:
                x_quantized = self.quantization_layer(memory)
                x_quantized = tf.reshape(x_quantized, (x_quantized.shape[0], -1))
                # Straight-through estimation, where we ignore round(x).
                memory = memory + tf.stop_gradient(x_quantized - memory)


        action = self.action(x)

        return action, memory, vq_loss
