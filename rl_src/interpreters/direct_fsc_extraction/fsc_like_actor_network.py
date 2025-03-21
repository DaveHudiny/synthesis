import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from keras import layers, models, activations


class FSCLikeActorNetwork(models.Model):
    def __init__(self, observation_shape: tf.TensorShape, 
                 action_range: int, 
                 memory_len: int, 
                 use_one_hot: bool = False, 
                 use_residual_connection: bool = True):
        super(FSCLikeActorNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.memory_len = memory_len
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.simple_rnn_for_memory = layers.SimpleRNN(
            memory_len, return_sequences=True, return_state=True)
        self.projection_network = layers.Dense(64, activation='relu')
        self.action = layers.Dense(self.action_range, activation=None)
        if not use_one_hot:
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
        self.use_residual_connection = use_residual_connection

    def get_initial_state(self, batch_size):
        zeros_like = tf.zeros(
            (batch_size, self.memory_len - self.one_hot_constant))
        return tf.concat([tf.ones((batch_size, self.one_hot_constant)), zeros_like], axis=-1)

    @tf.function
    def call(self, inputs, step_type: StepType, old_memory=None):
        # inputs = tf.concat([inputs, tf.cast(old_memory, tf.float32)], axis=-1)
        inputs = tf.concat([inputs], axis=-1)
        x = self.dense1(inputs)

        if self.use_residual_connection:
            x1, memory = self.simple_rnn_for_memory(
                x, initial_state=old_memory)
            x2 = self.projection_network(x1)
            x = layers.concatenate([x1, x2], axis=-1)
        else:
            x = self.dense2(x)
            x, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)

        # x2 = self.projection_network(x)
        # x = layers.concatenate(x1, axis=-1)

        memory = self.memory_function(memory)
        x_quantized = self.quantization_layer(memory)
        memory = memory + tf.stop_gradient(x_quantized - memory)

        action = self.action(x)
        # State-through estimation, where we ignore round(x)
        
        return action, memory
