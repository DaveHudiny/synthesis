import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from keras import layers, models, activations

class FSCLikeDictActorNetwork(models.Model):
    def __init__(self, observation_shape: tf.TensorShape,
                 action_range: int,
                 memory_len: int, use_one_hot: bool = False):
        super(FSCLikeDictActorNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.memory_len = memory_len
        self.dense1 = layers.Dense(128, activation='relu')
        self.grumender = layers.Dense(32, activation='tanh')
        self.simple_rnn_for_memory = layers.GRU(
            32, return_sequences=True, return_state=True)
        self.action = layers.Dense(self.action_range, activation=None)
        self.dictionary = tf.Variable(
            initial_value=tf.random.uniform((memory_len, 32), minval=-1, maxval=1),
            trainable=True,
            name="Embedding_Dict"
        )
        self.one_hot_constant = 1
        

    def get_initial_state(self, batch_size):
        zeros_like = tf.zeros(
            (batch_size, self.memory_len - self.one_hot_constant))
        return tf.concat([tf.ones((batch_size, self.one_hot_constant)), zeros_like], axis=-1)

    def sample_gumbel(self, shape, eps = 1e-20):
        """
        Sample from the Gumbel-Softmax distribution.
        """
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax(self, logits, temperature=1.0):
        """
        Sample from the Gumbel-Softmax distribution and return one-hot encoding.
        """
        gumbel_noise = self.sample_gumbel(tf.shape(logits))
        y = logits + gumbel_noise
        y = tf.nn.softmax(y / temperature, axis=-1)
        return y


    @tf.function
    def call(self, inputs, step_type: StepType, old_memory=None, training=True):
        # inputs = tf.concat([inputs, tf.cast(old_memory, tf.float32)], axis=-1)
        inputs = tf.where(step_type == StepType.LAST, 
                          tf.zeros_like(inputs), inputs)
        x = self.dense1(inputs)
        if old_memory is None:
            old_memory = self.get_initial_state(tf.shape(x)[0])
        old_memory = self.grumender(old_memory)
        x, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)

        # x2 = self.projection_network(x)
        # x = layers.concatenate(x1, axis=-1)

        memory_regularized = tf.nn.l2_normalize(memory, axis=-1)
        dictionary_normalized = tf.nn.l2_normalize(self.dictionary, axis=-1)
        similarity = tf.matmul(memory_regularized, dictionary_normalized, transpose_b=True) # Compute cosine similarity

        if training:
            memory = self.gumbel_softmax(similarity, temperature=1.0)
        else:
            memory = tf.one_hot(tf.argmax(similarity, axis=-1), depth=self.memory_len, dtype=tf.float32)

        action = self.action(x)
        # State-through estimation, where we ignore round(x).

        return action, memory
