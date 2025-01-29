import tensorflow as tf
from keras import layers, models
import numpy as np

class Encoder(tf.keras.Model):
        def __init__(self, latent_dim, num_bins):
            super(Encoder, self).__init__()
            self.dense1 = layers.Dense(64, activation='relu')
            self.dense2 = layers.Dense(latent_dim, activation=None)
            self.num_bins = num_bins

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = 1.5 * tf.tanh(x) + 0.5 * tf.tanh(-3 * x)
            # Quantization
            x_quantized = tf.round(x) 
            # straight-through estimator (STE)
            x_quantized = x + tf.stop_gradient(x_quantized - x)
            return x_quantized

class Decoder(tf.keras.Model):
        def __init__(self, output_dim):
            super(Decoder, self).__init__()
            self.dense1 = layers.Dense(64, activation='relu')
            self.dense2 = layers.Dense(output_dim, activation=None)

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            return x
        
class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, num_bins):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim, num_bins)
        self.decoder = Decoder(input_dim)

    def call(self, inputs):
        discrete_latent = self.encoder(inputs)
        reconstructed = self.decoder(discrete_latent)

        return reconstructed
    
    def get_discrete_state(self, inputs):
        discrete_latent = self.encoder(inputs)
        return discrete_latent
    
    def encode(self, inputs) -> tf.Tensor:
        return self.encoder(inputs)
    
    def decode(self, inputs) -> tf.Tensor:
        return self.decoder(inputs)
    
    