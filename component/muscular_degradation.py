import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


def get_muscular_degradation_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    rate = layers.Dense(1, activation=tf.exp)(inputs)
    p_y = tfp.layers.DistributionLambda(tfp.distributions.Poisson)(rate)
    model = keras.Model(inputs=inputs, outputs=p_y)
    return model
