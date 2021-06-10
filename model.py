from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet50
import tensorflow_probability as tfp


class MuscularDegradation(keras.Model, ABC):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class Sentiment(keras.Model, ABC):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class VocalCords(keras.Model, ABC):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class RespiratoryTract(keras.Model, ABC):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class CovidClassifier(keras.Model, ABC):
    def __init__(self):
        self.biomaker1 = MuscularDegradation()
        self.biomaker2 = VocalCords()
        self.biomaker3 = RespiratoryTract()
        self.biomaker4 = Sentiment()

    def call(self, inputs):
        pass


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    rate = layers.Dense(1, activation=tf.exp)(inputs)
    p_y = tfp.layers.DistributionLambda(tfp.distributions.Poisson)(rate)
    base_model = keras.Model(inputs=inputs, outputs=p_y)
    base_model.compile('adam', loss=tf.nn.sigmoid)
    return base_model


model = build_model((100, 100))
model.summary()
