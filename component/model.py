from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from component.muscular_degradation import get_muscular_degradation_model
from component.respiratory_tract import get_respiratory_tract_model
from component.sentiment import  get_sentiment_model
from component.vocal_cords import get_vocal_cords_model


class CovidClassifier(keras.Model, ABC):
    def __init__(self, input_shape):
        self.biomaker1 = get_muscular_degradation_model(input_shape=input_shape)
        self.biomaker2 = get_vocal_cords_model()
        self.biomaker3 = get_sentiment_model()
        self.biomaker4 = get_respiratory_tract_model()

    def call(self, inputs):
        x = self.biomaker1(inputs)
        v = self.biomaker2(x)
        l = self.biomaker3(x)
        s = self.biomaker4(x)
        concat = layers.concatenate([v, l, s], axis=-1)
        x = tf.keras.layers.GlobalAveragePooling2D()(concat)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        return x