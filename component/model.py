from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from component.muscular_degradation import get_muscular_degradation_model
from component.respiratory_tract import get_respiratory_tract_model
from component.sentiment import get_sentiment_model
from component.vocal_cords import get_vocal_cords_model


class CovidClassifier(keras.Model, ABC):
    def __init__(self, input_shape,
                 vocal_cords_path,
                 sentiment_path,
                 tract_path):
        self.biomaker1 = get_muscular_degradation_model(input_shape=input_shape)
        self.biomaker2 = get_vocal_cords_model()
        self.biomaker3 = get_sentiment_model()
        self.biomaker4 = get_respiratory_tract_model()
        self.load_weights(vocal_cords_path, sentiment_path, tract_path, )
        self.biomaker2 = self.biomaker2.get_layer('conv5_block3_add')
        self.biomaker3 = self.biomaker3.get_layer('conv5_block3_add')
        self.biomaker4 = self.biomaker4.get_layer('conv5_block3_add')

    def load_weights(self, vocal_cords_path, sentiment_path, tract_path, **kwargs):
        if vocal_cords_path is not None:
            self.biomaker2.load_weights(vocal_cords_path)
        if sentiment_path is not None:
            self.biomaker3.load_weights(sentiment_path)
        if tract_path is not None:
            self.biomaker4.load_weights(tract_path)

    def call(self, inputs, **kwargs):
        x = self.biomaker1(inputs)
        v = self.biomaker2(x)
        s = self.biomaker3(x)
        r = self.biomaker4(x)
        concat = layers.concatenate([v, s, r], axis=-1)
        x = tf.keras.layers.GlobalAveragePooling2D()(concat)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        return x
