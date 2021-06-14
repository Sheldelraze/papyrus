from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet50


def get_vocal_cords_model(input_shape):
    base_model = resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=None)
    x1 = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x2 = keras.layers.GlobalMaxPooling2D()(base_model.output)
    x = keras.layers.concatenate([x1, x2])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.7)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    return model
