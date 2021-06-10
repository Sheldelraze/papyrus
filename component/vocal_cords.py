from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet50


def get_vocal_cords_model(input_shape=(300, 200)):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    return model

