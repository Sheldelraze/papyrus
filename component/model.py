import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from component.muscular_degradation import get_muscular_degradation_model
from component.respiratory_tract import get_respiratory_tract_model
from component.sentiment import get_sentiment_model
from component.vocal_cords import get_vocal_cords_model


def get_covid_classifier(input_shape,
                 vocal_cords_path,
                 sentiment_path,
                 tract_path):
        biomaker1 = get_muscular_degradation_model(input_shape=input_shape)
        biomaker2 = get_vocal_cords_model(input_shape=input_shape)
        biomaker3 = get_sentiment_model(input_shape=input_shape)
        biomaker4 = get_respiratory_tract_model(input_shape=input_shape)
        if vocal_cords_path is not None:
            biomaker2.load_weights(vocal_cords_path)
        if sentiment_path is not None:
            biomaker3.load_weights(sentiment_path)
        if tract_path is not None:
            biomaker4.load_weights(tract_path)
        biomaker2 = keras.models.Model(inputs=[biomaker2.input], outputs=[biomaker2.get_layer('conv5_block3_add').output])
        biomaker3 = keras.models.Model(inputs=[biomaker3.input], outputs=[biomaker3.get_layer('conv5_block3_add').output])
        biomaker4 = keras.models.Model(inputs=[biomaker4.input], outputs=[biomaker4.get_layer('conv5_block3_add').output])
        v = biomaker2(biomaker1.output)
        s = biomaker3(biomaker1.output)
        r = biomaker4(biomaker1.output)
        x = layers.concatenate([v, s, r], axis=1)
        x = layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.7)(x)
        x = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        model = keras.models.Model(inputs=[biomaker1.input], outputs=[x])
        return model
