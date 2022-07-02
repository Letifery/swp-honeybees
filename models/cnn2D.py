import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from TestModel import TestModel
from CAF_GLU import GLU
from keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, BatchNormalization

class ConvNet():
    def cnn2D40x32x32(dropoutRate = 0.225, random_seed = 42):
        model = models.Sequential([
            layers.Conv2D(filters = 32, kernel_size = 5, input_shape = (40,32,32)),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'SAME', dilation_rate=2),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 16, kernel_size = (5,3), dilation_rate=1, strides = 2, padding = "SAME"),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 16, kernel_size = 5,  strides = 1, padding = 'SAME', dilation_rate=1),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 64, kernel_size = (5,3),  strides = 2, padding = 'SAME', dilation_rate=1),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 64, kernel_size = 3, strides = (2,1), dilation_rate=1, padding = 'SAME'),
            layers.BatchNormalization(),
            layers.Activation(GLU(bias = False, dim=-1, name='glu')),

            layers.Conv2D(filters = 5, kernel_size = (5,4)),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.AlphaDropout(rate = dropoutRate, seed = random_seed),

            layers.Flatten(),
            layers.Dense(64, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(8, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(5, activation = 'softmax')
            ])
        return model
