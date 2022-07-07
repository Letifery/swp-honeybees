import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from CAF import GLU
from keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from keras import datasets, layers, models
from io import StringIO
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class ConvNet2D():
    def setup_model(self, X,
                    num_of_pred_classes = 5,
                    dropoutRate = 0.225,
                    random_seed = 42):

        model = models.Sequential([
            layers.Conv2D(filters = 32, kernel_size = 5, input_shape = (np.shape(X)[1],np.shape(X)[2],np.shape(X)[3])),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'SAME'),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 16, kernel_size = (5,3), strides = 2, padding = "SAME"),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(filters = 16, kernel_size = 5,  strides = 1, padding = 'SAME'),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 64, kernel_size = (5,3),  strides = 2, padding = 'SAME'),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = 64, kernel_size = 3, strides = (2,1), padding = 'SAME'),
            layers.BatchNormalization(),
            layers.Activation(tfa.activations.mish),

            layers.Conv2D(filters = num_of_pred_classes, kernel_size = (4,4)),
            layers.BatchNormalization(),
            layers.Activation(GLU(bias = True, dim = 1, name='glu')),

            layers.AlphaDropout(rate = dropoutRate, seed = random_seed),

            layers.Flatten(),
            layers.Dense(64, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(8, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(num_of_pred_classes, activation = 'softmax')
            ])

        model.compile(optimizer = "Adam",
                      loss = "sparse_categorical_crossentropy",
                      metrics = ["accuracy"])

        return model


    def evaluate_model(self, model, X, Y,
                        path_load_save_model = "/home/vincent/Desktop/Models/test.hdf5",
                        num_of_classes = 5,
                        batch_size = 2,
                        epochs = 25):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42, stratify = Y)

        callbacks = [ModelCheckpoint(filepath = path_load_save_model,
                                         monitor = 'val_loss',
                                         verbose = 1,
                                         save_best_only = True,
                                         mode = 'min')]

        hist = model.fit(x= X_train,y = y_train, epochs = epochs,
               validation_data=(X_test, y_test), batch_size = batch_size, callbacks = callbacks)

        model = load_model( path_load_save_model , custom_objects={'GLU': GLU})

        _ = model.evaluate(x = X_test,y = y_test)

        #LOGGING
        with StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()

        conf = tf.math.confusion_matrix(y_test, np.argmax(model.predict(X_test),axis = 1), num_classes = num_of_classes ).numpy().tolist()

        #[loss, Accuracy, ct], confusion_matrix, summary stream
        return([model.evaluate(x = X_test ,y = y_test), conf], summary_string)
