import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils.CAF_GFU import GLU
from keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from keras import datasets, layers, models
from io import StringIO
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class ConvNet2D_ensemble_model():
    def setup_model(self, X,
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

            layers.Conv2D(filters = 2, kernel_size = (4,4)),
            layers.BatchNormalization(),
            layers.Activation(GLU(bias = True, dim = 1, name='glu')),

            layers.AlphaDropout(rate = dropoutRate, seed = random_seed),

            layers.Flatten(),
            layers.Dense(64, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(8, activation = 'relu', kernel_initializer='he_uniform'),
            layers.Dense(2, activation = 'softmax')
            ])

        model.compile(optimizer = "Adam",
                      loss = "sparse_categorical_crossentropy",
                      metrics = ["accuracy"])

        return model


    def evaluate_model(self, model, X, Y,
                        num_of_classes = 5,
                        path_load_save_model = "/home/vincent/Desktop/Models/cnn2D_binary.hdf5",
                        batch_size = 1,
                        epochs = 2):
        def create_data_ensemble_model(X_train, X_test, Y_train, Y_test, num_of_classes):
            def make_data_binary(Y, label):
                return np.array([1 if y == label else 0 for y in Y])

            def balancedSplit(X, Y):
                u, indices = np.unique(Y, return_counts=True)
                max_class_label = u[np.argmax(indices)]
                smallest_class = np.min(indices)
                X, Y = shuffle(X, Y)
                indices = np.array([])
                counter = 0

                for i, y in enumerate(Y):
                    if y != max_class_label:
                        indices = np.append(indices, i)
                    elif counter <= smallest_class:
                        indices = np.append(indices, i)
                        counter +=1

                indices = indices.astype(int)
                X = X[indices]
                Y = Y[indices]
                return (X, Y)

            data_ensemble_model = []

            for label in range(num_of_classes):
                Y_train_b, Y_test_b = make_data_binary(Y_train, label), make_data_binary(Y_test, label)
                X_train_b, Y_train_b = balancedSplit(X_train, Y_train_b)
                X_test_b, Y_test_b = balancedSplit(X_test, Y_test_b)
                data_ensemble_model.append([X_train_b, X_test_b, Y_train_b, Y_test_b])

            return data_ensemble_model

        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42, test_size= 0.2, stratify = Y)
        data = create_data_ensemble_model(X_train, X_test, y_train, y_test, num_of_classes)
        pred = np.zeros((np.shape(y_test)[0], num_of_classes))

        for i, d in enumerate(data):
            X_train_b, X_test_b, y_train_b, y_test_b = d

            callbacks = [ModelCheckpoint(filepath = path_load_save_model,
                                         monitor = 'val_loss',
                                         verbose = 1,
                                         save_best_only = True,
                                         mode = 'min')]

            hist = model.fit(x = X_train_b,y = y_train_b, epochs = epochs,
                   validation_data = (X_test_b, y_test_b), batch_size = batch_size, callbacks = callbacks)

            model = load_model(path_load_save_model , custom_objects={'GLU': GLU})

            pred[:, i] = model.predict(X_test)[:, 1]
            tf.keras.backend.clear_session()

        #LOGGING
        with StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()

        conf = tf.math.confusion_matrix(y_test, np.argmax(pred, axis = 1), num_classes = num_of_classes ).numpy().tolist()

        #[loss, Accuracy, ct], confusion_matrix, summary stream
        return([[(100/ len(y_test)) * np.sum(np.array([1 if y == np.argmax(pred_y) else 0 for y, pred_y in zip(y_test, pred)])),0], conf], summary_string)
