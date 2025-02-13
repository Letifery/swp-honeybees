from sklearn.model_selection import train_test_split
from keras import datasets, layers, models, activations
from io import StringIO

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from keras.callbacks import ModelCheckpoint

# Custom Activation Function GLU
class GLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, 2, self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "bias": self.bias,
            "dim": self.dim,
            "dense": self.dense,
        })
        return config

class CNN_LSTM():
    def setup_model(self, X):
        no_classes = 4
        model = models.Sequential()
        model.add(layers.ConvLSTM2D(filters=32,kernel_size=(3,3),input_shape =(*np.shape(X[0]), 1) ))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.AlphaDropout(0.08))
        model.add(layers.Conv2D(32,3, activation= activations.relu))
        model.add(layers.AlphaDropout(0.05))
        model.add(layers.BatchNormalization())
        

        model.add(layers.MaxPooling2D((2,2)))       
        model.add(layers.AlphaDropout(0.05))
        model.add(layers.Conv2D(24,3, activation= activations.relu))
        model.add(layers.AlphaDropout(0.1))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Activation(GLU(bias = False, dim=-1, name='glu')))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=16, activation=activations.relu))
        model.add(layers.Dense(units=8, activation=tfa.activations.mish))
        model.add(layers.Dense(units=4, activation=activations.softmax))

        model.summary()
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=["categorical_accuracy", "categorical_crossentropy"])
        return(model)

    def evaluate_model(self, model, X, y_one_hot, path_load_save_model = ""):
        def convert_to_np_array(array):
            array = np.array(array)
            for i in range(len(array)):
                array[i]=np.array(array[i])
                return array
                
        X_train, X_test, y_train, y_test = train_test_split(convert_to_np_array(X), y_one_hot,test_size=0.33, random_state=42)

        X_train = convert_to_np_array(X_train)
        X_test = convert_to_np_array(X_test)
        y_train = convert_to_np_array(y_train)
        y_test = convert_to_np_array(y_test)

        arr = np.array([image for sublist in X_train for image in sublist])
        X_train = arr.reshape((len(X_train), *np.shape(X[0])))

        arr = np.array([image for sublist in X_test for image in sublist])
        X_test = arr.reshape((len(X_test), *np.shape(X[0])))

        """callbacks = [ModelCheckpoint(filepath = path_load_save_model,
                                         monitor = 'val_categorical_accuracy',
                                         verbose = 1,
                                         save_best_only = True,
                                         mode = 'max')]"""

        hist = model.fit(x= X_train,y=y_train, batch_size=20, epochs=17 )

        _ = model.evaluate(x= X_test,y=y_test)

        preds = model.predict(X_test)
    
        #LOGGING 
        with StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()

        conf = tf.math.confusion_matrix(np.argmax(y_test,axis=1), np.argmax(preds,axis=1),num_classes=4).numpy().tolist()
        #[loss, Accuracy, ct], confusion_matrix, summary stream
        return([model.evaluate(x= X_test,y=y_test), conf], summary_string)
