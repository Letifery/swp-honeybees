from sklearn.model_selection import train_test_split
from keras import datasets, layers, models
from io import StringIO

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

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

class ConvNet3D_big_gated():
    def setup_model(self, X):
        no_classes = 4
        model = models.Sequential()
        model.add(layers.Conv3D(32,4, activation='relu', input_shape =(30, 220, 22, 1),kernel_regularizer='l2'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((1,2, 2)))
        model.add(layers.Conv3D(8,3, activation=tfa.activations.mish,kernel_regularizer='l2'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2,3, 3)))
        model.add(layers.Dropout(0.05))
        model.add(layers.Conv3D(8,2, activation=tfa.activations.mish,kernel_regularizer='l2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.02))
        model.add(layers.Activation(GLU(bias = False, dim=-1, name='glu')))

        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu', kernel_initializer='he_uniform',kernel_regularizer='l2'))
        model.add(layers.Dense(no_classes, activation='softmax'))

        model.summary()
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=["categorical_accuracy", "categorical_crossentropy"])
        return(model)

    def evaluate_model(self, model, X, y_one_hot):
        def convert_to_np_array(array):
            array = np.array(array)
            for i in range(len(array)):
                array[i]=np.array(array[i])
                return array
                
        X_train, X_test, y_train, y_test = train_test_split(convert_to_np_array(X), y_one_hot,test_size=0.33)

        X_train = convert_to_np_array(X_train)
        X_test = convert_to_np_array(X_test)
        y_train = convert_to_np_array(y_train)
        y_test = convert_to_np_array(y_test)
        
        arr = np.array([image for sublist in X_train for image in sublist])
        arr = arr.reshape((len(X_train), *np.shape(X[0])))

        hist = model.fit(x= arr,y=y_train, epochs=5)

        arr = np.array([image for sublist in X_test for image in sublist])
        arr = arr.reshape((len(X_test), *np.shape(X[0])))

        _ = model.evaluate(x= arr,y=y_test)

        preds = model.predict(arr)
    
        #LOGGING 
        with StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()

        conf = tf.math.confusion_matrix(np.argmax(y_test,axis=1), np.argmax(preds,axis=1),num_classes=4).numpy().tolist()
        #[loss, Accuracy, ct], confusion_matrix, summary stream
        return([model.evaluate(x= arr,y=y_test), conf], summary_string)
