from sklearn.model_selection import train_test_split
from keras import datasets, layers, models
from io import StringIO

from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

class ConvNet3D():
    def setup_model(self, X):
        model = models.Sequential()
        model.add(layers.Conv3D(16,3, activation='relu', input_shape =(*np.shape(X[0]), 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2, 4, 4)))
        model.add(layers.Conv3D(6,2, activation=tfa.activations.mish))
        model.add(layers.BatchNormalization())
    
        model.add(layers.Flatten())
        model.add(layers.Dense(8, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(4, activation='softmax'))

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
                
        X_train, X_testb, y_train, y_testb = train_test_split(convert_to_np_array(X), y_one_hot,test_size=0.33)
                                      
        X_train = convert_to_np_array(X_train)
        X_test = convert_to_np_array(X_testb)
        y_train = convert_to_np_array(y_train)
        y_test = convert_to_np_array(y_testb)
        
        arr = np.array([image for sublist in X_train for image in sublist])
        arr = arr.reshape((len(X_train), *np.shape(X[0])))

        hist = model.fit(x= arr,y=y_train, epochs=7)

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
