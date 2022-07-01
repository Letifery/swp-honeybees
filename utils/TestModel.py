import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time

# import from utils 
from CAF_GLU import GLU
from keras.models import Sequential, load_model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split

class TestModel():
    def __init__(self, X, Y, model,
                 num_of_classes = 5,
                 random_state = 42,
                 test_size = 0.2,
                 optimizer = "Adam",
                 loss = "sparse_categorical_crossentropy",
                 metrics = ['accuracy'],
                 epochs = 20,
                 batch_size = 2):

        self.X = X
        self.Y = Y
        self.model = model
        self.random_state = random_state
        self.test_size = test_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_pred_classes = num_of_classes


    def calculate_majoryty_class(self, data):
        u, indices = np.unique(data, return_counts=True)
        return np.round((100/ np.sum(indices))*np.max(indices),3)


    def compile_model(self, model, optimizer,loss ,metrics):
        t = time.time()
        model.compile(optimizer = optimizer,
                              loss = loss,
                              metrics = metrics)

        print("[Compile time of model] time needed: %s seconds" % (time.time()-t))
        return model


    def fit_model(self, model, X_train, X_test, Y_train, Y_test, epochs, batch_size):
        t = time.time()
        history = model.fit(X_train, Y_train, epochs = epochs,
                        validation_data=(X_test, Y_test), batch_size = batch_size)
        print("[Fit time of model] time needed: %s seconds" % (time.time()-t))
        return (history, model)

    def evaluate_model(self, model, X_train, X_test, Y_train, Y_test, num_pred_classes):
        print("Majoryty class of Y_train is %s percent" % (self.calculate_majoryty_class(Y_train)))
        print("Majoryty class of Y_test is %s percent" % (self.calculate_majoryty_class(Y_test)))
        print("\n")

        print("Confusion Matrix")
        print(tf.math.confusion_matrix(np.argmax(model.predict(X_test), axis = 1), Y_test, num_classes = num_pred_classes))
        t = time.time()
        score = model.evaluate(X_test, Y_test, verbose=0)
        evaluationToneSample = (time.time()-t)/(len(Y_test)-1)
        print("\n")
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        print("\n")

        print("[Evaluation time of one sample] time needed: %s seconds" % evaluationToneSample)

    def test_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, random_state = self.random_state, test_size= self.test_size)

        model = self.compile_model(self.model, self.optimizer,self.loss ,self.metrics)
        model.summary()
        history, model = self.fit_model(model, X_train, X_test, Y_train, Y_test, self.epochs, self.batch_size)
        self.evaluate_model(model, X_train, X_test, Y_train, Y_test, self.num_pred_classes)
