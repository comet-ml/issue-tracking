#Trains a simple deep NN on the MNIST dataset.
from __future__ import print_function

#pre install comet_ml by running : pip install comet_ml
#make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def main():
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train(x_train,y_train,x_test,y_test)

def build_model_graph(input_shape=(784,)):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])

    return model

def train(x_train,y_train,x_test,y_test):
    # Define model
    model = build_model_graph()

    # initiate an Experiment with your api key from https://www.comet.ml
    experiment = Experiment(api_key="YOUR-API-KEY", project_name='keras-examples')
    experiment.log_dataset_hash(x_train)

    # and thats it... when you run your code all relevant data will be tracked and logged in https://www.comet.ml/view/YOUR-API-KEY
    model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

if __name__ == '__main__':
    main()
