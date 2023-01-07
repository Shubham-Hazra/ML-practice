from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(
    (X_train.shape[0], X_train.shape[1]*X_train.shape[2])).astype('float32')
X_train = X_train/255
X_test = X_test.reshape(
    (X_test.shape[0], X_test.shape[1]*X_test.shape[2])).astype('float32')
X_test = X_test/255
input_shape = X_train.shape[1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def build_model():
    model = Sequential()
    model.add(keras.layers.Dense(
        input_shape, activation='relu', input_shape=(input_shape,)))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(10, activation='linear'))

    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = build_model()
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=10)

model.save('mnist_classification.h5')

model = keras.models.load_model('mnist_classification.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print(f"The accuracy of the model is {score[1]*100} %")

for i in range(5):
    random_num = randint(0, y_test.shape[0])
    print(
        f"The chosen number is {np.argmax(y_test[random_num], axis=None, out=None)}")
    # image = Image.fromarray(X_test[random_num].reshape(28, 28)*255)
    # image.show()
    num = model.predict(np.array([X_test[random_num]]), verbose=0)
    print(
        f"The number predicted by the model is {np.argmax(num, axis=None, out=None)}")
