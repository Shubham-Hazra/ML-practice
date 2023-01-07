from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from PIL import Image
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def build_model():
    model = Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(2, 2),
              activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# model = build_model()
# model.fit(X_train, y_train, validation_data=(
#     X_test, y_test), epochs=10, batch_size=200, verbose=2)

# model.save('mnist_classification_cnn.h5')


model = keras.models.load_model('mnist_classification_cnn.h5')

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
