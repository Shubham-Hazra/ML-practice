from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('mnist_classification_cnn.h5')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

random_num = randint(0, y_test.shape[0])
print(
    f"The chosen number is {np.argmax(y_test[random_num], axis=None, out=None)}")
image = Image.fromarray((
    X_test[random_num].reshape(28, 28)*255).astype(np.int32))
image = image.convert('RGB')
image.save('img.jpg')

image = Image.open('img.jpg')
image = image.convert('L')
image = np.array(image)
image = image.reshape(28, 28, 1)/255

num = model.predict(np.array([image]), verbose=0)
print(
    f"The number predicted by the model is {np.argmax(num, axis=None, out=None)} with accuracy {num.max()*100} %")
