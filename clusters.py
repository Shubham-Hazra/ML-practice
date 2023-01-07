from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
from tensorflow import keras

df = pd.read_csv('clusters/data/train.csv')

np.random.shuffle(df.values)

model = Sequential([
    keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(6, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
X = np.asanyarray(df.loc[:, df.columns.values != 'color']).astype(np.float64)
Y = df['color']
Y = np.asanyarray(pd.get_dummies(Y))
model.fit(X, Y, batch_size=64, epochs=20, verbose=1)

test_df = pd.read_csv('clusters/data/test.csv')
X_test = np.asanyarray(
    test_df.loc[:, test_df.columns.values != 'color']).astype(np.float64)
Y_test = test_df['color']
Y_test = np.asanyarray(pd.get_dummies(Y_test))

scores = model.evaluate(X_test, Y_test)
print(scores[1]*100)
