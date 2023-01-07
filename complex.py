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

df = pd.read_csv('complex/data/train.csv')

np.random.shuffle(df.values)

model = Sequential([
    keras.layers.Dense(256, input_shape=(2,), activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
X = np.asanyarray(df.loc[:, df.columns.values != 'color']).astype(np.float64)
Y = np.asanyarray(df['color']).astype(np.float64)
Y = to_categorical(Y)
model.fit(X, Y, batch_size=128, validation_split=0.2, epochs=50, verbose=1)

test_df = pd.read_csv('complex/data/test.csv')
X_test = np.asanyarray(
    test_df.loc[:, test_df.columns.values != 'color']).astype(np.float64)
Y_test = np.asanyarray(test_df['color']).astype(np.float64)
Y_test = to_categorical(Y_test)

scores = model.evaluate(X_test, Y_test)
print(scores[1]*100)
