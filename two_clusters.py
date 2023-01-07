from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import keras

df = pd.read_csv('clusters_two_categories/data/train.csv')


model = Sequential([
    keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(7, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

X = np.asanyarray(df[['x', 'y']]).astype(np.float64)
Y = df[['color', 'marker']]
Y = np.asanyarray(OrdinalEncoder().fit_transform(Y))
Y = Y.sum(axis=1)

np.random.RandomState(seed=42).shuffle(X)
np.random.RandomState(seed=42).shuffle(Y)


model.fit(X, Y, epochs=15, validation_split=0.3, verbose=1)

test_df = pd.read_csv('clusters_two_categories/data/test.csv')
X = np.asanyarray(test_df[['x', 'y']]).astype(np.float64)
Y = test_df[['color', 'marker']]
Y = np.asanyarray(OrdinalEncoder().fit_transform(Y))
Y = Y.sum(axis=1)
scores = model.evaluate(X, Y)
print(scores[1]*100)
