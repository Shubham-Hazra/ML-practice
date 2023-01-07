import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

# Load data
df = pd.read_csv("./clusters_two_categories/data/train.csv")

x_train = np.array(df[['x', 'y']])
y_train = np.array(df[['color', 'marker']])


enc = MultiLabelBinarizer()
y_train = enc.fit_transform(y_train)

np.random.RandomState(42).shuffle(x_train)
np.random.RandomState(42).shuffle(y_train)

num_classes = y_train.shape[1]

# Build model
model = Sequential([
    BatchNormalization(input_shape=(2,)),
    Dense(256, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(num_classes, activation='linear'),
])

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25, batch_size=128,
          verbose=1, validation_split=0.3)

model.save('two_clusters_improved.h5')

test_df = pd.read_csv('clusters_two_categories/data/test.csv')
X = test_df[['x', 'y']].values
Y = np.array(test_df[['color', 'marker']])
Y = MultiLabelBinarizer().fit_transform(Y)
scores = model.evaluate(X, Y)
print(scores[1]*100)
