import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import GRU, Dense

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

abalone_train = pd.read_csv(
    "test.csv",
    names=["x", "y", "z", "Sum"])

abalone_train.head()

print(abalone_train)

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Sum')

abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

abalone_model.compile(loss="mean_squared_error",
                      optimizer="adam")

def build_model():
    model = tf.keras.Sequential()
    model.add(GRU(50, input_shape=(None, 3), return_sequences=True))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model


model = build_model()
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(abalone_features, abalone_labels, epochs=100)
# abalone_model.fit(abalone_features, abalone_labels, epochs=100)

x_validate = pd.read_csv("test_validate.csv", names=["x", "y", "z"])

print(abalone_model.predict(x_validate))
