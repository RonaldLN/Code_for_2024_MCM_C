import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt

model = load_model("Shi_3.keras")

test_datas = []
for i in range(14):
    test_datas += [np.array([[[1 if _ == i else 0 for _ in range(14)]]]).astype(np.float32)]

for test_data in test_datas:
    print(test_data)

    pre = model.predict(test_data)
    print(pre)
