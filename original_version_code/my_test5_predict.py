import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import load_model


# def build_model():
#     model = Sequential()
#     model.add(GRU(50, input_shape=(None, 12), return_sequences=True))  # 使用50个GRU单元
#     model.add(Dense(1))  # 输出层，用于回归问题
#
#     return model
#
#
# # 编译模型
# model = build_model()
# model.load_weights("Shi_2.h5")

model = load_model("Shi_3.keras")


data = pd.read_csv("CA(test)_2.csv")
data = np.array([data])
# print(data)

data = data[:, :, 2:].astype(np.float32)
# print(data)
labels, data, val = data[:, :-15, :1], data[:, :-15, 1:], data[:, :, :]
Shi, val = val[:, :, :1], val[:, :, 1:]

pre = model.predict(val)
for _ in zip(pre[0], Shi[0]):
    print("pre: ", _[0], "act: ", _[1])

Shi = np.array(pre[0])
# write Shi to csv
df = pd.DataFrame(Shi)
df.to_csv('Shi_3.csv', index=False, header=False)