import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense


all_datas = []
for i in range(13, 18):
    # if i != 5 and i != 6:
    data = pd.read_csv(f"onlyone/{i:02d}01.csv")
    all_datas.append(np.array(data)[:159-3, :])
# data = pd.read_csv("CA(test)_2.csv")
data = np.stack(tuple(all_datas))
print(data)

data = data[:, :, 1:].astype(np.float32)  # delete name
print(data)

# input()

nums = len(data[0])
labels, data, val = data[:, :, -1:], data[:, :, :-1], data[:, -nums//3:, :]
# Shi, val = val[:, :, :1], val[:, :, 1:]
print(data)
print(labels)

# input()

def build_model():
    model = Sequential()
    model.add(GRU(50, input_shape=(None, 18), return_sequences=True))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model


# 编译模型
model = build_model()
model.compile(optimizer='adam', loss="mean_absolute_error")

# 训练模型
model.fit(data, labels, epochs=10000, batch_size=5)

# 保存权重文件
name = "Shi_onlyone"
model.save_weights(name + '.h5')
model.save(name + ".keras")

# pre = model.predict(val)
# for _ in zip(pre[0], Shi[0]):
#     print("pre: ", _[0], "act: ", _[1])

