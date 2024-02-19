import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense


def csv_to_np_array(file):
    data = pd.read_csv(file)
    return np.array([data])


# data = pd.read_csv("test.csv")
# print(type(data))

# 将csv文件的第一行作为 names
# data = pd.read_csv("CA(test).csv", names=["player1", "point_no", "next_three",
#                                       "win_or_not", "p1_sets_leading", "p1_games_leading_1",
#                                       "p1_score_leading_1", "1_server", "p1_ace", "p1_winner",
#                                       "p1_double_fault", "p1_unf_err", "p1_break_pt",
#                                       "p1_break_pt_won", "p1_break_pt_missed"])
# data.head()
data = pd.read_csv("CA(test)_2.csv")
data = np.array([data])
print(data)

data = data[:, :, 2:].astype(np.float32)
print(data)
nums = len(data[0])
labels, data, val = data[:, :-nums//3, :1], data[:, :-nums//3, 1:], data[:, -nums//3:, :]
Shi, val = val[:, :, :1], val[:, :, 1:]
print(data)
print(labels)


def build_model():
    model = Sequential()
    model.add(GRU(50, input_shape=(None, 14), return_sequences=True))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model


# 编译模型
model = build_model()
model.compile(optimizer='adam', loss="mean_absolute_error")

# 训练模型
model.fit(data, labels, epochs=10000, batch_size=1)

# 保存权重文件
name = "Shi_3"
model.save_weights(name + '.h5')
model.save(name + ".keras")

pre = model.predict(val)
for _ in zip(pre[0], Shi[0]):
    print("pre: ", _[0], "act: ", _[1])

