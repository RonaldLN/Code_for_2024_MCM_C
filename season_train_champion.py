import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense


all_datas = []
for i in range(13, 18):
    # if i != 5 and i != 6:
    data_csv = pd.read_csv(f"person_champion/{i:02d}01.csv")
    all_datas.append(np.array(data_csv)[:159-3, :])
data_csv = np.stack(tuple(all_datas))
print(data_csv)

data_csv = data_csv[:, :, 1:].astype(np.float32)  # delete name
print(data_csv)

# input()

train_y, train_x, copy_for_val = data_csv[:, :, :1], data_csv[:, :, 1:], data_csv[:, :, :]
print(train_x)
print(train_y)

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
model.fit(train_x, train_y, epochs=10000, batch_size=5)

# 保存权重文件
name = "Momen_person_champion"
model.save_weights(name + '.h5')
model.save(name + ".keras")
