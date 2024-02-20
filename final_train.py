import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense


data_csv = pd.read_csv("final_champion.csv")
data_csv = np.array([data_csv])
print(data_csv)

data_csv = data_csv[:, :, 2:].astype(np.float32)
print(data_csv)
nums = len(data_csv[0])
train_y, train_x, copy_for_val = data_csv[:, :-nums//3, :1], data_csv[:, :-nums//3, 1:], data_csv[:, :, :]
val_y, val_x = copy_for_val[:, :, :1], copy_for_val[:, :, 1:]
print(train_x)
print(train_y)


def build_model():
    model = Sequential()
    model.add(GRU(50, input_shape=(None, 14), return_sequences=True))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model


# 编译模型
model = build_model()
model.compile(optimizer='adam', loss="mean_absolute_error")

# 训练模型
model.fit(train_x, train_y, epochs=10000, batch_size=1)

# 保存权重文件
file_name = "Momen_final_3"
model.save_weights(file_name + '.h5')
model.save(file_name + ".keras")

pre_y = model.predict(val_x)
for p, v in zip(pre_y[0], val_y[0]):
    print("pre: ", p, "act: ", v)
