import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

# 生成虚拟数据
def generate_data():
    # 生成输入序列和对应的目标值
    seq_length = 10
    num_samples = 10
    x = np.random.rand(num_samples, seq_length, 3)  # 3维向量的输入序列
    y = np.sum(x, axis=2)  # 目标值是输入序列各元素之和

    # 将y的每个元素转换为自身的列表
    y_as_list = np.array([[[element] for element in row] for row in y])

    return x, y_as_list

# 构建模型
def build_model():
    model = Sequential()
    model.add(GRU(50, input_shape=(None, 3), return_sequences=True))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model

# 编译模型
model = build_model()
model.load_weights('gru_weights.h5')

# 生成数据
x_train, y_train = generate_data()

print("x_train: \n" + x_train)
print("y_train: \n" + y_train)

for a, b in zip(x_train[0], y_train[0]):
    print(f"sum {a} = {b}")

# 预测并计算每个序列输出的和
predictions = model.predict(x_train)

# 打印结果
print("Predictions:")
print(predictions)

for i in range(10):
    print(i, " -------")
    for a, b, c in zip(x_train[-5 + i], y_train[-5 + i], predictions[-5+i]):
        print(f"act_sum {a} = {b}, pre: {c}")
