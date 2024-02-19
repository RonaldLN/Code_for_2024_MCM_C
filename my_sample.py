import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
import matplotlib.pyplot as plt


# 加载数据
x_train, y_train = ...
x_val, y_val = ...

dim = 18

# 构建模型
def build_model():
    model = Sequential()
    model.add(GRU(50, input_shape=(None, dim), return_sequences=True, activation="sigmoid"))  # 使用50个GRU单元
    model.add(Dense(1))  # 输出层，用于回归问题

    return model

# 编译模型
model = build_model()
model.compile(optimizer='adam', loss="mean_squared_error")

# 训练模型
model.fit(x_train, y_train, epochs=500, batch_size=32)

# 保存权重文件、模型文件
model.save_weights('weights.h5')
model.save('model.keras')

# 预测并计算每个序列输出的和
predictions = model.predict(x_val)

# 分析准确预测和不准确预测的特征
matching_inputs = []
non_matching_inputs = []

for i in range(len(predictions)):
    # 判断预测输出与真实输出是否匹配
    if np.allclose(predictions[i], y_val[i], atol=0.1):
        matching_inputs.append(x_val[i])
    else:
        non_matching_inputs.append(x_val[0][i])

matching_inputs_np = np.array(matching_inputs)
non_matching_inputs_np = np.array(non_matching_inputs)

# 可视化每个维度
for i in range(matching_inputs_np.shape[1]):
    plt.figure()
    plt.hist(matching_inputs_np[:, i], bins=20)
    plt.title(f'输入维度 {i + 1}')
    # plt.show()
    plt.savefig(f"pic/匹配的输入维度 {i + 1}.png")
    plt.close()

for i in range(non_matching_inputs_np.shape[1]):
    plt.figure()
    plt.hist(non_matching_inputs_np[:, i], bins=20)
    plt.title(f'输入维度 {i+1}')
    # plt.show()
    plt.savefig(f"pic/不匹配的输入维度 {i + 1}.png")
    plt.close()
