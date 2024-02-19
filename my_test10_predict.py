import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Model, load_model


data = pd.read_csv("onlyone_3/1602.csv")
data = np.array([data])
val = data[:, :, 1:][:, :, :-1].astype(np.float32)

# 加载模型
model = load_model("Shi_everyone.keras")

pre = model.predict(val)
# for _ in zip(pre[0], Shi[0]):
#     print("pre: ", _[0], "act: ", _[1])
# print(data.shape, pre.shape)
# input()

# res = np.hstack((data[0], pre[0]))
# df = pd.DataFrame(res)
# df.to_csv('Shi_everyone.csv', index=False, header=False)



model_2 = load_model("Shi_onlyone.keras")

pre2 = model_2.predict(val)
res = np.hstack((data[0], pre2[0]))
# df = pd.DataFrame(res)
# df.to_csv('Shi_every_only_one_2.csv', index=False, header=False)



model_3 = load_model("Shi_onlyone_2.keras")

pre_3 = model_3.predict(val)
res = np.hstack((res, pre_3[0]))
df = pd.DataFrame(res)
df.to_csv('Shi_one_on_opponent.csv', index=False, header=False)



# all_datas = []
# for i in range(13, 18):
#     # if i != 5 and i != 6:
#     data = pd.read_csv(f"onlyone/{i:02d}01.csv")
#     all_datas.append(np.array(data)[:159-3, :])
# # data = pd.read_csv("CA(test)_2.csv")
# data = np.stack(tuple(all_datas))
# # print(data)
#
# data = data[:, :, 1:].astype(np.float32)  # delete name
# # print(data)
#
# # input()
#
# nums = len(data[0])
# labels, data = data[:, :, -1:], data[:, :, :-1]
# # Shi, val = val[:, :, :1], val[:, :, 1:]
# # print(data)
# # print(labels)
#
# model.fit(data, labels, epochs=10000, batch_size=5)
#
# name = "Shi_one_from_normal"
# model.save_weights(name + '.h5')
# model.save(name + ".keras")
#
# pre = model.predict(val)
#
# res = np.hstack((res, pre[0]))
# df = pd.DataFrame(res)
# df.to_csv('Shi_one_from_normal.csv', index=False, header=False)
