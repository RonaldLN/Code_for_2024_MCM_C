import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Model, load_model


data = pd.read_csv("person_3rd/1602.csv")
data = np.array([data])
val = data[:, :, 1:][:, :, :-1].astype(np.float32)

# 加载模型
model = load_model("Momen_public.keras")

pre = model.predict(val)
# for _ in zip(pre[0], Shi[0]):
#     print("pre: ", _[0], "act: ", _[1])
# print(data.shape, pre.shape)
# input()

# res = np.hstack((data[0], pre[0]))
# df = pd.DataFrame(res)
# df.to_csv('Momen_public.csv', index=False, header=False)



model_2 = load_model("Momen_person_champion.keras")

pre2 = model_2.predict(val)
res = np.hstack((data[0], pre2[0]))
# df = pd.DataFrame(res)
# df.to_csv('Momen_public_and_champion_2.csv', index=False, header=False)



model_3 = load_model("Momen_person_2nd.keras")

pre_3 = model_3.predict(val)
res = np.hstack((res, pre_3[0]))
df = pd.DataFrame(res)
df.to_csv('Momen_champion_on_opponent_final.csv', index=False, header=False)



"""
second train champion model based on public model
"""

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
# name = "Momen_champion_from_public"
# model.save_weights(name + '.h5')
# model.save(name + ".keras")
#
# pre = model.predict(val)
#
# res = np.hstack((res, pre[0]))
# df = pd.DataFrame(res)
# df.to_csv('Momen_champion_from_public.csv', index=False, header=False)
