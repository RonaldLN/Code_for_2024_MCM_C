import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense

# res = []
# for i in range(1, 33):
#     data = pd.read_csv(f"everyone/{i:02d}.csv")
#     res.append(len(np.array(data)))
#
# print(res)

res = [300, 300, 201, 201, 134, 134, 337, 337, 246, 246, 332, 332, 232, 232, 190, 190, 213, 213, 318, 318, 170, 170, 275, 275, 290, 290, 185, 185, 198, 198, 167, 167]

print(min(res))
print(len(list(filter(lambda x: x < 167, res))))

s = sum(res)

for x in res:
    sublist = list(filter(lambda y: y >= x, res))
    print(f"{s - sum(sublist)}, {len(sublist) * x}")


