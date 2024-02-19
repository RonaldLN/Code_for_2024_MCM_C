import pandas as pd
import numpy as np

all_data = pd.read_csv("Wimbledon_featured_matches.csv")

print(all_data)
print(all_data.columns)

all_data = np.array(all_data)

print(all_data)

# exit(0)

# 将 match_id 结尾为 13xx 的数据筛选出来，并且将每一个 id 的数据存在单独的 csv 文件中
data_of_one = np.array(list(filter(lambda x: x[2] == "Novak Djokovic", all_data)))

print(data_of_one)

result = {}
for data in data_of_one:
    if data[0] not in result:
        result[data[0]] = []
    result[data[0]].append(data)

for data in result.values():
    df = pd.DataFrame(data)

    # name like: 00 01 02 ...
    df.to_csv(f"onlyone_2/{data[0][0][-4:]}.csv", index=False, header=True)
