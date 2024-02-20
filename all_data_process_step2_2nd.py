import pandas as pd
import numpy as np

all_data = pd.read_csv("Wimbledon_featured_matches.csv")

print(all_data)
print(all_data.columns)

col_list = list(all_data.columns)
thing_to_find = ("match_id", "elapsed_time", "set_no", "game_no", "point_no",
                 "game_victor", "set_victor", "p1_net_pt", "p2_net_pt", "p1_net_pt_won",
                 "p2_net_pt_won", 'p1_points_won', 'p2_points_won')
# get the index of the column
index_to_find = []
for i in range(len(col_list)):
    if col_list[i] in thing_to_find:
        index_to_find.append(i)

# exit(0)

for i in ("1316", "1408", "1504", "1602", "1701"):
    data = pd.read_csv(f"persion_2nd/{i}.csv")
    data = np.array(data)
    # delete the column whose index is in index_to_find
    data = np.delete(data, index_to_find, axis=1)
    # print(data)

    # save the data to the file
    df = pd.DataFrame(data)
    df.to_csv(f"persion_2nd/{i}.csv", index=False, header=True)
