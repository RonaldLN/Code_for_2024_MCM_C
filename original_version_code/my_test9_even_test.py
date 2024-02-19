import pandas as pd
import numpy as np

"""
['player1', 'player2',
       'p1_sets', 'p2_sets', 
       'p1_games', 'p2_games', 
       'p1_score', 'p2_score', 
       'server', 'serve_no', 
       'point_victor',
       'p1_ace', 'p2_ace',
       'p1_winner', 'p2_winner', 
       'p1_double_fault', 'p2_double_fault',
       'p1_unf_err', 'p2_unf_err',
       'p1_break_pt', 'p2_break_pt', 
       'p1_break_pt_won', 'p2_break_pt_won', 
       'p1_break_pt_missed', 'p2_break_pt_missed']
"""

i = 2

data = pd.read_csv(f"everyone/{i:02d}.csv")
data = np.array(data)

index_to_delete = list(range(i%2, 8, 2)) + list(range(i%2+11, 25, 2))

data = np.delete(data, index_to_delete, axis=1)

"""
['player1',
       'p1_sets',
       'p1_games',
       'p1_score',
       'server', 'serve_no', 
       'point_victor',
       'p1_ace',
       'p1_winner', 
       'p1_double_fault',
       'p1_unf_err',
       'p1_break_pt', 
       'p1_break_pt_won', 
       'p1_break_pt_missed']
"""

print(data[2][6])
input()

for x in data:
    if x[4] == i % 2:
        x[4] = 1
    else:
        x[4] = 0

    if x[6] == i % 2:
        x[6] = 1
    else:
        x[6] = 0

# 计算势头
a, b, c = 0.55, 0.25, 0.2
for j in range(len(data) - 3):
    data[j][-1] = a * data[j+1][6] + b * data[j+2][6] + c * data[j+3][6]

# save the data to the file
df = pd.DataFrame(data)
df.to_csv(f"everyone/{i:02d}.csv", index=False, header=True)
