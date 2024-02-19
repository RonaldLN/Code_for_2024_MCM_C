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


for i in ("1309", "1405", "1503", "1602"):
    data = pd.read_csv(f"onlyone_3/{i}.csv")
    data = np.array(data)

    # 算 领先盘数 领先局数 领先球数 是不是一局最后三球 几球赢这局 势头
    # add a new column to the end of data
    new_column = np.zeros((len(data), 6))
    data = np.hstack((data, new_column))

    for j in range(len(data)):
        data[j][-6] = data[j][2] - data[j][3]
        data[j][-5] = data[j][4] - data[j][5]
        data[j][-4] = data[j][6] - data[j][7]
        # if i % 2 == 0:
        # data[j][-6] *= -1
        # data[j][-5] *= -1
        # data[j][-4] *= -1

        # 是否影响下一局
        if j < len(data) - 3:
            data[j][-3] = 0 if data[j][8] == data[j+3][8] else 1

        # 判断还剩几球赢这局
        me, they = (6, 7)  # if i % 2 == 1 else (7, 6)
        if data[j][4] == 6 and data[j][5] == 6:  # 判断是否抢七
            target = 10 if data[j][2] == 2 and data[j][3] == 2 else 7
            # 对手超过6就加分
            data[j][-2] = data[j][they] + 2 - data[j][me] if data[j][they] >= target - 1 else target - data[j][me]
        else:
            if data[j][they] < 3:
                data[j][-2] = 4 - data[j][me]
            elif data[j][they] == 3:
                data[j][-2] = 5 - data[j][me]
            else:
                data[j][-2] = 3


    index_to_delete = list(range(1, 8, 2)) + list(range(1+11, 25, 2))

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

    for x in data:
        if x[4] == 1:
            x[4] = 1
        else:
            x[4] = 0

        if x[6] == 1:
            x[6] = 1
        else:
            x[6] = 0

    # 计算势头
    a, b, c = 0.55, 0.25, 0.2
    for j in range(len(data) - 3):
        data[j][-1] = a * data[j+1][6] + b * data[j+2][6] + c * data[j+3][6]

    # save the data to the file
    df = pd.DataFrame(data)
    df.to_csv(f"onlyone_3/{i}.csv", index=False, header=True)
