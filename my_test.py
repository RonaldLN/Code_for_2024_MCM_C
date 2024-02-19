import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from d2l import torch as d2l
import torch
import torch.nn as nn
import csv
path="."
BCHAIN_MKPRU=pd.read_csv(path+"/BCHAIN-MKPRU.csv",dtype={"Date":np.str,"Value":np.float64})
LBMA_GOLD=pd.read_csv(path+"/LBMA-GOLD.csv",dtype={"Date":np.str,"Value":np.float64})
Data=pd.read_csv(path+"/C题处理后的中间文件2.csv")

def to_timestamp(date):
    return int(time.mktime(time.strptime(date,"%m/%d/%y")))

#将日期变为自然数
start_timestamp=to_timestamp(Data.iloc[0,0])
for i in range(Data.shape[0]):
    Data.iloc[i,0]=(to_timestamp(Data.iloc[i,0])-start_timestamp)/86400
print(Data)

batch_size=1 # 应该只能为1
start_input=30
input_size=Data.shape[0]#训练：通过前input_size天预测input_size+1天，预测：通过2到input_size+1天预测第input_size+2天
hidden_size=20
# input_size=200
output_size=1
layers_size=3
lr=10
num_epochs=100

# exit(0)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        """
        Initializes a GRUModel instance.

        Args:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            output_size (int): The number of output features
            layers_size (int): Number of recurrent layers

        """
        super().__init__()
        self.GRU_layer = nn.GRU(input_size, hidden_size, layers_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the GRUModel.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch, input_size)

        Returns:
            Tensor: Output tensor of shape (seq_len, batch, output_size)

        """
        x, _ = self.GRU_layer(x)
        x = self.linear(x)
        return x

device=torch.device("cuda")

gru=GRUModel(30, hidden_size, output_size, layers_size).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(gru.parameters(), lr)

# ji=np.array(Data.iloc[0:input_size,3].dropna())
# input_size=ji.shape[0]-2
#
# print("ji:",ji)

# pause to wait key enter to continue
# input("Press Enter to continue...")
ji=np.array(Data.iloc[0:input_size].dropna(), dtype=np.float32)
print("ji:",ji)
input_size=ji.shape[0]-2



trainB_x=torch.from_numpy(ji[input_size-30:input_size].reshape(-1,batch_size, 30)).to(torch.float32).to(device)
trainB_y=torch.from_numpy(ji[input_size].reshape(-1,batch_size,output_size)).to(torch.float32).to(device)

losses = []

for epoch in range(num_epochs):
    output = gru(trainB_x).to(device)
    loss = criterion(output, trainB_y)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss" + str(epoch) + ":", loss.item())

# 预测，以比特币为例
# pred_x_train=torch.from_numpy(np.array(Data.iloc[1:input_size+1,1]).reshape(-1,1,input_size)).to(torch.float32).to(device)
pred_x_train=torch.from_numpy(ji[input_size-29:input_size+1]).reshape(-1,1,30).to(torch.float32).to(device)
pred_y_train=gru(pred_x_train).to(device)
# print("prediction:",type(pred_y_train.item()))

# 将 PyTorch tensor 转换为 NumPy 数组
pred_y_train_np = pred_y_train.cpu().detach().numpy()

# 输出所有值
print("prediction:", pred_y_train_np)

# 或者输出展平后的数组
print("prediction (flattened):", pred_y_train_np.flatten())

print("actual:",ji[input_size+1])