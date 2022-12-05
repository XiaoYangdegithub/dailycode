# -*- coding:utf-8 -*-
import torch
import pandas as pd
import os

os.makedirs(os.path.join('../../Desktop/kaggle', 'data'), exist_ok=True)
data_file = os.path.join('../../Desktop/kaggle', 'data', 'house_tiny.csv')
with open(data_file, 'w')as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) # 将不为NA的值的均值填在NA处
print(inputs)
# 处理类别值或离散值，可做划分类处理，如将’NaN‘视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x,y)