# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch.utils import data

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True): # 调用框架中现有的API来读取数据 构造一个pytorc数据迭代器
    dataset = data.TensorDataset(*data_arrays) # 把输入的两类数据进行对应 *表示接受多个参数并将其放在一个元组中
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 重新排序 is_train区分是否为训练集

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))  # 得到x和y

# 使用框架预定义好的层
from torch import nn # nn神经网络

net = nn.Sequential(nn.Linear(2, 1)) # sequential是一个容器可以把模型放在list of layers里

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 初始化权重
net[0].bias.data.fill_(0) # 初始化偏差

# 计算均方误差
loss = nn.MSELoss()

# sgd随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 进行训练
num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch{epoch + 1}, loss{l:f}')