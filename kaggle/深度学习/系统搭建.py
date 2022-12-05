# 模型构造
# 层和块--多层感知机
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))  # Linear全连接层，第一层输入维度20输出256，且会自动初始化

X = torch.rand(2, 20)  # 2是批量大小
net(X)
'''print(X)'''
# 自定义块
class MLP(nn.Module):  # 定义了一个MLP的模块继承了nn.Module
    def __int__(self):  # 所用的层都在里面
        super().__int__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))  # F.relu是调用函数
