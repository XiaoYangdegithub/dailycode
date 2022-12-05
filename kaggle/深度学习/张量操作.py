# -*- coding:utf-8 -*-
import torch

a = torch.arange(12).reshape(2,6)
print(a[:1])   # 取第0行
print(a[1,:])  # 取第一行
print(a[:,1])  # 取第二列
print(a[::1,::2])  # 从0行开始每一行取一行，从0列开始每两列取一列
print(a.sum())  # 对a里的值求和
print(a)

b = torch.tensor([[1,3,2],[1,2,3]])
c = torch.tensor([[1,2,3], [1,1,1]])
print(b+c,b-c,b*c,b**c) # **是求幂运算，以b为底数，c为幂
# 广播机制
'''
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
a+b  #将两个均广播到三行两列的矩阵进行计算
结果为 [[0,1],
       [1,2],
       [2,3]]
 '''

d = torch.cat((b,c),dim=0) # 在列上合并
e = torch.cat((b,c),dim=-1) # 在行上合并
print(d,e)

f = torch.exp(d)
# 对d中的元素做指数运算
