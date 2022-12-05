# -*- coding:utf-8 -*-
import torch
A = torch.arange(20).reshape(5,4)
print(A)
print(A.T) # 矩阵的转置

B = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])
print(B)
print(B == B.T) #对称矩阵，转置等于原矩阵

C = A.clone() # 通过分配新的内存，将A的一个副本分配给C
print(A+C)

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)   # 计算总和或均值时保持轴数不变
# 方便通过广播机制计算A/ sum_A
print(A.cumsum(axis=0)) #在列上进行累加求和

# L2范数是向量元素平方和的平方根
u = torch.tensor([3.0, -4.0])
U = torch.norm(u)
print(U)
# L1范数是向量元素绝对值的和
N = torch.abs(u).sum()
print(N)
# F范数 矩阵元素的平方和的平方根
n = torch.norm(torch.ones((4,9)))
print(n)
