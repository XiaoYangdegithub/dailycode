# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
'''  # 单变量线性回归
path = 'data/regress_data1.csv'
data = pd.read_csv(path)
print(data.head())
print(data.describe())
data.plot(kind='scatter', x='人口', y='收益', figsize=(12, 8))
plt.xlabel('人口', fontsize = 18)
plt.ylabel('收益', rotation = 0, fontsize = 18)
plt.show() # 绘制出数据集的样子
'''
# 创建loss -- 创建一个以参数w为特征函数的代价函数 -- 代价函数（全部样本集的平均误差）
def computeCost(X, y, w):
    inner = np.power(((X * w.T) - y), 2)
    return np.sum(inner) / (2 * X.shape[0])
'''
data.insert(0, 'Ones', 1) # 在训练集上添加一列，可用向量化的解决方案来计算代价和梯度
print(data)

# 变量初始化
cols = data.shape[1]
X = data.iloc[:,:cols-1] # X是所有行，去掉最后一列
y = data.iloc[:,cols-1] # Y是所有行，只保存最后一列
print(X.head())
print(y.head())
# 代价函数是numpy矩阵，需转换X和Y,才可以使用，还需初始化w
X = np.matrix(X.values)
y = np.matrix(y.values)
w = np.matrix(np.array([0, 0]))
print(w) # w是一个(1,2)矩阵

# 计算代价函数
computeCost(X, y, w)
'''
# 批量梯度下降
def batch_gradientDescent(X, y, w, alpha, iters):
    temp = np.matrix(np.zeros(w.shape))
    parameters = int(w.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * w.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = w[0, j] - ((alpha / len(X)) * np.sum(term))

        w = temp
        cost[i] = computeCost(X, y, w)

    return w, cost
'''
# 初始化学习率α和要执行的迭代次数
alpha = 0.01
iters = 1000
# 运行梯度下降算法，将参数适用于训练集
g, cost = batch_gradientDescent(X, y, w, alpha, iters)
print(g)
print(computeCost(X, y, g))

# 绘制图形 更直观的看到拟合
x = np.linspace(data['人口'].min(), data['人口'].max, 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(data['人口'], data['收益'], label='训练数据')
ax.lengend(loc=2)
ax.set_xlabel('人口', fontsize=18)
ax.set_ylabel('收益',rotation=0, fontsize=18)
ax.set_title('预测收益和人口规模', fontsize=18)
plt.show()
'''
# 多变量线性回归
path = 'data/regress_data2.csv'
data2 = pd.read_csv(path)
print(data2.head())
# 特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())
# 重复第一部分的预处理步骤
data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
w2 = np.matrix(np.array([0,0,0]))
alpha = 0.01
iters = 1000
g2, cost2 = batch_gradientDescent(X2, y2, w2, alpha, iters)
computeCost(X2, y2, g2)

# 查看训练进程
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('迭代次数', fontsize=18)
ax.set_ylabel('代价', rotation=0, fontsize=18)
ax.set_title('误差和训练Epoch数', fontsize=18)
plt.show()


