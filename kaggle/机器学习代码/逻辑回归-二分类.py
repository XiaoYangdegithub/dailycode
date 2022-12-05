# 将逻辑回归应用于二分类问题，并添加正则化项增强鲁棒性 -- 根据学生两次测试的分数预测是否录取
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
path = 'data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())
print(data.shape)
'''
'''
# 构建散点图直观表现是否被录取
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
fig, ax = plt.subplots(figesize=(12, 8))
ax.scatter(positive['Exam 1'],
           positive['Exam 2'],
           s=50,
           c='b',
           marker='o',
           label='Admitted')
ax.scatter(negative['Exam 1'],
           negative['Exam 2'],
           s=50,
           c='r',
           marker='x',
           label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
'''
'''
# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# 定义代价函数评估结果
def cost(w, X, y):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * w.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * w.T)))
    return np.sum(first - second) / len(X)
# 数据处理
data.insert(0, 'Ones', 1)   # 在第0列添加表头名为ONES的一列且列上所有值均为1,方便使用向量化的解决方案来计算代价和梯度
cols = data.shape[1]  # shape[1]则是取列的数量,shape[0]是取行的数量
X = data.iloc[:, 0:cols - 1]  # 所有行去掉最后一列
y = data.iloc[:, cols - 1:cols]  # 所有行保留最后一列

X = np.array(X.values)
y = np.array(y.values)
w = np.zeros(3)  # 初始化w
print(cost(w, X, y))

# 计算梯度 批量梯度下降
def gradient(w, X, y):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(w.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * w.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad  # 计算一个梯度的步长
# 使用opt.fmin_tnc（）函数来优化函数来计算成本和梯度参数
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=w, fprime=gradient, args=(X, y))
# func - 优化的目标函数。x0 - 初值。fprime - 提供优化函数func的梯度函数 args - 传入优化函数的参数
print(result)
print(cost(result[0], X, y))
# 模型预测
def predict(w, X):
    probability = sigmoid(X * w.T)
    return [1 if x >= 0.5 else 0 for x in probability]
w_min = np.matrix(result[0])
predictions = predict(w_min, X)
correct = [
    1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
    for (a, b) in zip(predictions, y)
]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
'''

# 使用正则化处理解决更复杂的数据集 -- 根据芯片的两次测试结果判断是否丢弃芯片
path = 'data/ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print(data2.head())
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'],
           positive['Test 2'],
           s=50,
           c='b',
           marker='o',
           label='Accepted')
ax.scatter(negative['Test 1'],
           negative['Test 2'],
           s=50,
           c='r',
           marker='x',
           label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
# 创建一组多项式特征
degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

print(data2.head())

# 正则化代价函数
def costReg(w, X, y, learningRate):
