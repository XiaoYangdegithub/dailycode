# -*- coding:utf-8 -*-
import torch
from IPython import display
#from d2l import torch as d2l

batch_size = 256
#trian_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  #  每次随机读取256张图片

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition

# 验证softmax
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

# 实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])),w) + b)

# 实现交叉熵损失函数
# 创建数据y_hat包含两个样本在三个类别的预测概率，使用y作为y_hat中的概率索引
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)

# 预测值和真实值的比较
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)