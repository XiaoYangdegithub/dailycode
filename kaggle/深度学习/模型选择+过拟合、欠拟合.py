import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 使用一个三阶多项式生成训练和测试数据的标签
max_degree = 20  # 输入的特征数为20
n_train, n_text = 100, 100  # 训练集和测试集（验证集）的容量
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 20个w的值，其中前四个已经给出，其余均为0，为噪音

features = np.random.normal(size=(n_train + n_text, 1))
np.random.shuffle(features)  # 打乱顺序
ploy_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # 求feature的max.degree次方
for i in range(max_degree):
    ploy_features[:, i] /= math.gamma(i + 1)  # 计算函数中传递的数字的伽玛值
labels = np.dot(ploy_features, true_w)  # 矩阵乘法进行点积
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, ploy_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, ploy_features, labels]]

# print(features[:2], ploy_features[:2, :], labels[:2])
# 实现一个函数来评估模型在给定数据集上的损失
def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum, l.numel())
    return metric[0] / metric[1]

# 模型训练
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)  # torch.optim是一个实现了各种优化算法的库,SGD是随机梯度下降
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs],
                            ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss)),
                         evaluate_loss(net, test_iter, loss))
        print('weight:', net[0].weight.data.numpy())
'''
train(ploy_features[:n_train, :4], ploy_features[n_train:, :4],
      labels[:n_train], labels[n_train])

train(ploy_features[:n_train, :2], ploy_features[n_train:, :2],
      labels[:n_train], labels[n_train])  # 欠拟合，只用了前两行数据

train(ploy_features[:n_train, :], ploy_features[n_train:, :],
      labels[:n_train], labels[n_train], num_epochs=1500)  # 过拟合，用了所有列包括噪音

'''


