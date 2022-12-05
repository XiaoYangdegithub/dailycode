# -*- coding:utf-8 -*-
# 名字为回归，但实际上是分类问题，将值映射到(0~1)的区间上，并且所有预测值的和为1，可理解为概率分布
# 会通过线性回归公式o = xw+b 计算出o 再根据softmax函数y = softmax(o)计算出预测值 使用交叉熵来衡量预测和标号的区别
import torch
import torchvision # 对计算视觉的一个库
from torch.utils import data
from torchvision import transforms # 对数据操作的一个模组
from d2l import torch as d2l

d2l.use_svg_display() # 用svg展示图片
trans = transforms.ToTensor() # 将图片转为tensor类型 -- 简单的预处理
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)  # 确定训练集
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False,transform=trans, download=True)  # 确定测试集

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)  # 1表示类型 28为长宽
def get_fashion_mnist_labels(labels):  # 返回Fashion-MNIST数据集的文本标签
    text_labels = [
        't_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'sneaker', 'bag', 'ankle boot']

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): # 设置一系列图片
    figsize = (num_cols * scale, num_rows * scale)
    # 画几张图片

x,y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(x.reshape(18,28, 28), 2, 9, titles=get_fashion_mnist_labels()) # 把图片拿出来


