# -*- coding:utf-8 -*-
import torch
x = torch.arange(4.0)
print(x)
# 在计算y关于x的梯度之前，需要一个地方来存储梯度
# 创建存储梯度
x.requires_grad_(True)  # 等价于x = torch.arange(4.0, requires_grad=True)
X = x.grad  # 默认值是None
# 计算y 针对2*xT*x求梯度
y = 2 * torch.dot(x, x)  # 也就是2*x的平方，y是一个标量
print(y)
# 通过反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4 * x)  # 此时已计算成功

# 在默认情况下，pytorch会累积梯度，所以我们需要清除之前的值
x.grad.zero_() # _表示写进内容 该操作表示把x的梯度清零
y = x.sum()  # 重新计算一个新的x函数，y是一个标量
y.backward()
print(x.grad)

# 当y是非标量时的解决办法 -- 把y转化为标量
x.grad.zero_()
y = x * x
# 此时的y是一个非标量 通过sum()计算将非标量转化为标量再进行backward()
y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
x.grad

# 把计算移动到记录的计算图之外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

