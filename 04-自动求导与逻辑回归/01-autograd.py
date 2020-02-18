#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   01-autograd.py
@Time    :   2020/02/14 15:58:13
@Author  :   liululu 
@brief   :   自动求导机制
@Contact :   liululu827@163.com
@Desc    :   None
'''
'''
在autograd中，最常用的方法是torch.autograd.backward(),作用是自动求取各个节点的参数
'''


# here put the import lib
import torch
torch.manual_seed(10)


# ====================================== retain_graph：保存计算图 ==============================================
# 因为pytorch采用的是动态图机制，所以在每一次反向传播结束后，计算图就会释放掉
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True) 
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)  # 调用的就是torch.autograd.backward()方法 
    print(w.grad)
    y.backward()
    print(w.grad)

# ====================================== grad_tensors ：多梯度权重==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)


# ====================================== torch.autograd.gard ()方法：求取梯度==============================================
flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6 create_graph=True创建导数的计算图，用于高阶求导
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# ====================================== tips: 1 梯度不会自动清零，会叠加==============================================
flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()  # 下划线_操作代表inplace原位操作


# ====================================== tips: 2依赖于叶子结点的节点，他的requires_grad都是True ==============================================
flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 ==============================================
flag = True
# flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    a = a + torch.ones((1, ))
    print(id(a), a)

    a += torch.ones((1, ))  # 原位操作
    print(id(a), a)


# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)  # 叶子结点不能执行原位操作inplace

    y.backward()