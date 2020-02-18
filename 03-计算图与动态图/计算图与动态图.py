#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   计算图与动态图.py
@Time    :   2020/02/14 10:04:31
@Author  :   liululu 
@brief   :   y=(x + w)*(w + 1) 对w,x进行求导
@Contact :   liululu827@163.com
@Desc    :   None
'''

# here put the import lib
import torch
# 用户创建的节点称为叶子结点w,x，叶子结点可以通过tensor的is_leaf属性进行判断
# 设置叶子结点这个概念主要是为了节省内存，因为反向传播之后非叶子结点的梯度是会释放掉的
# 非叶子结点可以在执行反向传播之前通过retain_grad()保留，
w = torch.tensor([1.], requires_grad=True) 
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)     # retain_grad()
a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn： 创造非叶子结点的方法
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)

