[TOC]

# Dataset与Dataloader

一般来说PyTorch中深度学习训练的流程是这样的：

1. 创建Dateset
2. Dataset传递给DataLoader
3. DataLoader迭代产生训练数据提供给模型

对应的一般都会有这三部分代码：

```python
# 1、创建Dateset(可以自定义)     
dataset = MyDataset
# 2、Dataset传递给DataLoader    
dataloader= torch.utils.data.DataLoader(dataset
                                        , batch_size=64
                                        , shuffle=False
                                        , num_workers=8)
# 3、DataLoader迭代产生训练数据提供给模型     
for i in range(epoch):
    for index,(img,label) in enumerate(dataloader):
        pass
```

## 一、自定义自己的数据集MyDataset

**torch.utils.data. dataset** 是一个表示数据集的抽象类。任何自定义的数据集都需要继承这个类并覆写相关方法。

所谓数据集，其实就是一个负责处理**索引(index)到样本(sample)**映射的一个类(class)。

pytorch提供两种数据集的创建方式：Map式数据集以及Iterable式数据集，这里着重介绍map式数据集。

一个Map式的数据集必须要重写__getitem__(self, index), **len**(self) 两个内建方法，用来表示从索引到样本的映射（Map）. 这样一个数据集dataset，举个例子，当使用dataset[idx]命令时，可以在你的硬盘中读取你的数据集中第idx张图片以及其标签（如果有的话）;len(dataset)则会返回这个数据集的容量。

> PyTorch 读取图片，主要是通过 Dataset 类，所以先简单了解一下 Dataset 类。 Dataset
> 类作为所有的 datasets 的基类存在，所有的 datasets 都需要继承它，类似于 C++中的虚基
> 类。  源码结构如下

```python 
class Dataset(object):
"""
An abstract class representing a Dataset.
All other datasets should subclass it. All subclasses should override
``__len__``, that provides the size of the dataset, and ``__getitem__``,
supporting integer indexing in range from 0 to len(self) exclusive.
"""
def __getitem__(self, index):
	raise NotImplementedError
    
def __len__(self):
	raise NotImplementedError
    
def __add__(self, other):
	return ConcatDataset([self, other])
```

这里重点看 getitem 函数， getitem 接收一个 index，然后返回图片数据和标签，这个
index 通常指的是一个 list 的 index，这个 list 的每个元素就包含了图片数据的路径和标签信
息 。

然而，如何制作这个 list 呢，通常的方法是将图片的路径和标签信息存储在一个 txt
中，然后从该 txt 中读取。那么读取自己数据的基本流程就是：

- 制作存储了图片的路径和标签信息的 txt

- 将这些信息转化为 list，该 list 每一个元素对应一个样本

- 通过 getitem 函数，读取数据和标签，并返回数据和标签  

```python
import os
'''
    为数据集生成对应的txt文件
'''

train_txt_path = os.path.join("..", "..", "Data", "train.txt")
train_dir = os.path.join("..", "..", "Data", "train")

valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")
valid_dir = os.path.join("..", "..", "Data", "valid")


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):         # 若不是png文件，跳过
                    continue
                label = img_list[i].split('_')[0]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(valid_txt_path, valid_dir)
```



```python
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):  # 根据index索引返回图片以及标签label
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):  # 查看样本的数量
        return len(self.data_info)
    
    # 一般在写一个方法的时候, 默认会接受一个self的形参, staticmethod 但是在调用这个方法的使用可能并没有传递任何一个参数
    @staticmethod  
    def get_img_info(data_dir):  # 获取数据的路径以及标签
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info
```

上面是一个实例，这个例子中将制作存储了图片的路径和标签信息的 txt利用了get_img_info(data_dir)函数来实现，那么一般的构建dataset子类的模板为：

```python 
# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
	def __init__(self, txt_path, transform = None, target_transform = None):
		fh = open(txt_path, 'r')
		imgs = []
		for line in fh:
			line = line.rstrip()
			words = line.split()
			imgs.append((words[0], int(words[1])))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
        
	def __getitem__(self, index):
		fn, label = self.imgs[index]
		img = Image.open(fn).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img, label

	def __len__(self):
		return len(self.imgs)
```

首先看看初始化，初始化中从我们准备好的 txt 里获取图片的路径和标签， 并且存储在 self.imgs， self.imgs 就是上面提到的 list，其一个元素对应一个样本的路径和标签，其实就是 txt 中的一行。

初始化中还会初始化 transform， transform 是一个 Compose 类型，里边有一个 list， list中就会定义了各种对图像进行处理的操作，可以设置减均值，除标准差，随机裁剪，旋转，翻转，仿射变换等操作。

在这里我们可以知道，一张图片读取进来之后，会经过数据处理（数据增强），最终变成输入模型的数据。这里就有一点需要注意， PyTorch 的数据增强是将原始图片进行了处理，并不会生成新的一份图片，而是“覆盖”原图，当采用 randomcrop 之类的随机操作时，每个 epoch 输入进来的图片几乎不会是一模一样的，这达到了样本多样性的功能。

然后看看核心的 getitem 函数：

第一行： self.imgs 是一个 list，也就是一开始提到的 list， self.imgs 的一个元素是一个 str,包含图片路径，图片标签，这些信息是从 txt 文件中读取

第二行：利用 Image.open 对图片进行读取， img 类型为 Image ， mode=‘RGB’

第三行与第四行： 对图片进行处理，这个 transform 里边可以实现 减均值，除标准差，随机裁剪，旋转，翻转，放射变换，等等操作

当 Mydataset 构建好，剩下的操作就交给 DataLoder，在 DataLoder 中，会触发Mydataset 中的 getiterm 函数读取一张图片的数据和标签，并拼接成一个 batch 返回，作为模型真正的输入。下一小节将会通过一个小例子，介绍 DataLoder 是如何获取一个 batch，以及一张图片是如何被 PyTorch 读取，最终变为模型的输入的。  

## 二、DataLoader 

在 MyDataset 中，主要获取图片的索引以及定义如何通过索引读取图片及其标签。但是要触发 MyDataset 去读取图片及其标签却是在数据加载器 DataLoder 中。  

Dataset负责建立索引到样本的映射，DataLoader负责以特定的方式从数据集中迭代的产生 一个个batch的样本集合。在enumerate过程中实际上是dataloader按照其参数sampler规定的策略调用了其dataset的getitem方法。

先看一下实例化一个DataLoader所需的参数，我们只关注几个重点即可

```python
DataLoader(dataset  # 定义好的Map式或者Iterable式数据集。
           , batch_size=1  # 一个batch含有多少样本 (default: 1)。
           , shuffle=False  # 每一个epoch的batch样本是相同还是随机 (default: False)。
           , sampler=None  # 决定数据集中采样的方法. 如果有，则shuffle参数必须为False。
           , batch_sampler=None  # 和 sampler 类似，但是一次返回的是一个batch内所有样本的index。和 batch_size, shuffle, sampler, and drop_last 三个参数互斥。
           , num_workers=0  # 多少个子程序同时工作来获取数据，多线程。 (default: 0)
           , collate_fn=None  # 合并样本列表以形成小批量。
           , pin_memory=False  # 如果为True，数据加载器在返回前将张量复制到CUDA固定内存中
           , drop_last=False  #  如果数据集大小不能被batch_size整除，设置为True可删除最后一个不完整的批处理。如果设为False并且数据集的大小不能被batch_size整除，则最后一个batch将更小。
           , timeout=0  # 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。 (default: 0)
           , worker_init_fn=None)  # 每个worker初始化函数 (default: None)
```

dataset 没什么好说的，很重要，需要按照前面所说的两种dataset定义好，完成相关函数的重写。

batch_size 也没啥好说的，就是训练的一个批次的样本数。

shuffle 表示每一个epoch中训练样本的顺序是否相同，一般True。

**采样器： ** **sampler**重点参数，采样器，是一个迭代器。PyTorch提供了多种采样器，用户也可以自定义采样器。所有sampler都是继承 `torch.utils.data.sampler.Sampler`这个抽象类。

## 三、具体流程

1、 划分数据集，设计自己的数据集mydataset（）

2、构建MyDataset实例：

```python
train_data = MyDataset(data_dir=train_dir  # 路径
                       , transform=train_transform  # 数据预处理
                      )
valid_data = MyDataset(data_dir=valid_dir
                       , transform=valid_transform
                      )
```

3、初始化 DataLoder 时，将 train_data 传入，从而使 DataLoder 拥有图片的路径  

```python
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
```

4、在一个 iteration 进行时，才读取一个 batch 的图片数据 enumerate()函数会返回可迭代数
据的一个“元素  ，在这里 data 是一个 batch 的图片数据和标签， data 是一个 list 

```python 
for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):        
```

在`for i, data in enumerate(train_loader): `  处利用步进的方式进入`train_loader`中，会转到class `DataLoader(object): `当中，进入`def __iter__(self):`函数，光标会处于`if self.num_workers == 0:  # 判断是否多进程`

以单进程为例，跳入`class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):`中的`def __next__(self):  `函数， #在单进程中最主要的是next，获取index和data









































