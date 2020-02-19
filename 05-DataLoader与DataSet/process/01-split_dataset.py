#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   01-split_dataset.py
@Time    :   2020/02/19 09:58:11
@Author  :   liululu
@brief   :   划分数据集，制作my_dataset
@Contact :   liululu827@163.com
@Desc    :   None
'''


# here put the import lib
import os
import random
# shutil.copy(source,destination)将source的文件拷贝到destination，两个参数都是字符串格式。
import shutil


def makedir(new_dir):

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)
    dataset_dir = os.path.join('..', 'data', 'RMB_data')
    split_dir = os.path.join('..', 'data', 'my_dataset')
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")
    # print(train_dir)

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    '''
    root 所指的是当前正在遍历的这个文件夹的本身的地址
    dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    '''
    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:  # dataset_dir下面的所有文件夹名
            imgs = os.listdir(os.path.join(root, sub_dir))  # sub_dir文件夹下所有的文件名
            imgs = list(
                filter(
                    lambda x: x.endswith('.jpg'),
                    imgs))  # sub_dir 文件夹下所有的.jpg文件
            # 打乱
            random.shuffle(imgs)
            img_count = len(imgs)
            # 分比例
            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))
            # 分训练集验证集与测试集
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)
                # print(out_dir)

                makedir(out_dir)
                # 将文件复制到目标文件夹中
                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)
            print(
                'Class:{}, train:{}, valid:{}, test:{}'.format(
                    sub_dir,
                    train_point,
                    valid_point -
                    train_point,
                    img_count -
                    valid_point))
