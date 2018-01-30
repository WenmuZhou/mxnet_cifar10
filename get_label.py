# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午2:24
# @Author  : zhoujun

import os

data_dir = '/data/datasets/cifar-10/test/'
train_txt = '/data/datasets/cifar-10/test.txt'
dirs = os.listdir(data_dir)

with open(train_txt,mode='w',encoding='utf8') as xt:
    for dir_i in dirs:
        files = os.listdir(data_dir + dir_i)
        for i in files:
            xt.write(data_dir + dir_i + '/' + i + ' ' + dir_i + '\n')