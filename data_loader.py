# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午1:25
# @Author  : lyyang

from mxnet.gluon.data import vision
from mxnet import image
from mxnet import nd

train_test_augs = [
    image.RandomCropAug((image_size,image_size))
]

def transform(data, label, augs=None):
    data = data.astype('float32')
    if augs:
        for aug in augs:
            data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')



 # 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,transform=lambda X, y: transform(X, y, train_test_augs))
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,transform=lambda X, y: transform(X, y, train_test_augs))

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')