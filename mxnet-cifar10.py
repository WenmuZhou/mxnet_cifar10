# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午1:25
# @Author  : zhoujun

from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet.gluon.data import vision
from mxnet import nd
import net as net_collection
import utils
import numpy as np

input_str = '/data/datasets/cifar-10/'
batch_size = 32
image_size = 32
n_class = 10

train_test_augs = [
    image.RandomCropAug((image_size, image_size))
]
# 设定训练数据扩充形式
def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]),
                        brightness=0, contrast=0,
                        saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 设定测试数据扩充形式
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                     transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, transform=transform_test)
# 加载数据到迭代器
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')

# 设定超参数
ctx = utils.try_gpu()
num_epochs = 20
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 5
lr_decay = 0.1
# 初始化网络
net = net_collection.resnet18(10)
net.initialize(ctx=ctx, init=init.Xavier())
# net.load_params('models/19_0.90788_0.92952.params', ctx=ctx)
# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# 构造训练器
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': learning_rate, 'momentum': 0.9, 'wd': weight_decay})
# 执行训练过程
utils.train(train_data=train_data, test_data=test_data, net=net, loss=softmax_cross_entropy,
            trainer=trainer, ctx=ctx, num_epochs=num_epochs, print_batches=True,save_model='models')


