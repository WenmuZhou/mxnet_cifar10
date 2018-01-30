# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午3:08
# @Author  : zhoujun
from mxnet import image
from mxnet import nd
import matplotlib.pyplot as plt
import cv2
import utils
import net as net_collection
import numpy as np

image_size = 32


def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def get_label(label_path):
    '''
    得到标签词典
    :param label_path: 标签文本路径
    :return:
    '''
    label = {}
    with open(label_path) as t:
        t_lines = t.readlines()
        for line in t_lines:
            params = line.split()
            label[params[0]] = params[1]
    return label

def predict_mxnet(net, ctx, fname, label):
    '''
    使用mxnet对图像进行预测
    :param net:训练好的模型
    :param ctx:数据context
    :param fname:图像路径
    :param label:标签词典
    :return:预测类别及概率
    '''
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())
        img = image.ForceResizeAug((image_size, image_size))(img)
    data, _ = transform_test(img, -1)
    data = data.expand_dims(axis=0)
    out = net(data.as_in_context(ctx))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    return '置信度=%f, 类别 %s' % (prob, label[str(pred)])


def predict_cv(net, ctx, fname, label):
    img = cv2.imread(fname)
    img = cv2.resize(img, (image_size, image_size))
    data, _ = transform(nd.array(img), -1)
    plt.imshow(data.transpose((1, 2, 0)).asnumpy() / 255)
    data = data.expand_dims(axis=0)
    out = net(data.as_in_context(ctx))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    print(prob, pred)
    return '置信度=%f, 类别 %s' % (prob, label[str(pred)])


if __name__ == '__main__':
    label_path = '/data/datasets/cifar-10/label.txt'
    image_path = '/data/datasets/cifar-10/test/9/9_11.jpg'
    label_dict = get_label(label_path)
    print(label_dict)
    ctx = utils.try_gpu()
    net2 = net_collection.resnet18(10)
    net2.hybridize()
    net2.load_params('models/11_0.87632_0.89242.params', ctx=ctx)
    print(predict_mxnet(net=net2, ctx=ctx, fname=image_path, label=label_dict))
    plt.imshow(plt.imread(image_path))
    plt.show()
