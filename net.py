# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午1:32
# @Author  : zhoujun
from mxnet.gluon import nn
from mxnet import nd


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                   strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                       strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


def resnet18(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            Residual(256, same_shape=False),
            Residual(256),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes)
        )
    return net

def resnet10(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes)
        )
    return net

def alexnet(num_classes):
    '''
    net 224*224images
    :param num_classes:
    :return:
    '''
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            # 第一阶段
            nn.Conv2D(channels=96, kernel_size=11,
                      strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第二阶段
            nn.Conv2D(channels=256, kernel_size=5,
                      padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第三阶段
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=256, kernel_size=3,
                      padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第四阶段
            nn.Flatten(),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            # 第五阶段
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            # 第六阶段
            nn.Dense(10)
        )
    return  net

def vgg_block(num_convs, channels):
    out = nn.HybridSequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg(num_classes):
    architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    out = nn.HybridSequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))

    net = nn.HybridSequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        net.add(
            out,
            nn.Flatten(),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            nn.Dense(num_classes))
    return net

class Inception_v1(nn.HybridBlock):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception_v1, self).__init__(**kwargs)
        # path 1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1,
                                   activation='relu')
        # path 2
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1,
                                   activation='relu')
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                   activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1,
                                   activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                   activation='relu')
        # path 4
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1,
                                      strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1,
                                   activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)

class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outer most Sequential
        with self.name_scope():
            # block 1
            b1 = nn.HybridSequential()
            b1.add(
                nn.Conv2D(64, kernel_size=7, strides=2,
                          padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            # block 2
            b2 = nn.HybridSequential()
            b2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 3
            b3 = nn.HybridSequential()
            b3.add(
                Inception_v1(64, 96, 128, 16,32, 32),
                Inception_v1(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 4
            b4 = nn.HybridSequential()
            b4.add(
                Inception_v1(192, 96, 208, 16, 48, 64),
                Inception_v1(160, 112, 224, 24, 64, 64),
                Inception_v1(128, 128, 256, 24, 64, 64),
                Inception_v1(112, 144, 288, 32, 64, 64),
                Inception_v1(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 5
            b5 = nn.HybridSequential()
            b5.add(
                Inception_v1(256, 160, 320, 32, 128, 128),
                Inception_v1(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )
            # block 6
            b6 = nn.HybridSequential()
            b6.add(
                nn.Flatten(),
                nn.Dense(num_classes)
            )
            # chain blocks together
            self.net = nn.HybridSequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out