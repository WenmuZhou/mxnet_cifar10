# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午1:30
# @Author  : zhoujun
import mxnet as mx
from mxnet import autograd
from mxnet import nd
from mxnet import gluon
import time
import os


def try_gpu(gpu_index=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu_index)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=False, save_model=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if save_model and not os.path.exists(save_model):
        os.makedirs(save_model)
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time.time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1) == y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i + 1) % print_batches == 0:
                print("trained images %d. Loss: %f, Train acc %f" % (
                    n, train_loss / n, train_acc / m
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss / n, train_acc / m, test_acc, time.time() - start
        ))
        if save_model:
            net.save_params("%s/%d_%s_%s.params" % (save_model, epoch, train_acc / m, test_acc))


def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum(net(X).argmax(axis=1) == y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / n


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])
