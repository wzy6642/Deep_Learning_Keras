# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:44:00 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
from math import pow, floor


dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
seed = 4
np.random.seed(seed)


# 计算学习率
def step_decay(epoch):
    init_lrate = 0.01
    drop = 0.05
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1+epoch) / epochs_drop)
    return lrate


def create_model(init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 定义优化器
    learningRate = 0.01
    momentum = 0.9      # 动量参数
    decay_rate = 0.0    # 学习率衰减
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


"""
keras.callbacks.LearningRateScheduler(schedule) 
该回调函数是用于动态设置学习率 
参数： 
● schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）
"""
lrate = LearningRateScheduler(step_decay)
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=1, callbacks=[lrate])
model.fit(x, Y)

