# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:34:14 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
seed = 4
np.random.seed(seed)
def create_model(init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 定义优化器
    learningRate = 0.1
    momentum = 0.9      # 动量参数
    decay_rate = 0.005  # 学习率衰减
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=1)
model.fit(x, Y)

