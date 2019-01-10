# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:14:47 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
Y_labels = to_categorical(Y, num_classes=3)
seed = 4
np.random.seed(seed)


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = create_model()
# 设置检查点
filepath = '../model/weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x, Y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0, callbacks=callback_list)
# 加载最优权重到新的模型
new_model = create_model()
new_model.load_weights(filepath='../model/weights.best.h5')
scores = new_model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (new_model.metrics_names[1], scores[1]*100))

