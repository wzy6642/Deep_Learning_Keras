# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:28:05 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split


seed = 4
np.random.seed(seed)
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
x_train, x_increment, Y_train, Y_increment = train_test_split(x, Y, test_size=0.2, random_state=seed)
## 将标签转换成分类编码
#  to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。
## 其表现为将原有的类别向量转换为独热编码的形式。
Y_train_labels = to_categorical(Y_train, num_classes=3)


# rmsprop可缓解Adagrad算法学习率下降较快的问题
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = create_model()
model.fit(x_train, Y_train_labels, epochs=60, batch_size=5, verbose=2)
scores = model.evaluate(x_train, Y_train_labels, verbose=0)
print('Base %s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
model_json = model.to_json()
with open('../model/model.increment.json', 'w') as file:
    file.write(model_json)
model.save_weights('../model/model.increment.json.h5')
with open('../model/model.increment.json', 'r') as file:
    model_json = file.read()
new_model = model_from_json(model_json)
new_model.load_weights('../model/model.increment.json.h5')
# 新模型的编译过程应与原来模型保持一致(?)
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Y_increment_labels = to_categorical(Y_increment, num_classes=3)
# 新模型的训练过程可以与原来模型不一致
new_model.fit(x_increment, Y_increment_labels, epochs=60, batch_size=5, verbose=2)
scores = new_model.evaluate(x_increment, Y_increment_labels, verbose=0)
print('Increment %s: %.2f%%' % (new_model.metrics_names[1], scores[1]*100))

