# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:16:47 2019
YAML: 另一种标记语言
@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_yaml


dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
# 将标签转换成分类编码(每一列代表一类，是为1，否为0)
Y_labels = to_categorical(Y, num_classes=3)
seed = 4
np.random.seed(seed)


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # categorical:分类的  crossentropy:交叉熵
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = create_model()
model.fit(x, Y_labels, epochs=200, batch_size=5, verbose=0)
# evaluate: 评价
scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
# 将模型保存成YAML文件
model_yaml = model.to_yaml()
with open('../model/model.yaml', 'w') as file:
    file.write(model_yaml)
# 保存模型权重值
model.save_weights('../model/model.yaml.h5')
# 从YAML文件中加载模型
with open('../model/model.yaml', 'r') as file:
    model_yaml = file.read()
# 加载模型
new_model = model_from_yaml(model_yaml)
new_model.load_weights('../model/model.yaml.h5')
# 必须先编译模型
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 评估从YAML文件中加载的模型
scores = new_model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (new_model.metrics_names[1], scores[1]*100))

