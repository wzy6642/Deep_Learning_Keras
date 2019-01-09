# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:22:11 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
# 设定随机数种子
seed = 4
np.random.seed(seed)
"""
author: zhenyu wu
time: 2019/01/09 16:26
function: 深度学习框架构建
params: 
    optimizer: 优化器
    init: 初始化权重的方案
return:
    model: 创建好的深度学习模型
"""
def create_model(optimizer='adam', init='glorot_uniform'):
    # 创建模型
    model = Sequential()
    # 添加层
    model.add(Dense(units=4, kernel_initializer=init, input_dim=4, activation='relu'))
    model.add(Dense(units=6, kernel_initializer=init, activation='relu'))
    # 多分类，分几类这里填几
    model.add(Dense(units=3, kernel_initializer=init, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f) ' % (results.mean()*100, results.std()))

