# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:46:19 2019

@author: wzy
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
seed = 4
np.random.seed(seed)
def create_model(init='glorot_uniform'):
    model = Sequential()
    # kernel_constraint：对主权重矩阵进行约束,max_norm(m=3)：最大模约束,即权重不超过3
    model.add(Dense(units=4, activation='relu', kernel_initializer=init, kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init, kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 定义优化器
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))

