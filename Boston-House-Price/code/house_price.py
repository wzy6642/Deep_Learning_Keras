# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:47:55 2019

@author: wzy
"""
"""
CRIM: 城镇人均犯罪率
ZN: 住宅用地所占比例
INDUS: 城镇中非住宅用地所占比例
CHAS: 虚拟变量，用于回归分析
NOX: 环保指数
RM: 每栋住宅的房间数
AGE: 1940年以前建成的自住单位的比例
DIS: 距离5个波士顿的就业中心的加权距离
RAD: 距离高速公路的便利指数
TAX: 每一万元的不动产税率
PTRATIO: 城镇中教师和学生的比例
B: 城镇中黑人的比例
LSTAT: 地区中有多少房东属于低收入人群
MEDV: 自住房屋房价中位数
"""
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# 导入数据
dataset = datasets.load_boston()
x = dataset.data
Y = dataset.target
# 设定随机数种子
seed = 4
np.random.seed(seed)


def create_model(units_list=[13], optimizer='adam', init='normal'):
    # 构建模型
    model = Sequential()
    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))
    # 构建更多的隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))
    model.add(Dense(units=1, kernel_initializer=init))
    # 编译模型
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=1)
# 数据标准化，改进算法
steps = []
steps.append(('standardize', StandardScaler()))
steps.append(('mlp', model))
pipeline = Pipeline(steps)
# 设置算法评估基准
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# 注意，这里得出的results是负数，MSE指标要对其取相反数
results = cross_val_score(pipeline, x, Y, cv=kfold)
results = -results
print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))

