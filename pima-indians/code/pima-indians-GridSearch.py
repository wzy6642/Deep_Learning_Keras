# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:32:43 2019

@author: wzy
"""
"""
数据集中的feature解释：
Number of times pregnent: 怀孕次数
Plasma glucose concentration a 2 hours in an oral glucose tolerance test: 2小时口服葡萄糖耐量试验中血浆葡萄糖浓度
Diastolic blood pressure(mm Hg): 舒张压
Triceps skin fold thickness(mm): 三头肌皮褶皱厚度
2-hour serum insulin(mu U/ml): 2小时血清胰岛素
Body mass index(weight in kg/(height in m)^2): 身体质量指数
Diabetes pedigree function: 糖尿病谱系功能
Age(years): 年龄
Class variable(0 or 1): 是否是糖尿病
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


"""
author: zhenyu wu
time: 2019/01/09 15:42
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
    # 添加层,input_dim为特征的个数
    model.add(Dense(units=12, kernel_initializer=init, input_dim=8, activation='relu'))
    model.add(Dense(units=8, kernel_initializer=init, activation='relu'))
    # 这里的1为输出多少列结果，二分类为1列
    model.add(Dense(units=1, kernel_initializer=init, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 选用初始化随机数种子，确保输出结果的可重复
seed = 4
np.random.seed(seed)
# 避免第一行变为列名
dataset = pd.read_csv('../data/pima-indians-diabetes.csv', header=None, names=list(np.arange(9)))
# 修改列名
dataset.columns = ['Number of times pregnent', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 
                   'Diastolic blood pressure(mm Hg)', 'Triceps skin fold thickness(mm)', '2-hour serum insulin(mu U/ml)', 
                   'Body mass index(weight in kg/(height in m)^2)', 'Diabetes pedigree function', 'Age(years)', 
                   'Class variable(0 or 1)']
train_label = dataset['Class variable(0 or 1)']
dataset.drop('Class variable(0 or 1)',axis=1, inplace=True)
train_feature = dataset
# 数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
train_feature = min_max_scaler.fit_transform(train_feature)
# 创建模型 for scikit-learn
# verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
model = KerasClassifier(build_fn=create_model, verbose=0)
del dataset
# 构建需要调参的参数
param_grid = {}
param_grid['optimizer'] = ['rmsprop', 'adam']
param_grid['init'] = ['glorot_uniform', 'normal', 'uniform']
param_grid['epochs'] = [50, 100, 150, 200]
param_grid['batch_size'] = [5, 10, 20]
# 调参
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(train_feature, train_label)
# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
# cv_results_为字典，可以转为DataFrame
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
for mean, std, param in zip(means, stds, params):
    # %r打印时能够重现它所代表的对象
    print('%f (%f) with: %r' % (mean, std, param))

