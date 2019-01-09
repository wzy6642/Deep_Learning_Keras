# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:35:31 2019

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
from sklearn.model_selection import StratifiedKFold


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
# 将数据集分为train,test
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
cvscores = []
del dataset
# K折交叉验证
for train, validation in kfold.split(train_feature, train_label):
    # 创建模型
    model = Sequential()
    # 添加层
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型并自动评估模型
    model.fit(train_feature[train], train_label[train], epochs=150, batch_size=10, verbose=0)
    # 评估模型
    scores = model.evaluate(train_feature[validation], train_label[validation])
    print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)
# 输出均值和标准差
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))

