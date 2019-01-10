# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:50:02 2019

@author: wzy
"""
"""
年龄。(数字)
工作: 工作类型。(分类: 管理员，未知，失业者，经理，女佣，企业家，学生，蓝领，个体户，退休人员，技术人员，服务人员)
婚姻: 婚姻状况。(分类: 已婚，离婚，单身。注: 离婚指离异或丧偶)
教育: (分类: 未知，中学，小学，高中)
默认值: 是否具有信用?(二分类: 是，否)
余额: 年均余额，单位为欧元。(数字)
住房: 有住房贷款?(二分类: 是，否)
贷款: 有个人贷款?(二分类: 是，否)
联系人: 联系方式。(分类: 未知，固定电话号码，手机号码)
天: 最后一次联系日。(数字)
月: 最后一次联系的月份。(分类: Jan，Feb，Mar，...，Nov，Dec)
持续时间: 上次联系时间，以秒为单位。(数字)
广告系列: 在此广告系列和此客户的联系次数。(数字，包括上一个联系人)
pdays: 与客户上一次联系的间隔天数。(数字，-1表示以前没有联系过)
以前: 此广告系列之前和此客户的联系次数。(数字)
poutcome： 以前的营销活动的结果。(分类: 未知，其他，失败，成功)
客户是否订阅了定期存款?(二分类: 是，否)
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings


warnings.filterwarnings("ignore")
# 导入数据并将分类转化为数字
dataset = pd.read_csv('../data/bank.csv', delimiter=';')
# 将英文替换为数字
dataset['job'] = dataset['job'].replace(to_replace=['admin.', 'unknown', 'unemployed', 
       'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 
       'retired', 'technician', 'services'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['marital'] = dataset['marital'].replace(to_replace=['married', 'single', 'divorced'], 
       value=[0, 1, 2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown', 'secondary', 
       'primary', 'tertiary'], value=[0, 2, 1, 3])
dataset['default'] = dataset['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['housing'] = dataset['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['loan'] = dataset['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular', 'unknown', 'telephone'], 
       value=[0, 1, 2])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown', 'other', 'success', 
       'failure'], value=[0, 1, 2, 3])
dataset['month'] = dataset['month'].replace(to_replace=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataset['y'] = dataset['y'].replace(to_replace=['no', 'yes'], value=[0, 1])
train_label = dataset['y']
dataset.drop(labels=['y'], axis=1, inplace = True)
train_feature = dataset
# 数据标准化，去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
train_feature = StandardScaler().fit_transform(train_feature)
del dataset
# 设置随机数种子
seed = 4
np.random.seed(seed)


def create_model(units_list=[16], optimizer='adam', init='normal'):
    model = Sequential()
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=16, kernel_initializer=init))
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, train_feature, train_label, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
# 选择最优模型
param_grid = {}
param_grid['units_list'] = [[16], [30], [16, 8], [30, 8]]
# 调参
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(train_feature, train_label)
# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))

