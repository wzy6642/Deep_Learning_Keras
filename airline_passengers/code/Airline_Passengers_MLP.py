# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:45:05 2019

@author: wzy
"""
from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense


filename = '../data/international-airline-passengers.csv'
footer = 3
seed = 4
batch_size = 2
epochs = 400
look_back = 3


"""
函数说明：创建单层决策树的数据集
Parameters:
    dataset: 读取的数据集
Returns:
    dataX: 当月旅客数
    dataY: 下个月旅客数
Modify:
    2019-01-13
"""
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i+look_back, 0]
        dataX.append(x)
        y = dataset[i+look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


"""
函数说明：构造神经网络
Parameters:
    None
Returns:
    model: 网络结构
Modify:
    2019-01-13
"""
def build_model():
    model = Sequential()
    model.add(Dense(units=12, input_dim=look_back, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    np.random.seed(seed)
    # usecols=[1]: 读取第一列
    # engine: 使用分析引擎
    # skipfooter: 从文本尾部开始忽略(c引擎不支持)
    data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')
    train_size = int(len(dataset)*0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validation Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)
    # 构建通过训练数据集进行预测的图表数据
    # 依据给定形状和类型(shape[, dtype, order])返回一个新的空数组。
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train)+look_back, :] = predict_train
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train)+look_back*2+1: len(dataset)-1, :] = predict_validation
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')
    plt.show()
    
    