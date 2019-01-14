# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:21:59 2019

@author: wzy
"""
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


seed = 4
batch_size = 1
epochs = 100
filename = '../data/international-airline-passengers.csv'
footer = 3
look_back = 3


"""
函数说明：创建单层决策树的数据集
Parameters:
    dataset: 读取的数据集
Returns:
    dataX: 当月及当月前look_back月的旅客数
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
    """
    LSTM堆叠，在进行网络拓扑配置时，每个LSTM层之前的LSTM层必须返回序列，可以将LSTM层的
    return_sequences设为True实现这个功能
    """
    model.add(LSTM(units=4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(units=4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
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
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset)*0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)
    # 将输入转化成[样本，时间步长，特征]！！！！
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
    model = build_model()
    """
    每个训练批次之后，或者调用model.predict()或model.evaluate()函数之后，网络的状态
    都会重新设置。在Keras中，通过设置LSTM层的stateful为True，来保存LSTM层的内部状态，
    从而获得更好的控制。
    """
    for i in range(epochs):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
        mean_loss = np.mean(history.history['loss'])
        print('mean loss %.5f for loop %s' % (mean_loss, str(i)))
        # 重置网络状态
        model.reset_states()
    predict_train = model.predict(X_train, batch_size=batch_size)
    model.reset_states()
    predict_validation = model.predict(X_validation, batch_size=batch_size)
    # 反标准化
    predict_train = scaler.inverse_transform(predict_train)
    y_train = scaler.inverse_transform([y_train])
    predict_validation = scaler.inverse_transform(predict_validation)
    y_validation = scaler.inverse_transform([y_validation])
    train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
    print('Validation Score: %.2f RMSE' % validation_score)
    # 构建通过训练数据集进行预测的图表数据
    # 依据给定形状和类型(shape[, dtype, order])返回一个新的空数组。
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train)+look_back, :] = predict_train
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train)+look_back*2+1: len(dataset)-1, :] = predict_validation
    dataset = scaler.inverse_transform(dataset)
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')
    plt.show()

