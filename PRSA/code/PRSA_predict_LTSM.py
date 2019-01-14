# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:07:56 2019

@author: wzy
"""
"""
日期
PM2.5污染物浓度
露点温度
温度
压力
风向
风速
累计的降雪小时数
累计的降水小时数
"""
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pandas import DataFrame


filename = '../data/pollution_original.csv'
batch_size = 72
epochs = 13
n_input = 1
n_train_hours = 365*24*4
n_validation_hours = 24*5


"""
author: zhenyu wu
time: 2019/01/14 10:18
function: 用户输入的日期和时间是字符串，要处理日期和时间，首先必须把str转换为datetime。
          转换方法是通过datetime.strptime()实现，需要一个日期和时间的格式化字符串
params: 
    x: str类型时间
return:
    datetime类型时间
"""
def prase(x):
    return datetime.strptime(x, '%Y %m %d %H')


# 读取数据并作数据预处理
def load_dataset():
    # parse_dates: 合并year,month,day,hour列作为一个日期列使用
    # index_col用作行索引的列编号或者列名
    dataset = read_csv(filename, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    dataset['pollution'].fillna(dataset['pollution'].mean(), inplace=True)
    return dataset


# 一日间隔作为label(这里是重点)
def convert_dataset(data, n_input=1, out_index=0, dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_input, 0, -1):
        # pandas DataFrame.shift()函数可以把数据移动指定的位数(此处为列，向后移动一列)
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    cols.append(df[df.columns[out_index]])
    names += ['result']
    result = pd.concat(cols, axis=1)
    result.columns = names
    if dropnan:
        result.dropna(inplace=True)
    return result


# 处理汉字编码问题
def class_encode(data, class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]
    values = pd.DataFrame(data).values
    for index in class_indexs:
        values[:, index] = encoder.fit_transform(values[:, index])
    return pd.DataFrame(values) if type(data) is DataFrame else values


def build_model(lstm_input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=lstm_input_shape, return_sequences=True))
    model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    dataset = load_dataset()
    groups = [0, 1, 2, 3, 5, 6, 7]
    plt.figure()
    i = 1
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(dataset.values[:, group])
        plt.title(dataset.columns[group], y=0.5, loc='right')
        i = i + 1
    plt.show()
    data = class_encode(dataset, 4)
    dataset = convert_dataset(data, n_input=n_input)
    values = dataset.values.astype('float32')
    train = values[:n_train_hours, :]
    val = values[-n_validation_hours:, :]
    x_train, y_train = train[:, :-1], train[:, -1]
    x_val, y_val = val[:, :-1], val[:, -1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_train = x_train.reshape(x_train.shape[0], n_input, x_train.shape[1])
    x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(lstm_input_shape)
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs, verbose=1)
    prediction = model.predict(x_val)
    plt.figure()
    plt.plot(y_val, color='blue', label='Actual')
    plt.plot(prediction, color='green', label='Prediction')
    plt.legend(loc='upper right')
    plt.show()

