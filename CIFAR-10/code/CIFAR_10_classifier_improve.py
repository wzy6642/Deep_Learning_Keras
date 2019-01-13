# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:56:26 2019

@author: wzy
"""
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard


batch_size = 128
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5
log_filepath = './nin'


# 对图像做标准化（Z-Score）
def normalize_preprocessing(x_train, x_validation):
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        # i代表第几通道，这里采用channels_last
        # 标准化（Z-Score）
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_validation[:, :, :, i] = (x_validation[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_validation


# 根据epoch改变learning rate
def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004


def build_model(x_train):
    model = Sequential()
    # 正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚。
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.01), input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    # strides:步长
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    # 这个层对减少参数和降低过拟合风险有很大的作用
    # GAP将最后一层的每个特征图进行一个平均池化操作，形成一个特征点，再将这些特征点组成的特征向量送入softmax进行分类.
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(seed=4)
    (X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train)
    y_validation = keras.utils.to_categorical(y_validation)
    X_train, X_validation = normalize_preprocessing(X_train, X_validation)
    model = build_model(X_train)
    print(model.summary())
    # TensorBoard是一个可视化工具，能够有效地展示Tensorflow在运行过程中的计算图、各种指标随着时间的变化趋势以及训练中使用到的数据信息。
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    # 该回调函数是用于动态设置学习率 
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=cbks, validation_data=(X_validation, y_validation), verbose=1)
    model.save('nin.h5')

