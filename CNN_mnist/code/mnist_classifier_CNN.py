# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:32:47 2019

@author: wzy
"""
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend
# 返回默认图像数据格式约定 ('channels_first' 或 'channels_last')。
backend.set_image_data_format('channels_first')


seed = 4
np.random.seed(seed)
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')
X_train = X_train / 255
X_validation = X_validation / 255
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)


def create_model():
    model = Sequential()
    # 卷积----> 24*24
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # 池化----> 12*12
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 卷积----> 10*10
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # 池化----> 5*5
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    # 将(n,c,h,w)的特征图转化为(n,c*h*w)(必要操作)---->25
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=1)
score = model.evaluate(X_validation, y_validation, verbose=0)
print('CNN_Small: %.2f%%' % (score[1]*100))

