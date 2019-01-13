# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:40:37 2019

@author: wzy
"""
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras import backend
backend.set_image_data_format('channels_first')


(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(toimage(X_train[i]))
plt.show()
seed = 4
np.random.seed(seed)
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_train = X_train / 255.0
X_validation = X_validation / 255.0
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


def create_model(epochs=25):
    model = Sequential()
    # 图像边界信息丢失，即有些图像角落和边界的信息发挥作用较少。因此需要padding补0，左右对称补0
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate / epochs
    # 不使用牛顿动量的随机梯度下降
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


epochs = 25
model = create_model(epochs)
print(model.summary())
model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=1)
scores = model.evaluate(x=X_validation, y=y_validation, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))
# 打印网络结构
plot_model(model, to_file='../img/Flatten.png', show_shapes=True)

