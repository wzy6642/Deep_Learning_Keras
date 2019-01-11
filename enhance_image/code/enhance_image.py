# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:07:12 2019

@author: wzy
"""
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
from matplotlib import pyplot as plt
import os
backend.set_image_data_format('channels_first')


(X_train, y_train), (X_validation, y_validation) = mnist.load_data()
# 图像显示
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')
# 图像特征标准化
# featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行，对输入的图片每个通道减去每个通道对应均值。
# featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
imgGen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
imgGen.fit(X_train)
plt.figure()
# ImageDataGenerator本身是一个迭代器，在请求时返回批次的图像样本。可以通过flow()函数配置batch_size,并准备数据生成器且生成对象
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
# ZCA白化 Zero-phase Component Analysis Whitening
# 假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性，所以用于训练时输入是冗余的。白化的目的就是降低输入的冗余性。
imgGen = ImageDataGenerator(zca_whitening=True)
imgGen.fit(X_train)
# 创建目录并保存图像
try:
    os.mkdir('../image')
except:
    print('The fold is exist!')
plt.figure()
# save_to_dir: 保存的路径
# save_prefix: 文件名的前缀
# save_format: 文件类型
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9, save_to_dir='../image', save_prefix='Jan', save_format='png'):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
# 图像旋转 旋转范围为90
imgGen = ImageDataGenerator(rotation_range=90)
imgGen.fit(X_train)
plt.figure()
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
# 图像移动
# width_shift_range(): 水平平移范围
# height_shift_range(): 垂直平移范围
imgGen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
imgGen.fit(X_train)
plt.figure()
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
# 图像反转
# horizontal_flip(): 水平反转
# vertical_flip(): 垂直翻转
imgGen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
imgGen.fit(X_train)
plt.figure()
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
# 图像透视
imgGen = ImageDataGenerator(shear_range=0.5)
imgGen.fit(X_train)
plt.figure()
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break

