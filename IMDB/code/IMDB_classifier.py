# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:12:33 2019

@author: wzy
"""
from keras.datasets import imdb
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D


np.random.seed(seed=4)
top_words = 5000        # 只关注于前5000个单词
max_words = 500         # 每句话最多包含500个单词，多于500的部分被截断，少于500的句子用0补齐
out_dimension = 32
batch_size = 128
epochs = 2


def create_model():
    model = Sequential()
    # Embedding层只能作为模型的第一层
    # input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1 
    # output_dim：大于0的整数，代表全连接嵌入的维度 
    # input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。 
    model.add(Embedding(top_words, out_dimension, input_length=max_words))
    # 一维卷积
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # 一维池化
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    (X_train, y_train), (X_validation, y_validation) = imdb.load_data(num_words=top_words)

    # 合并训练数据集和评估数据集，a和b变为a上b下
    x = np.concatenate((X_train, X_validation), axis=0)
    y = np.concatenate((y_train, y_validation), axis=0)
    print('x shape is %s, y shape is %s' % (x.shape, y.shape))
    # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
    print('Classes: %s' % (np.unique(y)))
    # hstack会将多个value(value_one, value_two)的相同维度的数值组合在一起，并以同value同样的数据结构返回numpy数组
    print('Total words: %s' % len(np.unique(np.hstack(x))))
    # 保存每一条评论的单词数量
    result = [len(word) for word in x]
    print('Mean: %.2f words (STD: %.2f)' % (np.mean(result), np.std(result)))
    plt.subplot(121)
    plt.boxplot(result)
    plt.subplot(122)
    plt.hist(result)
    plt.show()


    # maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.在命名实体识别任务中，主要是指句子的最大长度
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_validation = sequence.pad_sequences(X_validation, maxlen=max_words)
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs, verbose=1)

