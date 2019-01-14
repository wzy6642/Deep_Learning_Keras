# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:35:19 2019

@author: wzy
"""
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model


seed = 4
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2


def build_model():
    model = Sequential()
    model.add(Embedding(top_words, out_dimension, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 对输入数据以及LTSM中循环的展开存储单元进行dropout
    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    np.random.seed(seed)
    (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=top_words)
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_val = sequence.pad_sequences(x_val, maxlen=max_words)
    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    scores = model.evaluate(x_val, y_val, verbose=1)
    print('Accuracy: %.2f%%' % (scores[1]*100))
    # 打印网络结构
    plot_model(model, to_file='../img/Flatten.png', show_shapes=True)

