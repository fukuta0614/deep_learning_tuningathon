# -*- coding: utf-8 -*-
from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2


def create_network(input_shape, output_shape):
    """
    Append reguralizers to ex02 model
    You can change reguralizers by `regularizer` option

    :param input_shape:
    :param output_shape:
    :param regularizer:
    :return:
    """
    lr = 1e-4
    model = Sequential()

    model.add(Convolution2D(32, 5, 5, border_mode='same',
                            input_shape=input_shape,
                            W_regularizer=l2(lr)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 5, 5, border_mode='same',
                            W_regularizer=l2(lr)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


####added
    model.add(Convolution2D(64, 5, 5, border_mode='same',
                            W_regularizer=l2(lr)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
######

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(lr)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    return model
