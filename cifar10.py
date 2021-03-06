# -*- coding: utf-8 -*-
from __future__ import print_function

"""
    Original Code is
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
"""
__author__ = 'ogata'

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import importlib
import click

# for CIFAR10 configurations
NUM_CLASSES = 10
IMG_ROWS, IMG_COLS = 32, 32  # input image dimensions
IMG_CHANNELS = 3  # the CIFAR10 images are RGB

LOG_DIR = './logs'


def load_datasets():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
    return X_train, Y_train, X_test, Y_test


def preprocess(X_train, X_test):
    X_train_normed = X_train.astype('float32') / 255.0
    X_test_normed = X_test.astype('float32') / 255.0
    return X_train_normed, X_test_normed


@click.command()
@click.argument('model_path')
@click.option('--batch_size', default=32, help='Number of batch size')
@click.option('--num_epoch', default=30, help='Number of epoch')
@click.option('--data_augmentation', default=False, is_flag=True,
              help='Enable data augmentation')
def run(model_path, batch_size, num_epoch, data_augmentation):
    print('==== params ====')
    print('model_path: {}, batch_size: {}, '
          'num_epoch: {}, data_augmentation: {}'
          .format(model_path, batch_size, num_epoch, data_augmentation))
    print('==== ====== ====')

    X_train, Y_train, X_test, Y_test = load_datasets()
    X_train, X_test = preprocess(X_train, X_test)

    mymodel = importlib.import_module(model_path)
    input_shape = X_train.shape[1:]
    output_shape = NUM_CLASSES
    model = mymodel.create_network(input_shape, output_shape)

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.02, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'],
                  )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=num_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[TensorBoard(log_dir=LOG_DIR), early_stopping])
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=num_epoch,
                            validation_data=(X_test, Y_test),
                            callbacks=[early_stopping])



if __name__ == '__main__':
    run()
