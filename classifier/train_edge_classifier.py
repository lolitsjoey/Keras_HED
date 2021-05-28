# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:06:00 2021

@author: joeba
"""

from __future__ import print_function
import os
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, Input, MaxPooling2D, MaxPooling1D, Dense, Flatten, SimpleRNN, concatenate
import keras
from keras import Model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical

def prepare_data(img_folder_to_classify, seed=False):
    imageFileNames = []
    truth = []
    for img_name in os.listdir(img_folder_to_classify):
        imageFileNames.append(img_folder_to_classify + img_name)
        if 'genuine' in img_name:
            truth.append(1)
        else:
            truth.append(0)
    truth = np.array(truth)

    #random state was 1
    if not seed:
        x_train, x_test, y_train, y_test = train_test_split(imageFileNames, truth, test_size=0.2)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    else:
        x_train, x_test, y_train, y_test = train_test_split(imageFileNames, truth, test_size=0.2, random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    return x_train, x_test, x_val, y_train, y_test, y_val

def grab_batches_with_generator(x_train, x_test, x_val, y_train, y_test, y_val, batchSize, blur=True, imShape=(480,480), gray=False, lbp=False, lbp_class=None, multichannel=False):
    if blur:
        train_batches = Inline_Generator(x_train, y_train, batchSize, imShape=imShape, gray=gray, blur=blur)
        val_batches = Inline_Generator(x_val, y_val, batchSize, imShape=imShape, gray=gray, blur=blur)
        test_batches = Inline_Generator(x_test, y_test, batchSize, imShape=imShape, gray=gray, blur=blur)
    elif multichannel:
        train_batches = Inline_Generator(x_train, y_train, batchSize, imShape=imShape, gray=gray, blur=blur, lbp=lbp, lbp_class=lbp_class, multichannel=multichannel)
        val_batches = Inline_Generator(x_val, y_val, batchSize, imShape=imShape, gray=gray, blur=blur, lbp=lbp, lbp_class=lbp_class, multichannel=multichannel)
        test_batches = Inline_Generator(x_test, y_test, batchSize, imShape=imShape, gray=gray, blur=blur, lbp=lbp, lbp_class=lbp_class, multichannel=multichannel)
    elif lbp:
        train_batches = Inline_Generator(x_train, y_train, batchSize, imShape, gray, lbp=lbp, lbp_class=lbp_class)
        val_batches = Inline_Generator(x_val, y_val, batchSize, imShape, gray, lbp=lbp, lbp_class=lbp_class)
        test_batches = Inline_Generator(x_test, y_test, batchSize, imShape, gray, lbp=lbp, lbp_class=lbp_class)
    else:
        train_batches = Inline_Generator(x_train, y_train, batchSize, imShape, gray=gray, blur=False)
        val_batches = Inline_Generator(x_val, y_val, batchSize, imShape, gray=gray, blur=False)
        test_batches = Inline_Generator(x_test, y_test, batchSize, imShape, gray=gray, blur=False)


    return train_batches, test_batches, val_batches

class Inline_Generator(keras.utils.Sequence):
    def __init__(self, imageFileNames, labels, batchSize, imShape=(480,480), gray=False, blur=False, lbp=False, lbp_class=None, multichannel=False):
        self.imageFileNames = imageFileNames
        self.labels = labels
        self.batchSize = batchSize
        self.imShape = imShape
        self.gray = gray
        self.blur = blur
        self.lbp = lbp
        self.local_binary_class = lbp_class
        self.multichannel = multichannel

    def __len__(self):
        return (np.ceil(len(self.imageFileNames) / float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.imageFileNames[idx * self.batchSize: (idx + 1) * self.batchSize]
        batch_y = self.labels[idx * self.batchSize: (idx + 1) * self.batchSize]
        if self.lbp & self.multichannel:
            batch_x1 = np.array([load_image_lbp(f[0], self.imShape, self.gray, self.local_binary_class)  for f in batch_x])
            batch_x2 = np.array([load_image_lbp(f[1], self.imShape, self.gray, self.local_binary_class)  for f in batch_x])
            batch_x_total = [batch_x1, batch_x2]
            batch_x_total = np.asarray(batch_x_total).astype('float32').transpose(1,2,0)
            return batch_x_total, batch_y
        elif self.blur:
            return np.array([load_image_no_blur(f, self.imShape, self.gray) for f in batch_x]) / 255.0, np.array(
                batch_y)
        elif self.lbp:
            batch_x = np.array([load_image_lbp(f, self.imShape, self.gray, self.local_binary_class) for f in batch_x])
            return batch_x[:, :, None], np.array(batch_y)
        else:
            return np.array([load_image(f, self.imShape,self.gray) for f in batch_x]) / 255.0, np.array(batch_y)

def load_image(file_path, imShape, gray):
    col = 1
    if gray:
        col = 0
    im = cv2.imread(file_path,col)
    im = cv2.resize(im, imShape)
    im = ((im - np.mean(im))/np.std(im))
    im = cv2.GaussianBlur(im,(9,9),0)
    if gray:
        return im[:, :, None]
    return im

def load_image_no_blur(file_path, imShape, gray):
    col = 1
    if gray:
        col = 0
    im = cv2.imread(file_path,col)
    im = cv2.resize(im, imShape)
    im = ((im - np.mean(im))/np.std(im))
    if gray:
        return im[:, :, None]
    return im

def load_image_lbp(file_path, imShape, gray, lbp_tool):
    col = 0
    im = cv2.imread(file_path,col)
    im = cv2.resize(im, imShape)
    hist = lbp_tool.describe(im)
    return hist


def classify_build(weights_dir):
    input = Input(shape=(480,480,3))
    i = Flatten()(input)
    h = Dense(500, activation='relu')(i)
    o = Dense(2, activation='softmax')(h)
    model = Model(inputs=input, outputs=o)

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))

    model.compile(optimizer=tf.optimizers.Adam(),
                      loss=tf.losses.CategoricalCrossentropy(),
                      metrics=[tf.metrics.CategoricalAccuracy()])
    return model

def classify_build_conv(weights_dir):
    input = Input(shape=(480,480,3))
    j = Conv2D(16,3,3, activation='relu')(input)
    o = MaxPooling2D()(j)
    e = Conv2D(16, 3, 3, activation='relu')(o)
    r = MaxPooling2D()(e)
    ee = Flatten()(r)
    rr = Dense(169, activation = 'relu')(ee)
    u = Dense(20, activation = 'relu')(rr)
    #s = Dense(1, activation = 'sigmoid')(u)
    s = Dense(2, activation='softmax')(u)
    model = Model(inputs=input, outputs=s)

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def classify_rnn(weights_dir, rnn_neurons, dense_neurons, feature_vec_length):
    model = keras.Sequential()
    inn = Input(shape=(feature_vec_length, 1))
    model.add(inn)
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(SimpleRNN(rnn_neurons))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def classify_rnn_2d(weights_dir, rnn_neurons, dense_neurons, feature_vec_length):
    chan = Input(shape=(feature_vec_length, 2))
    conv = Conv1D(filters=4, kernel_size=1, activation="tanh")(chan)
    maxpool = MaxPooling1D(pool_size=1, strides=2)(conv)  # Has shape (?, 8, 64)
    rnned = SimpleRNN(rnn_neurons)(maxpool)

    dense_1 = Dense(50, activation='relu')(rnned)
    dense_2 = Dense(dense_neurons, activation='relu')(dense_1)
    output = Dense(2, activation='softmax')(dense_2)
    model = Model(inputs=chan, outputs=output)

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

