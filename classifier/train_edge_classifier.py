# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:06:00 2021

@author: joeba
"""

from __future__ import print_function
import os
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dense, Flatten
import keras
from keras import Model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical

def prepare_data(img_folder_to_classify):
    imageFileNames = []
    truth = []
    for img_name in os.listdir(img_folder_to_classify):
        imageFileNames.append(img_folder_to_classify + img_name)
        if 'genuine' in img_name:
            truth.append(1)
        else:
            truth.append(0)
    truth = np.array(truth)

    x_train, x_test, y_train, y_test = train_test_split(imageFileNames, truth, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    return x_train, x_test, x_val, y_train, y_test, y_val

def grab_batches_with_generator(x_train, x_test, x_val, y_train, y_test, y_val, batchSize):
    train_batches = Inline_Generator(x_train, y_train, batchSize)
    val_batches = Inline_Generator(x_val, y_val, batchSize)
    test_batches = Inline_Generator(x_test, y_test, batchSize)

    return train_batches, test_batches, val_batches

def load_image(file_path, imShape):
    im = cv2.imread(file_path,1)
    im = cv2.resize(im, imShape)
    im = ((im - np.mean(im))/np.std(im))
    im = cv2.GaussianBlur(im,(9,9),0)
    return im

class Inline_Generator(keras.utils.Sequence):
    def __init__(self, imageFileNames, labels, batchSize):
        self.imageFileNames = imageFileNames
        self.labels = labels
        self.batchSize = batchSize

    def __len__(self):
        return (np.ceil(len(self.imageFileNames) / float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.imageFileNames[idx * self.batchSize: (idx + 1) * self.batchSize]
        batch_y = self.labels[idx * self.batchSize: (idx + 1) * self.batchSize]

        return np.array([load_image(f, (480, 480)) for f in batch_x]) / 255.0, np.array(batch_y)

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
    print(model.summary())
    return model


