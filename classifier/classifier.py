# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:06:00 2021

@author: joeba
"""

from __future__ import print_function
import os
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dense, Flatten
from utils.MM_data_parser import DataParser
from src.networks.hed_reduced import hed
import keras
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
from keras import Model
import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical

def load_image(file_path, imShape):
    im = cv2.imread(file_path)
    im = cv2.resize(im, imShape)
    im = cv2.medianBlur(im, 9)
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
    s = Dense(1, activation = 'sigmoid')(u)
    model = Model(inputs=input, outputs=s)

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    classifier = hed('weights_robust_lighting_texture.h5')

    imageFileNames = []
    labels = []
    train_station = './pl_train_station/input_hed_images/'
    dest_station = './pl_train_station/pretty_good_set_output/'
    for img_name in os.listdir(dest_station):
        imageFileNames.append(dest_station + img_name)
        if 'genuine' in img_name:
            labels.append(1)
        else:
            labels.append(0)

    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(imageFileNames, labels, test_size=0.2)
    batchSize = 20
    train_batches = Inline_Generator(x_train, y_train, batchSize)
    test_batches = Inline_Generator(x_test, y_test, batchSize)

    weights_dir = './gibberish.h5'
    save_weights_to = './model_dir/classifiers/classify_conv_median_blur.h5'

    model = classify_build_conv(weights_dir)

    history = model.fit(x=train_batches,
                               steps_per_epoch = int(len(x_train) // batchSize),
                               epochs = 12,
                               verbose = 1,
                            validation_data=test_batches,
                            validation_steps=int(len(x_test) // batchSize)
                            )
    model.save_weights(save_weights_to)
    print(model.summary())
