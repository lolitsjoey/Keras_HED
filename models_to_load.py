
import os
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, Input, MaxPooling2D, MaxPooling1D, Dense, Flatten, SimpleRNN, concatenate
import keras
from keras import Model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.feature import local_binary_pattern
from keras.utils import to_categorical

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
    return model


def fedSealModel(weights_dir):
    return tf.keras.models.load_model(weights_dir)


def classify_build_conv(weights_dir):
    input = Input(shape=(480, 480, 3))
    j = Conv2D(16, 3, 3, activation='relu')(input)
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

def classify_build_conv_6d(weights_dir):
    input = Input(shape=(480, 480, 6))
    j = Conv2D(16, 3, 3, activation='relu')(input)
    o = MaxPooling2D()(j)
    e = Conv2D(16, 3, 3, activation='relu')(o)
    r = MaxPooling2D()(e)
    ee = Flatten()(r)
    rr = Dense(169, activation='relu')(ee)
    u = Dense(20, activation='relu')(rr)
    #s = Dense(1, activation = 'sigmoid')(u)
    s = Dense(2, activation='softmax')(u)
    model = Model(inputs=input, outputs=s)

    if os.path.isfile(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded: {}'.format(weights_dir))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        assert len(hist) == self.numPoints + 2
        return hist