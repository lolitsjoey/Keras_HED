import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from hyperspec_classifier.sort_signal_lists import parse_hyperspec_lists
import random
from hyperspec_classifier.hyperspec_tools import build_cnn_model, listToArray, create_cwt_images
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from hyperspec_classifier.hyperspec_writer import write_hyper_spec


def get_batches(hyperspec_destination, wavelet, batch_size, scales, num_segs, plot = False):
    search_string = 'genuine'
    #TODO dictionary style needs to be done.
    gen_segs = parse_hyperspec_lists(hyperspec_destination, search_string, num_segs, plot = plot)
    gen_labels = []
    for idx, seg in enumerate(gen_segs.keys()):
        gen_labels = gen_labels + [num_segs - int(str(seg).split(' ')[-1])]

    search_string = 'counterfeit'
    cft_segs = parse_hyperspec_lists(hyperspec_destination, search_string, num_segs, plot = plot)
    cft_labels = []
    for idx, seg in enumerate(cft_segs.keys()):
        cft_labels = cft_labels + [num_segs - int(str(seg).split(' ')[-1]) + max(gen_labels) + 1]

    all_labels = np.array(gen_labels + cft_labels)
    all_segs = np.full((len(all_labels),224),0, dtype=float)
    for idx,seg in enumerate(gen_segs):
        all_segs[idx,:] = gen_segs[seg]
    final_idx = idx
    for idx,seg in enumerate(cft_segs):
        all_segs[idx + final_idx + 1,:] = cft_segs[seg]

    all_segs = all_segs[:,:,None]
    x_train, x_test, y_train, y_test = train_test_split(all_segs, all_labels, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # amount of pixels in X and Y
    rescale_size = 64
    # determine the max scale size
    n_scales = 64

    x_train_cwt = create_cwt_images(x_train, rescale_size, scales,  wavelet)
    x_test_cwt = create_cwt_images(x_test, rescale_size,  scales, wavelet)
    x_val_cwt = create_cwt_images(x_val, rescale_size, scales, wavelet)
    train_batches = Inline_Generator(x_train_cwt, y_train, batch_size)
    test_batches = Inline_Generator(x_test_cwt, y_test, batch_size)
    val_batches = Inline_Generator(x_val_cwt, y_val, batch_size)

    return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches

def compile_and_fit_model(model, train_batches, val_batches, n_epochs):
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    history = model.fit(x=train_batches,
                        epochs=n_epochs,
                        verbose=1,
                        validation_data=val_batches
                        )

    return model, history

class Inline_Generator(keras.utils.Sequence):
    def __init__(self, X_train, y_train, batch_size):
        self.x_train = X_train
        self.y_train = np.array(y_train)
        self.batchSize = batch_size

    def __len__(self):
        return (np.ceil(len(self.x_train) / float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.x_train[idx * self.batchSize: (idx + 1) * self.batchSize]
        batch_y = self.y_train[idx * self.batchSize: (idx + 1) * self.batchSize]

        return batch_x, batch_y

if __name__ == '__main__':
    LABEL_NAMES = ["Intaglio", "Offset", "Paper", "Cft_Intaglio", "Cft_Offset", "Cft_Paper"]
    #morl is best so far, gaus3 is interesting, gaus6 is the best for offset distinguish, gaus7/gaus8 scores the best consistently
    wavelet = 'gaus7'  # mother wavelet
    scales = np.arange(1, 65, 0.25)  # range of scales
    num_segs = 3
    rewrite_hyperspec = False


    hyperspec_destination = './hyperspec_224/'
    if rewrite_hyperspec:
        folder_of_folders = 'D:/FRNLib/noteLibrary/New_Genuines/'
        fileString = 'genuine'
        write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString)

        folder_of_folders = 'D:/FRNLib/noteLibrary/CounterfeitHyper/'
        fileString = 'counterfeit'
        write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString)

    x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches = get_batches(hyperspec_destination, wavelet, 10, scales)

    input_shape = (x_train_cwt.shape[1], x_train_cwt.shape[2], x_train_cwt.shape[3])
    # create cnn model
    cnn_model = build_cnn_model("relu", input_shape)
    # train cnn model
    trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, train_batches, val_batches, 5)

    trained_cnn_model.save_weights('./learning_to_learn_save.h5')













