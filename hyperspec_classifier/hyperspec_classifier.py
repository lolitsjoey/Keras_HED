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
from keras.layers import Dense, Input, SimpleRNN, Embedding, GRU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from hyperspec_classifier.hyperspec_writer import write_hyper_spec
import xgboost as xgb

def get_batches(hyperspec_destination, wavelet, batch_size, scales, num_segs, do_pca=False, plot=False):
    search_string = 'genuine'
    gen_segs = parse_hyperspec_lists(hyperspec_destination, search_string, num_segs, plot=plot)
    gen_labels = []
    all_note_labels = []
    for idx, seg in enumerate(gen_segs.keys()):
        gen_labels = gen_labels + [int(str(seg).split('_')[-1])]
        all_note_labels += [seg]

    search_string = 'counterfeit'
    cft_segs = parse_hyperspec_lists(hyperspec_destination, search_string, num_segs, plot=plot)
    cft_labels = []
    for idx, seg in enumerate(cft_segs.keys()):
        cft_labels = cft_labels + [num_segs + int(str(seg).split('_')[-1])]
        all_note_labels += [seg]

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
    x_lab_train, x_lab_test, y_lab_train, y_lab_test = train_test_split(all_note_labels, all_labels, test_size=0.2, random_state=42)
    x_lab_train, x_lab_val, y_lab_train, y_lab_val = train_test_split(x_lab_train, y_lab_train, test_size=0.2, random_state=42)
    index_order = x_lab_train + x_lab_val + x_lab_test
    # amount of pixels in X and Y
    rescale_size = 64
    # determine the max scale size
    n_scales = len(scales)

    n_samples = x_train.shape[0]
    n_signals = x_train.shape[2]

    x_train_cwt = create_cwt_images(x_train, rescale_size, scales,  wavelet)
    x_test_cwt = create_cwt_images(x_test, rescale_size,  scales, wavelet)
    x_val_cwt = create_cwt_images(x_val, rescale_size, scales, wavelet)

    train_batches = Inline_Generator(x_train_cwt, y_train, batch_size, num_segs)
    test_batches = Inline_Generator(x_test_cwt, y_test, batch_size, num_segs)
    val_batches = Inline_Generator(x_val_cwt, y_val, batch_size, num_segs)

    return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, index_order

def get_batches_pca_style(hyperspec_destination, batch_size):
    counter_cwt = pd.read_csv(hyperspec_destination + '/pca_frame_counterfeit.csv', index_col=0)
    genuine_cwt = pd.read_csv(hyperspec_destination + '/pca_frame_genuine.csv', index_col=0)
    counter_cwt_labels = np.array([int(row.split('_')[2].split('.')[0]) for row in counter_cwt.index])
    genuine_cwt_labels = np.array([int(row.split('_')[2].split('.')[0]) for row in genuine_cwt.index])
    counter_cwt_labels = counter_cwt_labels + max(genuine_cwt_labels) + 1
    all_arrays = np.array(pd.concat((counter_cwt, genuine_cwt)))
    all_labels = np.hstack((counter_cwt_labels, genuine_cwt_labels))
    all_labels = to_categorical(all_labels)
    x_train, x_test, y_train, y_test = train_test_split(all_arrays, all_labels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    train_batches = Inline_Generator(x_train, y_train, batch_size, num_segs)
    test_batches = Inline_Generator(x_test, y_test, batch_size, num_segs)
    val_batches = Inline_Generator(x_val, y_val, batch_size, num_segs)

    return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches

def compile_and_fit_model(model, train_batches, val_batches, n_epochs):
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    return model

class Inline_Generator(keras.utils.Sequence):
    def __init__(self, X_train, y_train, batch_size, num_segs):
        self.x_train = X_train
        self.y_train = np.array(y_train)
        self.batchSize = batch_size
        self.num_segs = num_segs

    def __len__(self):
        return (np.ceil(len(self.x_train) / float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.x_train[idx * self.batchSize: (idx + 1) * self.batchSize]
        batch_y = self.y_train[idx * self.batchSize: (idx + 1) * self.batchSize]

        batch_y = to_categorical(batch_y, self.num_segs*2)
        return batch_x, batch_y

def build_rnn_model(num_segs, feature_vec_length, rnn_neurons, dense_neurons):
    model = keras.Sequential()
    inn = Input(shape=(feature_vec_length,1))
    model.add(inn)
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(SimpleRNN(rnn_neurons))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dense(int(num_segs*2), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_seq_model(num_segs, feature_vec_length, rnn_neurons, dense_neurons):
    model = keras.Sequential()
    inn = Input(shape=(feature_vec_length,1))
    model.add(inn)
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(GRU(rnn_neurons))
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(Dense(dense_neurons))
    #model.add(SimpleRNN(rnn_neurons))
    model.add(Dense(32))
    model.add(Dense(int(num_segs*2), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    return model





def build_and_fit_xgb_model(X_train, y_train, X_test, y_test, X_val, y_val, n_depth, subsample, n_estimators, num_segs):
    from sklearn.decomposition import PCA
    x_train_pca = []
    for xx in X_train:
        pca = PCA(n_components=1)
        x_train_pca.append(pca.fit_transform(xx[:,:,0]))
    x_train_pca = np.array(x_train_pca)[:,:,0]

    x_test_pca = []
    for xx in X_test:
        pca = PCA(n_components=1)
        x_test_pca.append(pca.fit_transform(xx[:, :, 0]))
    x_test_pca = np.array(x_test_pca)[:,:,0]

    x_val_pca = []
    for xx in X_val:
        pca = PCA(n_components=1)
        x_val_pca.append(pca.fit_transform(xx[:, :, 0]))
    x_val_pca = np.array(x_val_pca)[:, :, 0]

    xgb_model = xgb.XGBClassifier(max_depth=n_depth,
                              objective='multi:softmax', # error evaluation for multiclass training
                              num_class=num_segs*2,
                              subsample=subsample, # randomly selected fraction of training samples that will be used to train each tree.
                              use_label_encoder=False,
                              n_estimators=n_estimators)
    eval_set = [(x_train_pca, y_train), (x_test_pca, y_test)]
    history = xgb_model.fit(x_train_pca, y_train, eval_metric=["merror"], eval_set=eval_set,verbose=True)
    return xgb_model, history, np.vstack((x_train_pca, x_val_pca, x_test_pca))














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













