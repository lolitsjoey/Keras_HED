import os
import cv2
import pandas as pd
import numpy as np
from keras import backend
import tensorflow as tf
import xgboost as xgb
from test_edge_model import tool_this_folder
from sklearn.decomposition import PCA
from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from hyperspec_classifier.hyperspec_classifier import compile_and_fit_model, build_rnn_model, build_and_fit_xgb_model, build_seq_model
from hyperspec_classifier.hyperspec_writer import write_hyper_spec, write_hyper_spec_just_feature
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras

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
        batch_x = batch_x[:,:,None]
        batch_y = to_categorical(batch_y, num_classes=self.num_segs*2)
        return batch_x, batch_y


def make_directories(dir_list):
    for item in dir_list:
        try:
            os.makedirs(item)
        except FileExistsError:
            continue

def train_test_val(data, labels, seeded=False):
    #    existing    0.2,    0.4
    x_train_and_test, x_val, y_train_and_test, y_val = train_test_split(data, labels, test_size=0.1, random_state=seeded)
    x_train, x_test, y_train, y_test = train_test_split(x_train_and_test, y_train_and_test, test_size=0.2, random_state=seeded)

    all = list(x_train.index) + list(x_test.index) + list(x_val.index)
    order = np.argsort(all)
    indexs = [idx for idx, path in enumerate(all) if path in x_test]

    return x_train, x_test, x_val, y_train, y_test, y_val, order, indexs


def get_hyper_batches(folder, batch_size, num_segs):
    hyper_csv = pd.read_csv(folder + os.listdir(folder)[0], index_col=0)
    labels = [int(note.split('_')[-1]) if 'genuine' in note else int(note.split('_')[-1]) + num_segs for note in hyper_csv.index]
    x_train, x_test, x_val, y_train, y_test, y_val, order, indexs = train_test_val(hyper_csv, labels, seeded=False)

    x_all = pd.concat( (x_train, x_val, x_test) )
    index_order = ['_'.join(img.split('_')[0:2]) for img in x_all.index]
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    train_batches = Inline_Generator(x_train, y_train, batch_size, num_segs)
    test_batches = Inline_Generator(x_test, y_test, batch_size, num_segs)
    val_batches = Inline_Generator(x_val, y_val, batch_size, num_segs)
    return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, index_order


def main(rgb_image_feat, scores_df):
    load_classifier_weights_dir = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/cwt.h5'
    save_classifier_weights_to = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/cwt.h5'
    retrain_classifier = True

    save_score_model_weights_to = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[-2]
    retrain_scoremodel = True

    new_tool_outputs = False
    spit_tool_output_here = '/'.join(rgb_image_feat.split('/')[0:-2]) + '/' + rgb_image_feat.split('/')[-2] + '_cwt/'

    make_directories([spit_tool_output_here, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1])])

    batchSize = 16
    epochs = 60
    rnn_neurons = 256
    dense_neurons = 64

    wavelet = 'gaus7'  # mother wavelet
    scales = np.arange(1, 32)  # range of scales
    num_segs = 3

    def cwt():
        if new_tool_outputs:
            folder_of_folders = 'D:/FRNLib/noteLibrary/just_100s/'
            fileString = 'genuine'
            write_hyper_spec_just_feature(folder_of_folders, spit_tool_output_here, num_segs, fileString, 'text' , scales, wavelet,
                             do_pca=True)

            folder_of_folders = 'D:/FRNLib/noteLibrary/CounterfeitHyper/'
            fileString = 'counterfeit'
            write_hyper_spec_just_feature(folder_of_folders, hyperspec_destination, num_segs, fileString, 'text' ,scales, wavelet,
                             do_pca=True)

        x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, index_order = get_hyper_batches(
            spit_tool_output_here, batchSize, num_segs)

        # define and train model
        trained_xgb_model, xgb_history = build_and_fit_xgb_model(x_train_cwt, y_train, x_test_cwt, y_test, n_depth=3,
                                                                 subsample=0.5, n_estimators=20, num_segs=num_segs)

        # make predictions for test data
        x_all = np.vstack((x_train_cwt, x_val_cwt, x_test_cwt))
        y_pred = trained_xgb_model.predict(x_all)
        # determine the total accuracy
        # y_train_1d = np.argmax(y_train, axis=1)
        # y_test_1d = np.argmax(y_test, axis=1)
        # y_val_1d = np.argmax(y_val, axis=1)
        y_all = np.hstack((y_train, y_val, y_test))
        accuracy = metrics.accuracy_score(y_all, y_pred)

        gen_cft_true = [1 if i < num_segs else 0 for i in y_all]
        gen_cft_pred = [1 if i < num_segs else 0 for i in y_pred]
        fig = confusion_matrix(gen_cft_true, gen_cft_pred)
        print(fig)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # train cnn model

        model = build_seq_model(num_segs, len(scales), rnn_neurons, dense_neurons)

        return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, y_pred, model, index_order

    x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, y_pred, model, index_order = cwt()

    if retrain_classifier:
        history = model.fit(x=train_batches,
                            epochs=epochs,
                            verbose=1,
                            validation_data=val_batches
                            )
        model.save_weights(save_classifier_weights_to)

    print('Evaluating Model...')
    loss, accuracy = model.evaluate(test_batches)
    print('Accuracy: %.2f' % (accuracy * 100))

    x_all = np.vstack((x_train, x_val, x_test))
    x_all = x_all[:, :, None]
    y_all = np.hstack((y_train, y_val, y_test))
    y_pred = model.predict(x_all)
    y_pred = [np.argmax(i) for i in y_pred]

    gen_cft_true = [1 if i < num_segs else 0 for i in y_all]

    layer_name = [layer.name for layer in model.layers][-2]
    n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
    dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(x_all)))
    dense_output = get_dense_output_no_images(dense_output, x_all, gen_cft_true, model, layer_name, binary=False,
                                              write=True, do_pca=True)

    dense_output['prediction'] = y_pred

    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                        load=not retrain_scoremodel,
                                                                                        load_name=load_score_model_weights_dir,
                                                                                        number_of_searches=2000,
                                                                                        save_name=save_score_model_weights_to)

    scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    scores_df[rgb_image_feat.split('/')[-2] + '_cwt_scores'] = [np.nan] * len(scores_df)
    index_order = [label.replace('C-26522','C26522') if 'C-26522' in label else label for label in index_order] #TODO GO BACK AND UNFUDGE THIS
    for ii, index_val in enumerate(index_order):
        scores_df.loc[index_val][rgb_image_feat.split('/')[-2] + '_cwt_scores'] = scores_in_order_of_index_order[ii]

    return scores_df
if __name__ == '__main__':
    main()

