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

from hyperspec_classifier.hyperspec_classifier import get_batches_pca_style, compile_and_fit_model, build_rnn_model, build_and_fit_xgb_model
from hyperspec_classifier.hyperspec_writer import write_hyper_spec
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with tf.device('CPU:0'):
        #load_classifier_weights_dir = './hyperspec_classifier/learning_to_learn_save.h5'
        load_classifier_weights_dir = './hyperspec_classifier/gibberish.h5'
        save_classifier_weights_to = './hyperspec_classifier/gibberish.h5'

        retrain_classifier = True
        retrain_scoremodel = True

        save_score_model_weights_to = './GAM_model/models/new_hyper_pca_att'
        load_score_model_weights_dir = './GAM_model/models/new_hyper_pca_att'

        layer_name = 'dense_1'
        batchSize = 24
        epochs = 20
        rnn_neurons = 40
        dense_neurons = 10

        rgb_image_feat = './hyperspec_classifier/hyperspec_inkwell_pca/rgb/'
        images_scored = './hyperspec_classifier/hyperspec_inkwell_pca/cwt/'

        #morl is best so far, gaus3 is interesting, gaus6 is the best for offset distinguish, gaus7/gaus8 scores the best consistently
        wavelet = 'mexh'  # mother wavelet
        scales = np.arange(1, 65, 0.25)  # range of scales
        num_segs = 4
        rewrite_hyperspec = False
        hyperspec_destination = './hyperspec_classifier/hyperspec_inkwell_pca/'

        def cwt():
            if rewrite_hyperspec:
                folder_of_folders = 'D:/FRNLib/noteLibrary/just_100s/'
                fileString = 'genuine'
                write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet, do_pca=True)

                folder_of_folders = 'D:/FRNLib/noteLibrary/CounterfeitHyper/'
                fileString = 'counterfeit'
                write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet, do_pca=True)


            x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches = get_batches_pca_style(
                                                                                                            hyperspec_destination, batchSize)

            # define and train model
            trained_xgb_model, xgb_history = build_and_fit_xgb_model(x_train_cwt, y_train, x_test_cwt, y_test, n_depth=10,
                                                                     subsample=0.5, n_estimators=200, num_segs=num_segs)

            # make predictions for test data
            y_pred = trained_xgb_model.predict(x_test_cwt)
            # determine the total accuracy
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
            y_val = np.argmax(y_val, axis=1)
            accuracy = metrics.accuracy_score(y_test, y_pred)

            gen_cft_true = [1 if i < num_segs else 0 for i in y_test]
            gen_cft_pred = [1 if i < num_segs else 0 for i in y_pred]
            fig = confusion_matrix(gen_cft_true, gen_cft_pred)
            print(fig)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            # train cnn model

            return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, trained_xgb_model

        x_train, x_test, x_val, y_train, y_test, y_val, trained_xgb_model = cwt()

        x_all = np.vstack((x_train, x_val, x_test))
        y_all = np.hstack((y_train, y_val, y_test))
        y_pred = trained_xgb_model.predict(x_all)
        fig = confusion_matrix(y_all, y_pred)
        print(fig)
        dense_output = pd.DataFrame(x_all)
        dense_output['prediction'] = y_pred
        dense_output['truth'] = y_all

        ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                            pca_quality=0.99,
                                                                                            genuine_classes = [0,1,2,3],
                                                                                            load= not retrain_scoremodel,
                                                                                            load_name=load_score_model_weights_dir,
                                                                                            number_of_searches=10000,
                                                                                            save_name=save_score_model_weights_to)

        write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, images_scored=images_scored,
                                                                 rgb_dir=rgb_image_feat, genuine_classes = [0,1,2,3])






