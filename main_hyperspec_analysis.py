import os
import cv2
import pandas as pd
import numpy as np
from keras import backend
import tensorflow as tf
from test_edge_model import tool_this_folder
import math
from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

from hyperspec_classifier.hyperspec_classifier import get_batches, compile_and_fit_model
from hyperspec_classifier.hyperspec_tools import build_cnn_model
from hyperspec_classifier.hyperspec_writer import write_hyper_spec_from_array

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from hyperspec_classifier.hyperspec_classifier import compile_and_fit_model, build_rnn_model, build_and_fit_xgb_model, build_seq_model

def make_directories(dir_list):
    for item in dir_list:
        try:
            os.makedirs(item)
        except FileExistsError:
            continue

def main(rgb_image_feat, scores_df, folder_index):
    load_classifier_weights_dir = './gibberish.h5'
    save_classifier_weights_to = './temp_models_in_progress/cwt' + '_' + rgb_image_feat.split('/')[-2] + '_classifier/cwt.h5'
    retrain_classifier = True

    retrain_classifier = True
    retrain_scoremodel = True

    save_score_model_weights_to = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './score_models_per_feature/cwt' + '_' + rgb_image_feat.split('/')[-2]
    retrain_scoremodel = True

    hyperspec_destination = '/'.join(rgb_image_feat.split('/')[0:-2]) + '/' + rgb_image_feat.split('/')[-2] + '_waves/'
    feature = rgb_image_feat.split('/')[-2]

    make_directories([hyperspec_destination, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1]), hyperspec_destination + 'cwt/', hyperspec_destination + 'rgb/'])

    batchSize = 24
    epochs = 10

    # morl is best so far, gaus3 is interesting, gaus6 is the best for offset distinguish, gaus7/gaus8 scores the best consistently
    wavelet = 'mexh'  # mother wavelet
    scales = np.arange(1, 65)  # range of scales
    num_segs = 4
    rewrite_hyperspec = True


    def cwt():
        if rewrite_hyperspec:
            folder_index = ['C26459', 'C26522', 'A20-2', 'A16', 'A13', '100 - new']
            for folder in folder_index:
                folder_of_folders = 'D:/FRNLib/HSI Features/' + folder + '-HSI/'
                fileString = 'counterfeit'
                if '100 - new' in folder:
                    fileString = 'genuine'

                write_hyper_spec_from_array(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet, feature)


        x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, index_order = get_batches(
            hyperspec_destination, wavelet, batchSize, scales, num_segs)

        input_shape = (x_train_cwt.shape[1], x_train_cwt.shape[2], x_train_cwt.shape[3])
        # create cnn model
        cnn_model = build_cnn_model("relu", input_shape, len(np.unique(y_test)))
        # train cnn model
        trained_cnn_model = compile_and_fit_model(cnn_model, train_batches, val_batches, 5)
        trained_cnn_model.save_weights(save_classifier_weights_to)

        return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, trained_cnn_model, index_order


    x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model, index_order = cwt()

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
    y_all = np.hstack((y_train, y_val, y_test))


    layer_name = [layer.name for layer in model.layers][-2]
    n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]

    dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(x_all)))
    dense_output = get_dense_output_no_images(dense_output, x_all, y_all, model, layer_name, binary=False, write=True)
    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                        genuine_classes=[0, 1, 2, 3],
                                                                                        load=not retrain_scoremodel,
                                                                                        load_name=load_score_model_weights_dir,
                                                                                        number_of_searches=10000,
                                                                                        save_name=save_score_model_weights_to)
    scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    scores_df[rgb_image_feat.split('/')[-2] + '_cwt_scores'] = [np.nan] * len(scores_df)
    index_order = [label.replace('C-26522', 'C26522') if 'C-26522' in label else label for label in
                   index_order]  # TODO GO BACK AND UNFUDGE THIS
    for ii, index_val in enumerate(index_order):
        scores_df.loc[index_val][rgb_image_feat.split('/')[-2] + '_cwt_scores'] = scores_in_order_of_index_order[ii]

    return scores_df

# '''


if __name__ == '__main__':
    with tf.device('CPU:0'):
        folder_index = ['C26459', 'C26522', 'A20-2', 'A16', 'A13', '100 - new']
        num_in_each = [12, 12, 80, 64, 52, 320]

        index_list = []
        for fd, num in zip(folder_index, num_in_each):
            index_list = index_list + [fd + '_' + str(math.floor(i/4)) + '_' + str(i%4) for i in range(num)]
        scored_frame = pd.DataFrame(
            index=index_list)

        scores_df = main('D:/scoring_and_profiling/TrsSeal/', scored_frame, 0)
        scores_df.to_csv('./first_look.csv')



