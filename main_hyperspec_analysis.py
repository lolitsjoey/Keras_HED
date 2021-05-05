import os
import cv2
import pandas as pd
import numpy as np
from keras import backend
import tensorflow as tf
from test_edge_model import tool_this_folder

from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

from hyperspec_classifier.hyperspec_classifier import get_batches, compile_and_fit_model
from hyperspec_classifier.hyperspec_tools import build_cnn_model
from hyperspec_classifier.hyperspec_writer import write_hyper_spec

if __name__ == '__main__':
    with tf.device('CPU:0'):
        #load_classifier_weights_dir = './hyperspec_classifier/learning_to_learn_save.h5'
        load_classifier_weights_dir = './hyperspec_classifier/gibberish.h5'
        save_classifier_weights_to = './hyperspec_classifier/learning_to_learn_cwt_inkwell.h5'

        retrain_classifier = True
        retrain_scoremodel = True

        save_score_model_weights_to = './GAM_model/models/inkwell_save_and_load_a21ff'
        load_score_model_weights_dir = './GAM_model/models/inkwell_save_and_load_a21ff'

        layer_name = 'dense_1'
        batchSize = 24
        epochs = 20


        #morl is best so far, gaus3 is interesting, gaus6 is the best for offset distinguish, gaus7/gaus8 scores the best consistently
        wavelet = 'mexh'  # mother wavelet
        scales = np.arange(1, 65, 0.25)  # range of scales
        num_segs = 4
        rewrite_hyperspec = False
        hyperspec_destination = './hyperspec_classifier/hyperspec_inkwell/'

        def cwt():
            if rewrite_hyperspec:
                folder_of_folders = 'D:/FRNLib/noteLibrary/just_100s/'
                fileString = 'genuine'
                write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet)

                folder_of_folders = 'D:/FRNLib/noteLibrary/CounterfeitHyper/'
                fileString = 'counterfeit'
                write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet)




            x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches = get_batches(hyperspec_destination, wavelet, batchSize, scales, num_segs)

            input_shape = (x_train_cwt.shape[1], x_train_cwt.shape[2], x_train_cwt.shape[3])
            # create cnn model
            cnn_model = build_cnn_model("relu", input_shape, len(np.unique(y_test)))
            # train cnn model
            trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, train_batches, val_batches, 5)
            trained_cnn_model.save_weights(save_classifier_weights_to)

            return x_train_cwt, x_test_cwt, x_val_cwt, y_train, y_test, y_val, train_batches, test_batches, val_batches, trained_cnn_model

        x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model = cwt()

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

        n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
        dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(x_all)))
        dense_output = get_dense_output_no_images(dense_output, x_all, y_all, model, layer_name, binary = False, write = True)

        ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                            genuine_classes = [0,1,2,3],
                                                                                            load= not retrain_scoremodel,
                                                                                            load_name=load_score_model_weights_dir,
                                                                                            number_of_searches=10000,
                                                                                            save_name=save_score_model_weights_to)

        #write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, genuine_classes = [0,1,2,3],images_scored=hyperspec_destination + 'cwt/', rgb_dir=hyperspec_destination + 'rgb/')
        write_out_scores_noimages(ordered_scores, ordered_labels, arguments, ordered_truth, array_scored=x_all, genuine_classes = [0,1,2,3])
#'''





