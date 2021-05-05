import pandas as pd
import numpy as np
from keras import backend
import os
import tensorflow as tf
from test_edge_model import tool_this_folder

from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

if __name__ == '__main__':
    with tf.device('CPU:0'):
        load_classifier_weights_dir = './model_dir/classifiers/classify_bridges.h5'
        save_classifier_weights_to = './model_dir/classifiers/classify_bridges.h5'
        retrain_classifier = False

        save_score_model_weights_to = './GAM_model/models/chris_test'
        load_score_model_weights_dir = './GAM_model/models/chris_test'
        retrain_scoremodel = False

        rgb_image_feat = 'D:/FRNLib/stripes/all/'


        new_tool_outputs = False
        tool_images_in_this_folder = 'D:/FRNLib/stripes/all/'
        spit_tool_output_here = 'D:/FRNLib/stripes/edges/'


        load_tool_weights_from = './model_dir/weights_robust_lighting_texture.h5'

        img_folder_to_classify = spit_tool_output_here
        img_folder_to_test_classifier = 'D:/FRNLib/stripes/edges_test_set/'

        '''
        tool_images_in_this_folder = 'D:/FRNLib/new_federal_seals/all/'
        spit_tool_output_here = 'D:/FRNLib/new_federal_seals/edges/'
        rgb_image_feat = 'D:/FRNLib/new_federal_seals/all/'
        img_folder_to_classify = spit_tool_output_here
        img_folder_to_test_classifier = spit_tool_output_here
        '''

        layer_name = 'dense_1'
        batchSize = 16
        epochs = 20
        num_classes = 2

        def edge(load_weights_dir, img_folder_to_classify):
            if new_tool_outputs:
                tool_this_folder(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
            x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify)
            train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                                   y_train, y_test, y_val, batchSize)
            model = classify_build_conv(load_weights_dir)
            return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model

        x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model = edge(load_classifier_weights_dir, img_folder_to_classify)

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


        n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
        dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(os.listdir(img_folder_to_test_classifier))))
        dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(os.listdir(img_folder_to_test_classifier))))
        dense_output = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name, write=True)
        ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                            num_classes,
                                                                                            genuine_classes=[1],
                                                                                            pca_quality=0.99,
                                                                                            pca_splines=20,
                                                                                            pca_lam=0.2,
                                                                                            pred_splines=5,
                                                                                            pred_lam=0.01,
                                                                                            number_of_searches=2000,
                                                                                            load=not retrain_scoremodel,
                                                                                            load_name=load_score_model_weights_dir,
                                                                                            save_name=save_score_model_weights_to)

        write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, images_scored=img_folder_to_test_classifier, rgb_dir=rgb_image_feat)

