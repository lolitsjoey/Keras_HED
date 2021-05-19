import pandas as pd
import numpy as np
from keras import backend
import os
import tensorflow as tf
from test_edge_model import tool_this_folder
from auto_dct import loadModel
from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

from auto_dct import evaluateModel

def make_directories(dir_list):
    for item in dir_list:
        try:
            os.makedirs(item)
        except FileExistsError:
            continue

def main(rgb_image_feat, scores_df):
    load_classifier_weights_dir = './gibberish.h5'
    save_classifier_weights_to = './score_models_per_feature/dct' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/dct.h5'
    retrain_classifier = True

    save_score_model_weights_to = './score_models_per_feature/dct' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './score_models_per_feature/dct' + '_' + rgb_image_feat.split('/')[-2]
    retrain_scoremodel = True

    new_tool_outputs = True
    tool_images_in_this_folder = rgb_image_feat
    spit_tool_output_here = '/'.join(rgb_image_feat.split('/')[0:-2]) + '/' + rgb_image_feat.split('/')[-2] + '_dct/'

    make_directories([spit_tool_output_here, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1])])

    load_tool_weights_from = ['./model_dir/newlrencoderweights.h5', './model_dir/newlrdecoderweights.h5']

    img_folder_to_classify = spit_tool_output_here
    img_folder_to_test_classifier = spit_tool_output_here


    batchSize = 16
    epochs = 35
    num_classes = 2

    def dct(load_weights_dir, img_folder_to_classify):
        if new_tool_outputs:
            evaluateModel(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
        x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify)
        train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                               y_train, y_test, y_val, batchSize,
                                                                               blur=False)
        model = classify_build_conv(load_weights_dir)
        return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model

    x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model = dct(
        load_classifier_weights_dir, img_folder_to_classify)

    if retrain_classifier:
        history = model.fit(x=train_batches,
                            epochs=epochs,
                            verbose=0,
                            validation_data=val_batches
                            )
        model.save_weights(save_classifier_weights_to)

    print('Evaluating Model...')
    loss, accuracy = model.evaluate(test_batches)
    print('Accuracy: %.2f' % (accuracy * 100))

    layer_name = [layer.name for layer in model.layers][-2]
    n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
    dense_output = pd.DataFrame(columns=range(n_dense_neurons),
                                index=range(len(os.listdir(img_folder_to_test_classifier))))
    dense_output, index_order = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name, write=False,
                                    dct=True)
    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                        num_classes,
                                                                                        genuine_classes=[1],
                                                                                        pca_quality=0.99,
                                                                                        pca_splines=10,
                                                                                        pca_lam=0.2,
                                                                                        pred_splines=5,
                                                                                        pred_lam=0.2,
                                                                                        number_of_searches=2000,
                                                                                        load=not retrain_scoremodel,
                                                                                        load_name=load_score_model_weights_dir,
                                                                                        save_name=save_score_model_weights_to)

    scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    scores_df[rgb_image_feat.split('/')[-2] + '_dct_scores'] = [np.nan] * len(scores_df)
    for ii, index_val in enumerate(index_order):
        scores_df.loc[index_val][rgb_image_feat.split('/')[-2] + '_dct_scores'] = scores_in_order_of_index_order[ii]


    # write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, images_scored=img_folder_to_test_classifier, rgb_dir=rgb_image_feat)
    return scores_df

if __name__ == '__main__':
    with tf.device('CPU:0'):
        main(rgb_image_feat, scores_df)

