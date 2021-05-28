import pandas as pd
import numpy as np
from keras import backend
import os
import tensorflow as tf
from test_edge_model import tool_this_folder
from skimage import feature
from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator, classify_rnn
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
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

def make_directories(dir_list):
    for item in dir_list:
        try:
            os.makedirs(item)
        except FileExistsError:
            continue


def main(rgb_image_feat, scores_df):

    save_classifier_weights_to = './temp_models_in_progress/edge' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/edge.h5'
    load_classifier_weights_dir = './temp_models_in_progress/edge' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/edge.h5'
    load_classifier_weights_dir = './temp_models_in_progress/edge_all_classifier/asdasdaedge.h5'
    retrain_classifier = True

    save_score_model_weights_to = './temp_models_in_progress/edge' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './temp_models_in_progress/edge' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './temp_models_in_progress/edge_all_64f17'
    retrain_scoremodel = True

    new_tool_outputs = False
    tool_images_in_this_folder = rgb_image_feat
    #spit_tool_output_here = '/'.join(rgb_image_feat.split('/')[0:-2]) + '/' + rgb_image_feat.split('/')[-2] + '_edges/'
    spit_tool_output_here = 'D:/scoring_and_profiling/StripeCrop_edges_aug/'

    make_directories([spit_tool_output_here, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1])])

    load_tool_weights_from = './model_dir/weights_robust_lighting_texture.h5'

    img_folder_to_classify = 'D:/scoring_and_profiling/StripeCrop_edges_aug/'
    img_folder_to_test_classifier = 'D:/scoring_and_profiling/StripeCrop_edges/'

    batchSize = 32
    rnn_neurons = 120
    dense_neurons = 14
    radius = 1.631
    points = 6
    lbp_class = LocalBinaryPatterns(points, radius)
    feature_vec_length = points + 2
    epochs = 10
    num_classes = 2

    def edge(load_weights_dir, img_folder_to_classify):
        if new_tool_outputs:
            tool_this_folder(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
        x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify)
        train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                               y_train, y_test, y_val, batchSize, blur=False, lbp=True, lbp_class=lbp_class)

        model = classify_rnn(load_weights_dir, rnn_neurons, dense_neurons, feature_vec_length)
        return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model

    x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model = edge(
        load_classifier_weights_dir, img_folder_to_classify)

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

    layer_name = [layer.name for layer in model.layers][-2]
    n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
    dense_output = pd.DataFrame(columns=range(n_dense_neurons),
                                index=range(len(os.listdir(img_folder_to_test_classifier))))
    dense_output, index_order = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name,
                                                 write=False,
                                                 edge=True,
                                                 TEMP=True,
                                                 lbp_class=lbp_class)
    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                        num_classes,
                                                                                        genuine_classes=[1],
                                                                                        pca_quality=0.99,
                                                                                        pca_splines=20,
                                                                                        pca_lam=0.2,
                                                                                        pred_splines=5,
                                                                                        pred_lam=0.2,
                                                                                        number_of_searches=2000,
                                                                                        load=not retrain_scoremodel,
                                                                                        load_name=load_score_model_weights_dir,
                                                                                        save_name=save_score_model_weights_to)

    # scores in os.listdir
    scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    please_work = pd.DataFrame(index=index_order, columns=['scores'])
    please_work['scores'] = scores_in_order_of_index_order
    please_work.to_csv('./pleeeeeease.csv')

    scores_df[rgb_image_feat.split('/')[-2] + '_edge_scores'] = [np.nan] * len(scores_df)
    for ii, index_val in enumerate(index_order):
        scores_df.loc[index_val][rgb_image_feat.split('/')[-2] + '_edge_scores'] = scores_in_order_of_index_order[ii]

    # write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth,
    #                  images_scored=img_folder_to_test_classifier, rgb_dir=rgb_image_feat)
    scores_df.to_csv('./fucking_finally.csv')
    return scores_df


if __name__ == '__main__':
    folder_index = ['100-cft-big', '100-gen-big', '100-gen', 'A13', 'A16', 'A20-2', 'C26522', 'C26459']
    num_in_each = [190, 251, 80, 13, 16, 20, 3, 3]

    index_list = []
    for fd, num in zip(folder_index, num_in_each):
        index_list = index_list + [fd + '_' + str(i) for i in range(num)]
    scored_frame = pd.DataFrame(
        index=['_'.join(note.split('_')[0:2]) for note in index_list])
    with tf.device('CPU:0'):
        rgb_image_feat = 'D:/FRNLib/federal seals/all/'
        main(rgb_image_feat, scored_frame)

