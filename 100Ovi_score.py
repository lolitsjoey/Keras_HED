import pandas as pd
import numpy as np
from keras import backend
import os
import tensorflow as tf
from test_edge_model import tool_this_folder
from skimage import feature
from classifier.train_edge_classifier import classify_build_conv,  grab_batches_with_generator, classify_rnn_2d, classify_build_conv_6d,  prepare_data
from classifier.test_edge_classifier import get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages, print_stats, learn_transform_scores
import pygam
import keras
import decimal
from sklearn.model_selection import train_test_split
import pygam
from pygam.datasets import wage, toy_interaction
from pygam import LinearGAM, LogisticGAM, s, f
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import cv2
import shutil
from keras.utils import to_categorical
import pickle
import uuid
from pygam.datasets import mcycle
from sklearn.linear_model import LogisticRegression
import decimal
import math
import warnings


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

    save_classifier_weights_to = './temp_models_in_progress/100ovi_classifier_no_blur/edge.h5'
    load_classifier_weights_dir = './temp_models_in_progress/100ovi_classifier_no_blur/edge.h5'
    retrain_classifier = False

    save_score_model_weights_to = './temp_models_in_progress/100ovi_scoremodel'
    load_score_model_weights_dir = './temp_models_in_progress/100ovi_scoremodel'
    retrain_scoremodel = True

    new_tool_outputs = False
    tool_images_in_this_folder = 'D:/scoring_and_profiling/100ovi_aug/'
    spit_tool_output_here = 'D:/scoring_and_profiling/100ovi_edges_aug/'

    make_directories([spit_tool_output_here, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1]), './temp_models_in_progress/edge_score_model/'])

    load_tool_weights_from = './model_dir/weights_robust_lighting_texture.h5'

    img_folder_to_classify = 'D:/scoring_and_profiling/100ovi_edges_aug/'
    img_folder_to_test_classifier = 'D:/scoring_and_profiling/100ovi_edges_aug/'
    #'rnn_neurons': 150, 'dense_neurons': 19, 'radius': 3.63, 'points': 4
    batchSize = 40
    radius = 3.63
    points = 4
    lbp_class = LocalBinaryPatterns(points, radius)
    feature_vec_length = points + 2
    epochs = 10
    num_classes = 2

    def edge(load_weights_dir, img_folder_to_classify):
        if new_tool_outputs:
            tool_this_folder(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
        x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify, seed=420)
        train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                               y_train, y_test, y_val, batchSize, blur=False, lbp=False, lbp_class=lbp_class, multichannel=False)
        #model = classify_rnn_2d(load_weights_dir, rnn_neurons, dense_neurons, feature_vec_length)
        model = classify_build_conv(load_weights_dir)
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
                                                 write=True,
                                                 edge=True,
                                                 TEMP=True,
                                                 lbp_class=lbp_class)

    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                        num_classes,
                                                                                        genuine_classes=[1],
                                                                                        pca_quality=4,
                                                                                        pca_splines=20,
                                                                                        pca_lam=0.1,
                                                                                        pred_splines=5,
                                                                                        pred_lam=0.2,
                                                                                        number_of_searches=2000,
                                                                                        load=not retrain_scoremodel,
                                                                                        load_name=load_score_model_weights_dir,
                                                                                        save_name=save_score_model_weights_to,
                                                                                        confidence=False)

    # dense_output,
    # num_classes,
    # genuine_classes = [1],
    # pca_quality = 0.99,
    # pca_splines = 20,
    # pca_lam = 0.2,
    # pred_splines = 5,
    # pred_lam = 0.2,
    # number_of_searches = 2000,
    # load = not retrain_scoremodel,
    # load_name = load_score_model_weights_dir,
    # save_name = save_score_model_weights_to
    # scores in os.listdir
    scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    please_work = pd.DataFrame(index=index_order, columns=['scores'])
    please_work['scores'] = scores_in_order_of_index_order
    please_work.to_csv('./pleeeeeease100ovi2.csv')

    # y = np.array(dense_output['truth'])
    # del dense_output['truth']
    # if retrain_scoremodel:
    #     rand_gam, new_values, titles = create_rand_gam(2000, dense_output, y, pca_splines=20, pca_lam=0.2)
    #     save_model(save_score_model_weights_to, rand_gam)
    # else:
    #
    #     new_values = dense_output
    #     rand_gam = load_model(load_score_model_weights_dir)
    #
    # scores = rand_gam.predict_proba(new_values)
    # predictions = np.array(rand_gam.predict(new_values), dtype=np.int)
    #
    # if retrain_scoremodel:
    #
    #     ordered_scores, ordered_labels, arguments = learn_transform_scores(scores, predictions, load_score_model_weights_dir, save_score_model_weights_to, load=not retrain_scoremodel)
    #     ordered_truth = np.array(y)[arguments]
    #     print_stats(rand_gam, ordered_scores, ordered_labels, ordered_truth, new_values, y)
    #     scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
    #     please_work = pd.DataFrame(index=index_order, columns=['scores'])
    #     please_work['scores'] = scores_in_order_of_index_order
    #     please_work.to_csv('./pleeeeeease.csv')
    #
    # else:
    #     transformed_scores = score_transform(scores, predictions, load_score_model_weights_dir)
    #     please_work = pd.DataFrame(index=index_order, columns=['scores'])
    #     please_work['scores'] = transformed_scores
    #     please_work.to_csv('./pleeeeeease.csv')


def decimal_from_value(value):
    return decimal.Decimal(value)

def score_transform(score, predictions, load_name):
    try:
        scores = np.array(score, dtype=decimal.Decimal)
        trans_scores = []
        for idx, scr in enumerate(scores):
            extrema = pd.read_csv(load_name + '/' + load_name.split('/')[-1] + '_extreme_scores.csv',
                              converters={'bounds': decimal_from_value})
            slope = (100 - 0) / (extrema.values[1][0] - extrema.values[0][0])
            lower_b = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(extrema.values[0][0])))
            decimal.getcontext().prec = abs(int(math.floor(math.log10(lower_b)))) + 5
            upper_b = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(extrema.values[1][0])))
            #print('{}:   upp {}  low {}   scr {}'.format(load_name, round(upper_b,3), round(lower_b,3), round(scr,3)))
            if scr >= upper_b:
                scr = upper_b
                extreme = 1
            elif scr <= lower_b:
                scr = lower_b
                extreme = 1
            temp_score = round(slope * (- decimal.Decimal(1 / scr - 1).ln() - extrema.values[0][0]), 3)
            if (extreme == 0) & (predictions[idx]==1):
                temp_score = temp_score + (100-temp_score) * decimal.Decimal(0.7)
            trans_scores.append(temp_score)
        return trans_scores
    except Exception as e:
        print(e)

def save_model(save_name, rand_gam):
    if os.path.exists(save_name):
        suffix = str(uuid.uuid4())[0:5]
        print('Save Path already exists, appending ({})'.format(suffix))
        save_name = save_name + '_' + str(suffix)
        os.makedirs(save_name)
    else:
        os.makedirs(save_name)

    with open(save_name + '/' + save_name.split('/')[-1] + '_model.pkl', 'wb') as f:
        pickle.dump(rand_gam, f)
    return save_name

def load_model(load_name):
    with open(load_name + '/' + load_name.split('/')[-1] + '_model.pkl', 'rb') as f:
        rand_gam = pickle.load(f)
    return rand_gam


def create_rand_gam(number_of_searches, new_values, y, pca_splines, pca_lam):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    lams = np.random.rand(number_of_searches, new_values.shape[1])  # random points on [0, 1], with shape (1000, 3)
    lams = lams * 8 - 4  # shift values to -4, 4
    lams = 10 ** lams  # transforms values to 1e-4, 1e4
    lams[:,-1] = [10 ** i for i in np.random.rand(number_of_searches) * 4]

    titles = []
    dtype_string = []
    for i in range(new_values.shape[1]):
        titles.append(str(i))
        if i == 0:
            x = s(i, n_splines=pca_splines, lam=pca_lam)
        else:
            x = x + s(i, n_splines=pca_splines, lam=pca_lam)
        dtype_string.append('numerical')
    rand_gam = LogisticGAM(x).gridsearch(np.array(new_values), y, lam=lams)
    return rand_gam, new_values, titles

def get_dense_output(dense_df, img_folder_to_classify, model, layer_name, imShape = (480,480), write = False, edge=False, dct=False, TEMP=False, lbp_class=None):
    predictions = []
    truth = []
    index_order = []
    for idx, img_0 in enumerate(os.listdir(img_folder_to_classify)):
        if idx == 0:
            print(img_0)
        if TEMP:
            test_img = cv2.resize(cv2.imread(img_folder_to_classify + img_0, 3), imShape)
            test_img = ((test_img - np.mean(test_img)) / np.std(test_img))
            model_input = test_img[None,:,:,:]
            answer = model.predict(model_input)
        else:
            test_img = ((test_img - np.mean(test_img)) / np.std(test_img))
            if edge:
                test_img = cv2.GaussianBlur(test_img, (9, 9), 0)
            model_input = test_img[None, :, :, :]
            answer = model.predict(model_input)
        predictions.append(int(np.argmax(answer)))
        index_order.append('_'.join(img_0.split('_')[0:2]))

        if 'genuine' in img_0:
            truth.append(1)
        else:
            truth.append(0)

        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(model_input)
        dense_df.iloc[idx,:] = list(intermediate_output)[0]

        if idx%100 == 99:
            print('Got predictions from {} Images...'.format(idx))

    dense_df['truth'] = truth
    dense_df['prediction'] = predictions
    if write:
        dense_df.to_csv('./intermediate_output.csv')

    return dense_df, index_order

def get_dense_output_6d(dense_df, img_folder_to_classify, model, layer_name, imShape = (480,480), write = False, edge=False, dct=False, TEMP=False, lbp_class=None):
    predictions = []
    truth = []
    index_order = []
    for idx, (img_0, img_1) in enumerate(
            zip(os.listdir(img_folder_to_classify[0]), os.listdir(img_folder_to_classify[1]))):
        if idx == 0:
            print(img_0)
        if TEMP:
            test_img_0 = cv2.resize(cv2.imread(img_folder_to_classify[0] + img_0, 1), imShape)
            test_img_1 = cv2.resize(cv2.imread(img_folder_to_classify[1] + img_1, 1), imShape)
            model_input = np.concatenate((test_img_0, test_img_1), axis=2)[None,:,:,:]
            answer = model.predict(model_input)
        else:
            test_img = ((test_img - np.mean(test_img)) / np.std(test_img))
            if edge:
                test_img = cv2.GaussianBlur(test_img, (9, 9), 0)
            model_input = test_img[None, :, :, :]
            answer = model.predict(model_input)
        predictions.append(int(np.argmax(answer)))
        index_order.append('_'.join(img_0.split('_')[0:2]))

        if 'genuine' in img_0:
            truth.append(1)
        else:
            truth.append(0)

        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(model_input)
        dense_df.iloc[idx, :] = list(intermediate_output)[0]

        if idx % 100 == 99:
            print('Got predictions from {} Images...'.format(idx))

    dense_df['truth'] = truth
    dense_df['prediction'] = predictions
    if write:
        dense_df.to_csv('./intermediate_output.csv')

    return dense_df, index_order


if __name__ == '__main__':
    with tf.device('CPU:0'):
        folder_index = ['100-cft-big', '100-gen-big', '100-gen', 'A13', 'A16', 'A20-2', 'C26522', 'C26459']
        num_in_each = [190, 251, 80, 13, 16, 20, 3, 3]

        index_list = []
        for fd, num in zip(folder_index, num_in_each):
            index_list = index_list + [fd + '_' + str(i) for i in range(num)]
        scored_frame = pd.DataFrame(
            index=['_'.join(note.split('_')[0:2]) for note in index_list])
        #with tf.device('CPU:0'):
        rgb_image_feat = 'D:/FRNLib/federal seals/all/'
        main(rgb_image_feat, scored_frame)

