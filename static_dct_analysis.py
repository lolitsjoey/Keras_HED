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
    save_classifier_weights_to = './temp_models_in_progress/dct' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/dct.h5'
    load_classifier_weights_dir = './temp_models_in_progress/dct' + '_' + rgb_image_feat.split('/')[
        -2] + '_classifier/dct.h5'
    retrain_classifier = True

    save_score_model_weights_to = './temp_models_in_progress/dct' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './temp_models_in_progress/dct' + '_' + rgb_image_feat.split('/')[-2]
    load_score_model_weights_dir = './temp_models_in_progress/dct_all_64f17'
    retrain_scoremodel = True

    new_tool_outputs = False
    tool_images_in_this_folder = rgb_image_feat
    # spit_tool_output_here = '/'.join(rgb_image_feat.split('/')[0:-2]) + '/' + rgb_image_feat.split('/')[-2] + '_dcts/'
    spit_tool_output_here = 'D:/scoring_and_profiling/StripeCrop_dct/'

    make_directories([spit_tool_output_here, '/'.join(save_classifier_weights_to.split('/')[0:-1]),
                      '/'.join(load_classifier_weights_dir.split('/')[0:-1])])

    load_tool_weights_from = ['./model_dir/dctencoderweights.h5', './model_dir/dctdecoderweights.h5']

    img_folder_to_classify = 'D:/scoring_and_profiling/StripeCrop/'
    img_folder_to_test_classifier = 'D:/scoring_and_profiling/StripeCrop/'

    batchSize = 16
    epochs = 10
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
    dense_output, index_order = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name, write=False,
                                    dct=True)

    y = np.array(dense_output['truth'])
    del dense_output['truth']
    if retrain_scoremodel:
        rand_gam, new_values, titles = create_rand_gam(2000, dense_output, y, pca_splines=20, pca_lam=0.2)
        save_model(save_score_model_weights_to, rand_gam)
    else:

        new_values = dense_output
        rand_gam = load_model(load_score_model_weights_dir)

    scores = rand_gam.predict_proba(new_values)
    predictions = np.array(rand_gam.predict(new_values), dtype=np.int)

    if retrain_scoremodel:
        ordered_scores, ordered_labels, arguments = learn_transform_scores(scores, predictions,
                                                                           load_score_model_weights_dir,
                                                                           save_score_model_weights_to,
                                                                           load=not retrain_scoremodel)
        ordered_truth = np.array(y)[arguments]
        print_stats(rand_gam, ordered_scores, ordered_labels, ordered_truth, new_values, y)
        scores_in_order_of_index_order = ordered_scores[np.argsort(arguments)]
        please_work = pd.DataFrame(index=index_order, columns=['scores'])
        please_work['scores'] = scores_in_order_of_index_order
        please_work.to_csv('./pleeeeeease.csv')

    else:
        transformed_scores = score_transform(scores, predictions, load_score_model_weights_dir)
        please_work = pd.DataFrame(index=index_order, columns=['scores'])
        please_work['scores'] = transformed_scores
        please_work.to_csv('./pleeeeeease.csv')

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

