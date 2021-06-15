import os
import cv2
import numpy as np
import pandas as pd
from models_to_load import LocalBinaryPatterns, classify_build_conv, classify_rnn_2d, classify_build_conv_6d, fedSealModel
from src.networks.hed_reduced import hed
from src.networks.hed_ultrafine import hed_refined
from prepare_images_lookup import lookup_model_input
from classifier.score_from_network import load_score_model
from auto_dct import load_dct_model
from auto_fft import load_fft_model
from sklearn.cluster import KMeans
import math
import decimal
import random
import keras


def get_edge_map(feat_image, tool_model):
    resized_feat_image = cv2.resize(feat_image, (480, 480))
    edge_map = tool_model.predict(resized_feat_image[None, :, :, :])

    edge_map = edge_map[0][0, :, :, 0]
    edge_map = ((edge_map - np.mean(edge_map)) / np.std(edge_map))
    edge_map = cv2.GaussianBlur(edge_map, (9, 9), 0)
    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    return edge_map


def loadImage(image, imageShape=(64,64)):
    return cv2.resize(image, imageShape) / 255.


def get_dct_map(feat_image, tool_model):
    resized_feat_image = cv2.resize(feat_image, (64, 64))
    resized_feat_image = cv2.cvtColor(resized_feat_image, cv2.COLOR_RGB2GRAY)
    resized_feat_image = np.expand_dims(np.array(resized_feat_image[None, :, :]/255.), axis=-1)
    decodedImages = tool_model.predict(resized_feat_image)

    dct_map = (decodedImages[0] * 255).astype("uint8")
    dct_map = cv2.resize(cv2.cvtColor(dct_map, cv2.COLOR_GRAY2RGB), (480, 480))
    dct_map = ((dct_map - np.mean(dct_map)) / np.std(dct_map))
    return dct_map

def get_fft_map(feat_image, tool_model):
    resized_feat_image = cv2.resize(feat_image, (128, 128))
    resized_feat_image = cv2.cvtColor(resized_feat_image, cv2.COLOR_RGB2GRAY)
    resized_feat_image = np.expand_dims(np.array(resized_feat_image[None, :, :]/255.), axis=-1)
    decodedImages = tool_model.predict(resized_feat_image)

    fft_map = (decodedImages[0] * 255).astype("uint8")
    fft_map = cv2.resize(cv2.cvtColor(fft_map, cv2.COLOR_GRAY2RGB), (480, 480))
    fft_map = ((fft_map - np.mean(fft_map)) / np.std(fft_map))
    return fft_map


def getFeature(roi, image, by_index=False):
    if by_index:
        y_fac = 1
        x_fac = 1
    else:
        y_fac = image.shape[0]
        x_fac = image.shape[1]

    minY = int(roi[0] * y_fac)
    minX = int(roi[1] * x_fac)
    maxY = int(roi[2] * y_fac)
    maxX = int(roi[3] * x_fac)
    if len(image.shape) == 2:
        return image[minY:maxY, minX:maxX]
    return image[minY:maxY, minX:maxX, :]



def analyseFeature(fDict, models, noteRGB):
    feat_name = fDict["className"]
    feat_image = getFeature(fDict["roi"], noteRGB.array, by_index=True)
    # feat_num = len(os.listdir('./testing/note_features/'))
    # cv2.imwrite('./testing/note_features/feat_image{}.bmp'.format(feat_num), feat_image)
    tool_list = ['edge_', 'dct_', 'fft_']

    tool_scores = np.zeros(3)
    tool_feature_vector = np.zeros(3, dtype=object)
    for idx, method in enumerate(tool_list):
        tool_model = models[method + 'tool']
        classifier = models[method + feat_name + '_classifier']
        [score_model, pca] =  [method + feat_name + '_scorer']

        if method == 'edge_':
            tool_output_image = get_edge_map(feat_image, tool_model)
        if method == 'dct_':
            tool_output_image = get_dct_map(feat_image, tool_model)
        if method == 'fft_':
            tool_output_image = get_fft_map(feat_image, tool_model)

        classifier_prediction = classifier.predict(tool_output_image[None, :, :, :])
        classifier_prediction = np.argmax(classifier_prediction)
        layer_name = [layer.name for layer in classifier.layers][-2]
        intermediate_layer_model = keras.Model(inputs=classifier.input,
                                               outputs=classifier.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(tool_output_image[None, :, :, :])

        feat_vector = list(intermediate_output)[0]
        tool_feature_vector[idx] = feat_vector
        score_frame = pd.DataFrame(index=range(1), columns=range(len(feat_vector)))
        score_frame.iloc[0, :] = feat_vector
        transformed_feat_vector = pca.transform(score_frame)
        transformed_feat_vector = np.append(transformed_feat_vector, np.array(classifier_prediction).reshape(-1, 1),
                                            axis=1)

        score = score_model.predict_proba(transformed_feat_vector)
        transformed_score = score_transform(score, './model_weights/scoremodel_weights/' + method + feat_name)
        transformed_score = transformed_score / 100
        if transformed_score <= 0.001:
            transformed_score = 0.001
        if transformed_score >= 0.999:
            transformed_score = 0.999
        tool_scores[idx] = transformed_score

    mean_tool_score = round(np.mean(tool_scores), 3)
    if mean_tool_score <= 0.001:
        mean_tool_score = 0.001
    if mean_tool_score >= 0.999:
        mean_tool_score = 0.999

    return {
               "name": fDict["className"],
               "side": "front",
               "score": mean_tool_score,
               "text": "PLACEHOLDER",
               "feature_vector": "43.54,54.32,54.6564,4345.544",
               "region_on_note": [{"x": int(fDict['roi'][1]), "y": int(fDict['roi'][0])},
                                  {"x": int(fDict['roi'][3]), "y": int(fDict['roi'][0])},
                                  {"x": int(fDict['roi'][3]), "y": int(fDict['roi'][2])},
                                  {"x": int(fDict['roi'][1]), "y": int(fDict['roi'][2])}],
               "sub_tags": getSubTags(tool_list, tool_scores)
           }, tool_scores


def getSubTags():
    subTagList = []
    analysisTools = ['edge_detection', 'colour', 'shape', 'ink_density', 'paper_type', 'ir_response']
    for i in range(2, random.randint(4, 5)):
        subTagList.append({"name": analysisTools[i],
                           "score": round(random.random(), 3)})
    return subTagList


def get_classifier_dict():
    classifier_dict = {}
    for (root, dirs, files) in os.walk('./model_weights/classifier_weights/'):
        if len(files) == 0:
            continue
        if 'h5' in files[0]:
            classifier_name = root.split('/')[-1]
            classifier_dict[classifier_name] = classify_build_conv(root + '/' + files[0])
    return classifier_dict


def get_scoremodel_dict():
    scoremodel_dict = {}
    for score_model_folder in os.listdir('./model_weights/scoremodel_weights/'):
        scoremodel_dict[score_model_folder + '_scorer'] = load_score_model('./model_weights/scoremodel_weights/'
                                                                           + score_model_folder)
    return scoremodel_dict


def transform_and_append_prediction(feature_vector, feature_pca, class_prediction):
    score_frame = pd.DataFrame(index=range(1), columns=range(len(feature_vector)))
    score_frame.iloc[0, :] = feature_vector
    transformed_feat_vector = feature_pca.transform(score_frame)
    transformed_feat_vector = np.append(transformed_feat_vector, np.array(class_prediction).reshape(-1, 1),
                                        axis=1)
    return transformed_feat_vector


def decimal_from_value(value):
    return decimal.Decimal(value)

def score_transform(score, prediction, load_name):
    try:
        scores = np.array(score, dtype=decimal.Decimal)
        extrema = pd.read_csv(load_name + '/' + load_name.split('/')[-1] + '_extreme_scores.csv',
                          converters={'bounds': decimal_from_value})
        slope = (100 - 0) / (extrema.values[1][0] - extrema.values[0][0])
        scr = scores[0]
        lower_b = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(extrema.values[0][0])))
        decimal.getcontext().prec = abs(int(math.floor(math.log10(lower_b)))) + 5
        upper_b = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(extrema.values[1][0])))
        #print('{}:   upp {}  low {}   scr {}'.format(load_name, round(upper_b,3), round(lower_b,3), round(scr,3)))
        if scr >= upper_b:
            scr = upper_b
        elif scr <= lower_b:
            scr = lower_b
        pre_score = round(slope * (- decimal.Decimal(1 / scr - 1).ln() - extrema.values[0][0]), 3)
        if prediction == 1:
            return pre_score + (100 - pre_score)*decimal.Decimal(0.7)
        else:
            return pre_score
    except Exception as e:
        print(e)


def apply_bounds(score):
    transformed_score = score
    if transformed_score <= 0.1:
        transformed_score = 0.1
    if transformed_score >= 99.9:
        transformed_score = 99.9
    return transformed_score


def main():
    folder_index = ['100-cft-big', '100-gen-big', '100-gen', 'A13', 'A16', 'A20-2', 'C26522', 'C26459']
    num_in_each = [190, 251, 80, 13, 16, 20, 3, 3]

    index_list = []
    for fd, num in zip(folder_index, num_in_each):
        index_list = index_list + [fd + '_' + str(i) for i in range(num)]
    scored_frame = pd.DataFrame(
        index=['_'.join(note.split('_')[0:2]) for note in index_list])

    classifier_dict = get_classifier_dict()
    scoremodel_dict = get_scoremodel_dict()
    model_dict = {**classifier_dict, **scoremodel_dict}

    for iii, paper_scores in enumerate(os.listdir('./paper_models')):
        model_dict[str(iii) + '_Paper_scoring'] = (load_score_model('./paper_models/' + paper_scores), './paper_models/' + paper_scores)

    model_dict['TrsSeal_RGB_classifier'] = classify_build_conv_6d('./well_trained_models/cnn_TrsSeal_classifier/edge.h5')
    model_dict['TrsSeal_RGB_scoring'] = (
    load_score_model('./well_trained_models/cnn_TrsSeal_score'), './well_trained_models/cnn_TrsSeal_score')

    model_dict['Stripe_RGB_classifier'] = classify_rnn_2d('./well_trained_models/cnn_Stripe_classifier/edge.h5',
                                                         rnn_neurons=120, dense_neurons=14, feature_vec_length=8)
    model_dict['Stripe_RGB_scoring'] = (
    load_score_model('./well_trained_models/cnn_Stripe_score'), './well_trained_models/cnn_Stripe_score')

    model_dict['100ovi_RGB_classifier'] = classify_build_conv('./well_trained_models/cnn_100ovi_classifier/edge.h5')
    model_dict['100ovi_RGB_scoring'] = (
    load_score_model('./well_trained_models/cnn_100ovi_score'), './well_trained_models/cnn_100ovi_score')

    model_dict['FedSeal_RGB_classifier'] = fedSealModel('./model_weights/fed_seal_kish/fedsealnewopt.h5')
    model_dict['FedSeal_RGB_scoring'] = (
    load_score_model('./well_trained_models/cnn_FedSeal_score'), './well_trained_models/cnn_FedSeal_score')

    model_dict['Paper_RGB_Model'] = classify_build_conv_6d('./well_trained_models/paper_type_model/edge.h5')
    model_dict['Paper_RGB_Scoring'] = (load_score_model('./well_trained_models/paper_type_score'), './well_trained_models/paper_type_score')

    tool_dict = {'edge_tool': hed('./model_weights/edge_weights/weights_robust_lighting_texture.h5'),
                 'dct_tool': load_dct_model(['./model_weights/dct_weights/newlrencoderweights.h5',
                                             './model_weights/dct_weights/newlrdecoderweights.h5']),
                 'fft_tool': classify_build_conv_6d('./well_trained_models/paper_type_model/edge.h5'),
                 'lbp_tool': LocalBinaryPatterns(numPoints=6, radius=1.631)}

    model_dict = {**model_dict, **tool_dict}

    #scored_frame = pd.read_csv('./Stripe.csv', index_col=0)
    for folder in os.listdir('D:/scoring_and_profiling/'):
        if ('edges' in folder) or ('dct' in folder) or ('fft' in folder) or ('cwt' in folder) or ('waves' in folder) or ('aug' in folder) or ('Crop' in folder) or ('Crest' in folder) or ('Teeth' in folder):
            continue
        print('---- scoring {} ----'.format(folder))


        #model_dict[folder + '_RGB_classifier'] = \model_weights\classifier_weights
        index_order = []
        scores_list = []
        dict_of_paper_vectors = {}
        dict_of_paper_predictions = {}
        for feat_path1, feat_path2 in zip(os.listdir('D:/scoring_and_profiling/WtrMrk/'), os.listdir('D:/scoring_and_profiling/NoteLoc/')):
            feat_img1 = cv2.imread('D:/scoring_and_profiling/WtrMrk/' + feat_path1, 1)
            feat_img2 = cv2.imread('D:/scoring_and_profiling/NoteLoc/' + feat_path2, 1)
            model_input = lookup_model_input('Paper', feat_img1, model_dict, feat_img2)
            paper_classifier = model_dict['Paper_RGB_Model']
            paper_prediction = np.argmax(paper_classifier.predict(model_input))
            [feature_scorer, feature_pca], feature_load_name = model_dict['Paper_RGB_Scoring']

            layer_name = [layer.name for layer in paper_classifier.layers][-2]
            intermediate_layer_model = keras.Model(inputs=paper_classifier.input,
                                                   outputs=paper_classifier.get_layer(layer_name).output)
            feature_vector = list(intermediate_layer_model.predict(model_input))[0]
            transformed_paper_vector = transform_and_append_prediction(feature_vector, feature_pca, paper_prediction)
            dict_of_paper_vectors['_'.join(feat_path1.split('_')[0:2])] = transformed_paper_vector
            dict_of_paper_predictions['_'.join(feat_path1.split('_')[0:2])] = paper_prediction


        for iii, feat_path in enumerate(os.listdir('D:/scoring_and_profiling/' + folder + '/')):
            feat_img = cv2.imread('D:/scoring_and_profiling/' + folder + '/' + feat_path, 1)
            model_input = lookup_model_input(folder, feat_img, model_dict)

            if model_input is None:
                tool_list = ['edge_', 'dct_', 'fft_']
                tool_feature_vector = np.zeros(3, dtype=object)
                index_order.append('_'.join(feat_path.split('_')[0:2]))
                temp_scores = []
                for idx, method in enumerate(tool_list):
                    if method != 'fft_':
                        tool_model = model_dict[method + 'tool']
                        classifier = model_dict[method + folder + '_classifier']
                        [score_model, pca] = model_dict[method + folder + '_scorer']
                    else:
                        [feature_scorer, feature_pca], feature_load_name = model_dict[str(iii) + '_Paper_scoring']

                    if method == 'edge_':
                        tool_output_image = get_edge_map(feat_img, tool_model)
                    if method == 'dct_':
                        tool_output_image = get_dct_map(feat_img, tool_model)
                    if method == 'fft_':
                        transformed_paper_vector = dict_of_paper_vectors['_'.join(feat_path.split('_')[0:2])]
                        score = feature_scorer.predict_proba(transformed_feat_vector)

                        isGenuine = feature_scorer.predict(transformed_paper_vector)[0]
                        transformed_score = apply_bounds(score_transform(score, int(isGenuine), feature_load_name))
                        temp_scores.append(transformed_score)
                    else:
                        classifier_prediction = classifier.predict(tool_output_image[None, :, :, :])
                        classifier_prediction = np.argmax(classifier_prediction)
                        layer_name = [layer.name for layer in classifier.layers][-2]
                        intermediate_layer_model = keras.Model(inputs=classifier.input,
                                                               outputs=classifier.get_layer(layer_name).output)
                        intermediate_output = intermediate_layer_model.predict(tool_output_image[None, :, :, :])

                        feat_vector = list(intermediate_output)[0]
                        tool_feature_vector[idx] = feat_vector
                        score_frame = pd.DataFrame(index=range(1), columns=range(len(feat_vector)))
                        score_frame.iloc[0, :] = feat_vector
                        transformed_feat_vector = pca.transform(score_frame)
                        transformed_feat_vector = np.append(transformed_feat_vector,
                                                            np.array(classifier_prediction).reshape(-1, 1),
                                                            axis=1)

                    score = score_model.predict_proba(transformed_feat_vector)
                    isGenuine = score_model.predict(transformed_feat_vector)[0]
                    transformed_score = score_transform(score, int(isGenuine), './model_weights/scoremodel_weights/' + method + folder)
                    temp_scores.append(transformed_score)
                scores_list.append(temp_scores)

            else:
                feature_classifier = model_dict[folder + '_RGB_classifier']
                [feature_scorer, feature_pca], feature_load_name = model_dict[folder + '_RGB_scoring']
                class_prediction = np.argmax(feature_classifier.predict(model_input))

                layer_name = [layer.name for layer in feature_classifier.layers][-2]
                intermediate_layer_model = keras.Model(inputs=feature_classifier.input,
                                                       outputs=feature_classifier.get_layer(layer_name).output)
                feature_vector = list(intermediate_layer_model.predict(model_input))[0]

                transformed_feat_vector = transform_and_append_prediction(feature_vector, feature_pca, class_prediction)

                score = feature_scorer.predict_proba(transformed_feat_vector)

                isGenuine = feature_scorer.predict(transformed_feat_vector)[0]
                transformed_score = apply_bounds(score_transform(score, int(isGenuine), feature_load_name))

                scores_list.append([transformed_score,transformed_score,transformed_score])
                index_order.append('_'.join(feat_path.split('_')[0:2]))


        for idx, method in enumerate(['edge', 'dct', 'fft']):
            scored_frame[folder + '_' + method + '_scores'] = [np.nan for i in range(len(scored_frame))]
            for ii, index_val in enumerate(index_order):
                scored_frame[folder + '_' + method + '_scores'].loc[index_val] = scores_list[ii][idx]
        scored_frame.to_csv('./' + folder + 'monday.csv')
if __name__ == '__main__':
    main()


