import cv2
import os
from src.networks.hed_reduced import hed
from classifier.train_edge_classifier import classify_build_conv
import keras
import numpy as np
import pandas as pd
from classifier.score_from_network import score_notes_from_network
import tensorflow as tf
folder = 'D:/scoring_and_profiling/Feather/'
from auto_dct import loadModel

def loadImage(image):

    return cv2.resize(image, (64,64))/255.

with tf.device('CPU:0'):
    for feather in os.listdir(folder):
        '''
        edge_weights = './model_dir/weights_robust_lighting_texture.h5'
        model = hed(edge_weights)

        read_img = cv2.imread(folder + feather, 1)


        resized_img = cv2.resize(read_img, (480, 480))
        predict_edge = model.predict(resized_img[None, :, :, :])

        edge_map = predict_edge[0][0, :, :, 0]
        edge_map = ((edge_map - np.mean(edge_map)) / np.std(edge_map))
        edge_map = cv2.GaussianBlur(edge_map, (9, 9), 0)

        color_edge_map = cv2.cvtColor(edge_map,cv2.COLOR_GRAY2RGB)

        load_classifier_from = './score_models_per_feature/edge_Feather_classifier/edge.h5'
        class_model = classify_build_conv(load_classifier_from)

        prediction = class_model.predict(color_edge_map[None, :, :, :])
        print(prediction)
        layer_name = [layer.name for layer in class_model.layers][-2]
        intermediate_layer_model = keras.Model(inputs=class_model.input,
                                               outputs=class_model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(color_edge_map[None, :, :, :])
        intermediate_layer_model.summary()
        print(color_edge_map[None, :, :, :].shape)
        print(type(color_edge_map[0][0][0]))
        with open('./logfile.txt', 'w+') as output:
            output.write(str(intermediate_output))
        array_to_score = list(intermediate_output)[0]

        score_frame = pd.DataFrame(index = range(1), columns = range(20))
        score_frame.iloc[0,:] = array_to_score
        score_frame['truth'] = 77
        score_frame['prediction'] = np.argmax(prediction)

        load_score_model_from = './score_models_per_feature/edge_Feather'

        ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(score_frame,
                                                                                             num_class=2,
                                                                                             genuine_classes=[1],
                                                                                             pca_quality=0.99,
                                                                                             pca_splines=20,
                                                                                             pca_lam=0.2,
                                                                                             pred_splines=5,
                                                                                             pred_lam=0.2,
                                                                                             number_of_searches=4000,
                                                                                             load=True,
                                                                                             load_name=load_score_model_from,
                                                                                             save_name='gibber gab')
        print(ordered_scores)
        '''
        load_tool_weights_from = ['./model_dir/newlrencoderweights.h5', './model_dir/newlrdecoderweights.h5']
        model = loadModel(load_tool_weights_from)

        read_img = cv2.imread(folder + feather, 0)
        resized_img = np.expand_dims(np.array([loadImage(read_img)]), axis=-1)
        encodedImages = model.encoder(resized_img).numpy()
        decodedImages = model.decoder(encodedImages).numpy()
        resized_img = (decodedImages[0] * 255).astype("uint8")
        test_img = cv2.resize(cv2.cvtColor(resized_img,cv2.COLOR_GRAY2RGB), (480,480))
        test_img = ((test_img - np.mean(test_img)) / np.std(test_img))

        load_classifier_from = './score_models_per_feature/dct_Feather_classifier/dct.h5'
        class_model = classify_build_conv(load_classifier_from)

        prediction = class_model.predict(test_img[None, :, :, :])
        print(prediction)
        layer_name = [layer.name for layer in class_model.layers][-2]
        intermediate_layer_model = keras.Model(inputs=class_model.input,
                                               outputs=class_model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(test_img[None, :, :, :])
        intermediate_layer_model.summary()

        with open('./logfile.txt', 'w+') as output:
            output.write(str(intermediate_output))
        array_to_score = list(intermediate_output)[0]

        score_frame = pd.DataFrame(index=range(1), columns=range(20))
        score_frame.iloc[0, :] = array_to_score
        score_frame['truth'] = 77
        score_frame['prediction'] = np.argmax(prediction)

        load_score_model_from = './score_models_per_feature/dct_Feather'

        ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(score_frame,
                                                                                            num_class=2,
                                                                                            genuine_classes=[1],
                                                                                            pca_quality=0.99,
                                                                                            pca_splines=20,
                                                                                            pca_lam=0.2,
                                                                                            pred_splines=5,
                                                                                            pred_lam=0.2,
                                                                                            number_of_searches=4000,
                                                                                            load=True,
                                                                                            load_name=load_score_model_from,
                                                                                            save_name='gibber gab')
        print(ordered_scores)