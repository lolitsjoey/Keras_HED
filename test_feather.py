import cv2
import os
from src.networks.hed_reduced import hed
from classifier.train_edge_classifier import classify_build_conv
import keras
import numpy as np
import pandas as pd
from classifier.score_from_network import score_notes_from_network

folder = 'D:/scoring_and_profiling/Feather/'

for feather in os.listdir(folder):
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
    
    layer_name = [layer.name for layer in class_model.layers][-2]
    intermediate_layer_model = keras.Model(inputs=class_model.input,
                                           outputs=class_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(color_edge_map[None, :, :, :])
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