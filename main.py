import os
import cv2
import pandas as pd
import numpy as np
from keras import backend

from test_edge_model import tool_this_folder

from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output, get_dense_output_no_images
from classifier.score_from_network import score_notes_from_network, write_out_scores, write_out_scores_noimages

from hyperspec_classifier.hyperspec_classifier import get_batches, compile_and_fit_model
from hyperspec_classifier.hyperspec_tools import build_cnn_model
from hyperspec_classifier.hyperspec_writer import write_hyper_spec

load_tool_weights_from = './model_dir/weights_robust_lighting_texture.h5'
tool_images_in_this_folder = 'D:/FRNLib/federal seals/all/'
spit_tool_output_here = 'D:/FRNLib/federal seals/edges/'
new_tool_outputs = False

img_folder_to_classify = 'D:/FRNLib/federal seals/edges/'
img_folder_to_test_classifier = 'D:/FRNLib/federal seals/edges/'

'''# EDGE ---
load_classifier_weights_dir = './model_dir/classifiers/classify_conv_fed_seal.h5'
save_classifier_weights_to = './model_dir/classifiers/classify_conv_fed_seal_ttt.h5'
'''# EDGE ---

#''' # CWT ---
#load_classifier_weights_dir = './hyperspec_classifier/learning_to_learn_save.h5'
load_classifier_weights_dir = './hyperspec_classifier/gibberish.h5'
save_classifier_weights_to = './hyperspec_classifier/learning_to_learn_cwt_inkwell.h5'
#''' #CWT ---


retrain_classifier = True

save_score_model_weights_to = './GAM_model/models/joey_cwt_inkwell_score'
load_score_model_weights_dir = './gibberish.pkl'
retrain_scoremodel = True

rgb_image_feat = 'D:/FRNLib/federal seals/all/'
layer_name = 'dense_1'
batchSize = 10
epochs = 40


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


def edge(load_weights_dir, img_folder_to_classify):
    if new_tool_outputs:
        tool_this_folder(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
    x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify)
    train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                           y_train, y_test, y_val, batchSize)
    model = classify_build_conv(load_weights_dir)
    return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model


#x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model = edge(load_classifier_weights_dir, img_folder_to_classify)
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

n_dense_neurons = backend.int_shape(model.get_layer(layer_name).output)[1]
'''
dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(os.listdir(img_folder_to_test_classifier))))
dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(os.listdir(img_folder_to_test_classifier))))
dense_output = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name, write = True)
ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                    number_of_searches=40000,
                                                                                    load= not retrain_scoremodel,
                                                                                    save_name=save_score_model_weights_to)
write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, images_scored=img_folder_to_test_classifier, rgb_dir=rgb_image_feat)
'''

#'''
x_all = np.vstack( (x_train, x_val, x_test) )
y_all = np.hstack( (y_train, y_val, y_test) )
dense_output = pd.DataFrame(columns=range(n_dense_neurons), index=range(len(x_all)))
dense_output = get_dense_output_no_images(dense_output, x_all, y_all, model, layer_name, binary = False, write = True)

ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                    genuine_classes = [0,1,2,3],
                                                                                    load= not retrain_scoremodel,
                                                                                    number_of_searches=10000,
                                                                                    save_name=save_score_model_weights_to)
#write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, genuine_classes = [0,1,2,3],images_scored=hyperspec_destination + 'cwt/', rgb_dir=hyperspec_destination + 'rgb/')
write_out_scores_noimages(ordered_scores, ordered_labels, arguments, ordered_truth, array_scored=x_all, genuine_classes = [0,1,2,3])
#'''