import os
import cv2
import pandas as pd
from keras import backend

from test_edge_model import tool_this_folder

from classifier.train_edge_classifier import classify_build_conv, prepare_data, grab_batches_with_generator
from classifier.test_edge_classifier import get_dense_output
from classifier.score_from_network import score_notes_from_network, write_out_scores

load_tool_weights_from = './model_dir/weights_robust_lighting_texture.h5'
tool_images_in_this_folder = 'D:/FRNLib/federal seals/all/'
spit_tool_output_here = 'D:/FRNLib/federal seals/edges/'
new_tool_outputs = False

img_folder_to_classify = 'D:/FRNLib/federal seals/edges/'
img_folder_to_test_classifier = 'D:/FRNLib/federal seals/edges/'
load_classifier_weights_dir = './model_dir/classifiers/classify_conv_fed_seal.h5'
save_classifier_weights_to = './model_dir/classifiers/classify_conv_fed_seal_ttt.h5'
retrain_classifier = True

save_score_model_weights_to = './GAM_model/models/joey_ttt.pkl'
load_score_model_weights_dir = './gibberish.pkl'
retrain_scoremodel = True

rgb_image_feat = 'D:/FRNLib/federal seals/all/'
layer_name = 'dense_1'
batchSize = 20
epochs = 3


def edge(load_weights_dir, img_folder_to_classify):
    if new_tool_outputs:
        tool_this_folder(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here)
    x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(img_folder_to_classify)
    train_batches, test_batches, val_batches = grab_batches_with_generator(x_train, x_test, x_val,
                                                                           y_train, y_test, y_val, batchSize)
    model = classify_build_conv(load_weights_dir)
    return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches, model

def cwt():
    x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches =
    return x_train, x_test, x_val, y_train, y_test, y_val, train_batches, test_batches, val_batches


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
dense_output = get_dense_output(dense_output, img_folder_to_test_classifier, model, layer_name, write = True)

ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network(dense_output,
                                                                                    number_of_searches=10000,
                                                                                    save_name=save_score_model_weights_to)
write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, img_folder_to_test_classifier, rgb_image_feat)

