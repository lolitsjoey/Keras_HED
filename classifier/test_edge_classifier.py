import os
import keras
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from classifier.train_edge_classifier import classify_build, classify_build_conv
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.decomposition import PCA

def get_dense_output(dense_df, img_folder_to_classify, model, layer_name, imShape = (480,480), write = False, edge=False, dct=False, TEMP=False, lbp_class=None):
    predictions = []
    truth = []
    index_order = []
    for idx, img in enumerate(os.listdir(img_folder_to_classify)):
        if idx==0:
            print(img)
        test_img = cv2.resize(cv2.imread(img_folder_to_classify + img, 3), imShape)
        if TEMP:
            test_img = cv2.resize(cv2.imread(img_folder_to_classify + img, 0), imShape)
            hist = lbp_class.describe(test_img)
            template = np.zeros((1, len(hist)))
            template[0, :] = hist
            model_input = template[:, :, None]
            answer = model.predict(model_input)
        else:
            test_img = ((test_img - np.mean(test_img)) / np.std(test_img))
            if edge:
                test_img = cv2.GaussianBlur(test_img, (9, 9), 0)
            model_input = test_img[None, :, :, :]
            answer = model.predict(model_input)
        predictions.append(int(np.argmax(answer)))
        index_order.append('_'.join(img.split('_')[0:2]))

        if 'genuine' in img:
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


def get_dense_output_no_images(dense_df, array_to_classify, truth, model, layer_name, imShape = (480,480), binary = True, write = False, do_pca=False):
    predictions = []
    for idx,img in enumerate(array_to_classify):
        if do_pca:
            img = np.reshape(img, (-1, len(img), 1))
        else:
            img = img[None, :, :, :]

        answer = model.predict(img)
        if binary:
            predictions.append(answer[0][0])
        else:
            predictions.append(np.argmax(answer))
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(img)
        dense_df.iloc[idx,:] = list(intermediate_output)[0]

        if idx%100 == 99:
            print('Got predictions from {} Images...'.format(idx))

    dense_df['truth'] = truth
    dense_df['prediction'] = predictions
    if write:
        dense_df.to_csv('./intermediate_output.csv')

    return dense_df



def dendo_and_write_clusters_from_df(dfData2, weights_name, labs):
    weights_name = 'classify_conv_fed_seal.h5'
    weights_dir = './model_dir/classifiers/' + weights_name
    model = classify_build_conv(weights_dir)

    rgb_image_feat = 'D:/FRNLib/federal seals/all/'
    img_folder_to_classify = 'D:/FRNLib/federal seals/edges/'
    pca = PCA(0.98)
    pca.fit(dfData2)
    new_values = pca.transform(dfData2)
    links = linkage(new_values, 'ward')

    plt.figure(figsize=(15, 12))
    dendrogram(links, labels=labs)
    plt.show()

    cut = input('Where you cutting boi!?: ')

    clusterArray = cut_tree(links, height = int(cut))
    clusterArray2 = [(lab,x[0]) for lab,x in zip(image_names, clusterArray)]

    num_clusters = len(np.unique(clusterArray))
    if os.path.exists('./analysis pics/' + weights_name[0:-3] +'/'):
        shutil.rmtree('./analysis pics/' + weights_name[0:-3])

    for i in range(num_clusters):
        os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/analysis pics/' + weights_name[0:-3] + '/' + str(i) + '/rgb/', exist_ok= True)
        os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/analysis pics/' + weights_name[0:-3] + '/' + str(i) + '/edge/', exist_ok=True)

    for name, label in clusterArray2:
        rgbImage = cv2.resize(cv2.imread(rgb_image_feat + name[0:-10] + '.jpg'), (240, 240))
        edgeImage = cv2.imread(img_folder_to_classify + name)
        cv2.imwrite('./analysis pics/' + weights_name[0:-3] + '/' + str(label) + '/rgb/' + name[0:-10] + '.bmp', rgbImage)
        cv2.imwrite('./analysis pics/' + weights_name[0:-3] + '/' + str(label) + '/edge/' + name[0:-10] + '.bmp', edgeImage)


