import os
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dense, Flatten
from utils.MM_data_parser import DataParser
from src.networks.hed_reduced import hed
import keras
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
from keras import Model
import numpy as np
import shutil
import pdb
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from classifier import classify_build, classify_build_conv
import pandas as pd
from sklearn.cluster        import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.decomposition import PCA

weights_name = 'classify_conv_median_blur.h5'
weights_dir = './model_dir/classifiers/classify_conv_median_blur.h5'

model = classify_build_conv(weights_dir)
train_station = './pl_train_station/pretty_good_set_input/'
test_station = './pl_train_station/output_hed_images/'
preds = []
img_names = []
dfData = pd.DataFrame(columns = range(169), index = range(len(os.listdir(test_station))))
dfData2 = pd.DataFrame(columns = range(20), index = range(len(os.listdir(test_station))))
predictions = []
labs = []
truth = []
for idx,img in enumerate(os.listdir(test_station)):
    test_img = cv2.resize(cv2.imread(test_station + img), (480,480))
    if 'genuine' in img:
        predictions.append(img)
        labs.append(img[0:4])
        truth.append(1)
    else:
        predictions.append(img)
        labs.append(img[0:4])
        truth.append(0)
    answer = model.predict(test_img[None, :, :, :])
    preds.append(answer)
    img_names.append(img)
    layer_name = 'dense'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(test_img[None, :, :, :])
    dfData.iloc[idx,:] = list(intermediate_output)[0]
    layer_name = 'dense_1'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(test_img[None, :, :, :])
    dfData2.iloc[idx,:] = list(intermediate_output)[0]

dfData2['truth'] = truth
dfData2.to_csv('./intermediate_output.csv')
pca = PCA(0.98)
pca.fit(dfData)
new_values = pca.transform(dfData)
#links = linkage(dfData.T.values, 'ward')
links = linkage(new_values, 'ward')
plt.figure(figsize=(15, 12))
dendrogram(links, labels = labs)
plt.show()

cut = input('Where you cutting boi!?')

clusterArray = cut_tree(links, height = int(cut))
clusterArray2 = [(lab,x[0]) for lab,x in zip(predictions,clusterArray)]

num_clusters = len(np.unique(clusterArray))
if os.path.exists('./analysis pics/' + weights_name[0:-3]):
    shutil.rmtree('./analysis pics/' + weights_name[0:-3])

for i in range(num_clusters):
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/analysis pics/' + weights_name[0:-3] + '/' + str(i) + '/rgb/', exist_ok= True)
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/analysis pics/' + weights_name[0:-3] + '/' + str(i) + '/edge/', exist_ok=True)


for name, label in clusterArray2:
    rgbImage = cv2.resize(cv2.imread(train_station + name[0:-10] + '.bmp'), (480,480))
    edgeImage = cv2.imread(test_station + name)
    cv2.imwrite('./analysis pics/' + weights_name[0:-3] + '/' + str(label) + '/rgb/' + name[0:-10] + '.bmp', rgbImage)
    cv2.imwrite('./analysis pics/' + weights_name[0:-3] + '/' + str(label) + '/edge/' + name[0:-10] + '.bmp', edgeImage)




'''
links = linkage(dfData2.T.values, 'ward')
plt.figure(figsize=(15, 12))
dendrogram(links, labels = labs)
plt.show()
'''
