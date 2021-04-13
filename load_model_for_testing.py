import os
from src.utils.HED_data_parser import DataParser
from src.networks.hed_reduced import hed
import keras
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt

model = hed('./model_dir/weights_robust_lighting_texture.h5')
print(model.summary)
test_station = './adv_l2l_test_station/'
dest_dir = './adv_l2l_test_station/'

for img in os.listdir(test_station):
    if '_edges' in img:
        continue
    read_img = cv2.imread(test_station + img, 1)
    resized_img = cv2.resize(read_img, (480, 480))
    predict_edge = model.predict(resized_img[None, :, :, :])
    cv2.imwrite(dest_dir + img[0:-4] + '_edges.bmp', predict_edge[0][0, :, :, 0] * 256)
