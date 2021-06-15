import os
#from src.networks.hed_ultrafine import hed_refined
from src.networks.hed_reduced import hed
from skimage.feature import hog

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import shutil
import time 

def tool_this_folder(load_weights_from, test_images_in_this_folder, spit_edge_maps_here):
    model = hed(load_weights_from)
    #model = hed_refined(load_weights_from)
    for img in os.listdir(test_images_in_this_folder):
        if '_edges' in img:
            continue
        read_img = cv2.imread(test_images_in_this_folder + img, 1)
        try:
            resized_img = cv2.resize(read_img, (480, 480))
        except Exception:
            print(test_images_in_this_folder + img)
        predict_edge = model.predict(resized_img[None, :, :, :])
        cv2.imwrite(spit_edge_maps_here + img[0:-4] + '_edges.bmp', predict_edge[0][0, :, :, 0] * 255)

if __name__ == '__main__':
    with tf.device('CPU:0'):
        load_weights_from = './model_dir/fine_weights_probably_useless.h5'
        load_weights_from = './model_dir/weights_robust_lighting_texture.h5'

        test_images_in_this_folder = 'D:/trs_seal/'
        spit_edge_maps_here = 'D:/demo_edges/'
        print('WARNING REMOVING PREVIOUS EDGES IN {}'.format(spit_edge_maps_here))
        time.sleep(3)
        
        try:
            shutil.rmtree(spit_edge_maps_here)
            os.makedirs(spit_edge_maps_here)
        except FileNotFoundError:
            os.makedirs(spit_edge_maps_here)
            
        tool_this_folder(load_weights_from, test_images_in_this_folder, spit_edge_maps_here)
