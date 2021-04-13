# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:27:40 2021

@author: joeba
"""
import os
from src.utils.HED_data_parser import DataParser
from src.networks.hed import hed
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt



if __name__ == "__main__":
    #environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir('./model_dir/'): os.makedirs('./model_dir/')
    
    model = hed()
    
    for image in os.listdir('./test_images/'):
        name = image.split('/')[-1]
        x_batch = []
        im = Image.open('./test_images/' + image)
        im = im.resize((480,480))
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32)
        prediction = model.predict(x_batch)
        for pred in range(len(prediction)):
            plt.imshow(prediction[pred][0,:,:,0])
            plt.show()
        mask = np.zeros_like(im[:,:,0])
        for i in range(len(prediction)):
            mask += np.reshape(prediction[i],(480,480))
        ret,mask = cv2.threshold(mask,np.mean(mask)+1.2*np.std(mask),255,cv2.THRESH_BINARY)
        cv2.imwrite("output/%s"%name,mask)