import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import os
from PIL import Image, ImageDraw

def add_texture(img_dir,dest_dir):
    for shoe_img in os.listdir(img_dir):
        shoe = cv2.imread(img_dir + shoe_img)
        shoe = cv2.resize(shoe, (250,250))
        shoe = shoe.ravel()
        prc = random.choice(range(30,100))
        no_white = len(shoe[shoe == 255])
        prop_array = np.array(list(np.ones(255)) + [prc*255], dtype = int)
        shoe[shoe == 255] = random.choices(range(256), weights = prop_array, k = no_white)
        shoe = shoe.reshape((250,250,3))
        cv2.imwrite(dest_dir + shoe_img, shoe)

img_dir = './train_station/train_images/'
shoe = cv2.imread(img_dir + '10_AB.jpg')
shoe = cv2.resize(shoe, (250,250))
x = np.linspace(-125, 125, 250)
y = np.array([int(d) for d in x**2 + 2*x + 2])
y = y[y<250]
shoe[x + 125,y]
print('yay')
