import os
from src.networks.hed_reduced import hed
import cv2


def tool_this_folder(load_weights_from, test_images_in_this_folder, spit_edge_maps_here):
    model = hed(load_weights_from)
    for img in os.listdir(test_images_in_this_folder):
        if '_edges' in img:
            continue
        read_img = cv2.imread(test_images_in_this_folder + img, 1)
        resized_img = cv2.resize(read_img, (480, 480))
        predict_edge = model.predict(resized_img[None, :, :, :])
        cv2.imwrite(spit_edge_maps_here + img[0:-4] + '_edges.bmp', predict_edge[0][0, :, :, 0] * 256)
