from __future__ import print_function
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
import random
import albumentations as A
import sys

def generate_minibatches(dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(range(len(dataParser.samples)), dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(range(len(dataParser.samples)), dataParser.batch_size_train*2)
        ims, ems, _ = dataParser.get_batch(batch_ids)
        yield(ims, [ems, ems, ems, ems])

def augmentImage(image, n):
    aug = A.Compose([
        # A.Flip(p=0.25),
        A.RandomGamma(gamma_limit=(20, 300), p=0.15),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=3, p=0.4),
        # A.RandomRotate90(p=0.25),
        # A.RGBShift(p=0.75),
        A.GaussNoise(p=0.12)
    ])

    outData = []
    for i in range(n):
        augmented = aug(image=image)
        outData.append(augmented['image'])
    return outData

def add_texture(img_dir,dest_dir):
    for shoe_img in os.listdir(img_dir):
        shoe = cv2.imread(img_dir + shoe_img)
        shoe = cv2.resize(shoe, (250,250))
        shoe = shoe.ravel()
        prc = random.choice(range(20,100))
        no_white = len(shoe[shoe == 255])
        prop_array = np.array(list(np.ones(255)) + [prc*255], dtype = int)
        shoe[shoe == 255] = random.choices(range(256), weights = prop_array, k = no_white)
        shoe = shoe.reshape((250,250,3))
        cv2.imwrite(dest_dir + shoe_img, shoe)

def test_results(testing_dir):
    for img in os.listdir(testing_dir):
        if '_edges' in img:
            continue
        read_img = cv2.imread(testing_dir + img,1)
        resized_img = cv2.resize(read_img, (480,480))
        predict_edge = model.predict(resized_img[None,:,:,:])

        cv2.imwrite(testing_dir + img + '_edges.bmp', predict_edge[0][0,:,:,0]*256)
######
if __name__ == "__main__":
    # params
    model_name = 'HEDSeg'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')
    folder_with_train_images = './train_station/train_images/'
    extra_texture_parent = './adv_l2l_train_station/'
    extra_texture_images_to = extra_texture_parent + 'train_images/'
    extra_texture_truth_to = extra_texture_parent + 'gnd_truth/'
    load_weights_from = './model_dir/weights_of_reduced_edge.h5' # change this to gibberish if you
                                                                 # wanna train from scratch
    save_model_weights_to = './model_dir/weights_robust_lighting_texture_2.h5'
    lst_with_folder_names = './HED-stuff/train_pair.lst'

    if os.path.exists(save_model_weights_to):
        response = input("Happy to overwrite {}?".format(save_model_weights_to.split))
        if response == 'y':
            pass
        else:
            sys.exit()

    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # prepare images
    response = input("New Input Images?")
    if response == 'y':
        add_texture(folder_with_train_images, extra_texture_images_to)
        for img in os.listdir(extra_texture_images_to):
            img_for_aug = cv2.imread(extra_texture_images_to + img)
            augList = augmentImage(img_for_aug, 7)
            for idx, augImage in enumerate(augList):
                cv2.imwrite(extra_texture_images_to + img[0:-4] + '_{}.bmp'.format(idx + 1), augImage)
                gnd_truth = cv2.imread(extra_texture_truth_to + img)
                cv2.imwrite(extra_texture_truth_to + img[0:-4] + '_{}.bmp'.format(idx + 1), gnd_truth)
    else:
        print('Continuing with existing training set: {}'.format(extra_texture_images_to))

    batch_size_train = 10
    dataParser = DataParser(batch_size_train, lst_with_folder_names, extra_texture_parent, extra_texture_images_to)

    # model
    model = hed(load_weights_from)
    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=2, batch_size = batch_size_train,
                                        write_graph=False, write_grads=True, write_images=False)
    train_history = model.fit_generator(
                        generate_minibatches(dataParser,),
                        # max_q_size=40, workers=1,
                        steps_per_epoch=dataParser.steps_per_epoch,  #batch size
                        epochs=4,
                        validation_data=generate_minibatches(dataParser, train=False),
                        validation_steps=dataParser.validation_steps,
                        callbacks=[checkpointer, csv_logger, tensorboard])

    print(train_history)
    model.save_weights(save_model_weights_to)








