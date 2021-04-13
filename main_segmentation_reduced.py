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
        A.RandomGamma(gamma_limit=(20, 300), p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.Rotate(limit=3, p=0.6),
        # A.RandomRotate90(p=0.25),
        # A.RGBShift(p=0.75),
        A.GaussNoise(p=0.25)
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
    img_dir = './train_station/train_images/'
    dest_dir = './adv_l2l_train_station/train_images/'
    weights_dir = './model_dir/weights_of_reduced_edge.h5'
    save_model_weights_to = './model_dir/weights_robust_lighting_texture.h5'

    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # prepare images
    response = input("New Input Images?")
    if response == 'y':
        add_texture('./train_station/train_images/', './adv_l2l_train_station/train_images/')
        for img in os.listdir(dest_dir):
            img_for_aug = cv2.imread(dest_dir + img)
            augList = augmentImage(img_for_aug, 7)
            for idx, augImage in enumerate(augList):
                cv2.imwrite(dest_dir + img[0:-4] + '_{}.bmp'.format(idx + 1), augImage)
                gnd_truth = cv2.imread('./adv_l2l_train_station/gnd_truth/' + img)
                cv2.imwrite('./adv_l2l_train_station/gnd_truth/' + img[0:-4] + '_{}.bmp'.format(idx + 1), gnd_truth)
    else:
        pass

    batch_size_train = 10
    dataParser = DataParser(batch_size_train, './HED-stuff/train_pair.lst', './adv_l2l_train_station/', './adv_l2l_train_station/train_images/')

    # model
    model = hed(weights_dir)
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








