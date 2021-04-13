from __future__ import print_function
import os
from src.utils.HED_data_parser import DataParser
from src.networks.hed import hed
import keras
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt


def generate_minibatches(dataParser, train=True):

    while True:
        if train:
            batch_ids = np.random.choice(range(len(dataParser.samples)), dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(range(len(dataParser.samples)), dataParser.batch_size_train*2)
        ims, ems, _ = dataParser.get_batch(batch_ids)
        yield(ims, [ems, ems, ems, ems, ems, ems])

######
if __name__ == "__main__":
    # params
    model_name = 'HEDSeg'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')

    batch_size_train = 5

    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    dataParser = DataParser(batch_size_train)

    # model
    model = hed()
    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=2, batch_size = batch_size_train,
                                        write_graph=False, write_grads=True, write_images=False)
    train_history = model.fit_generator(
                        generate_minibatches(dataParser,),
                        # max_q_size=40, workers=1,
                        steps_per_epoch=dataParser.steps_per_epoch,  #batch size
                        epochs=10,
                        validation_data=generate_minibatches(dataParser, train=False),
                        validation_steps=dataParser.validation_steps,
                        callbacks=[checkpointer, csv_logger, tensorboard])

    print(train_history)
    #model.save('./model_dir/trained_on_shoes')
    model.save_weights('./model_dir/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for img in os.listdir('./test_station/'):
        if '_edges' in img:
            continue
        read_img = cv2.imread('./test_station/' + img,1)
        resized_img = cv2.resize(read_img, (480,480))
        predict_edge = model.predict(resized_img[None,:,:,:])
        plt.imshow(resized_img)
        plt.show()
        plt.imshow(predict_edge[0][0,:,:,0])
        plt.show()
        cv2.imwrite('./test_station/' + img[0:-4] + '_edges.bmp', predict_edge[0][0,:,:,0]*256)        
    






