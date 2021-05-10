import tensorflow as tf
import keras
from keras.models import Model
from keras import layers, losses
import numpy as np
import os
import cv2

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(128, 128, 1)),
      layers.Conv2D(16, (4, 4), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (4, 4), activation='relu', padding='same', strides=2),
      layers.Conv2D(4, (2, 2), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(4, kernel_size=2, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(8, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(4, 4), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def loadImage(filePath):

    return cv2.resize(cv2.imread(filePath, cv2.IMREAD_GRAYSCALE), (64,64))/255.

def loadTestData(testInputPath):

    inputImageSet = [(testInputPath+file) for file in os.listdir(testInputPath)]
    files = [file for file in os.listdir(testInputPath)]
    print(len(inputImageSet))

    inputImages = np.expand_dims(np.array([loadImage(str(file_name)) for file_name in inputImageSet]), axis=-1)

    print('Printing dimensions of test image arrays...')
    print(inputImages.shape)

    return inputImages, files

def loadModel(load_tool_weights_from):

    loadedModel = Denoise()
    loadedModel.build((32000, 128, 128, 1))
    loadedModel.encoder.load_weights(load_tool_weights_from[0])
    loadedModel.decoder.load_weights(load_tool_weights_from[1])

    return loadedModel

def evaluateModel(load_tool_weights_from, tool_images_in_this_folder, spit_tool_output_here):

    model = loadModel(load_tool_weights_from)
    model.summary()
    inputTestData, names = loadTestData(tool_images_in_this_folder)
    encodedImages = model.encoder(inputTestData).numpy()
    decodedImages = model.decoder(encodedImages).numpy()

    for i in range(len(decodedImages)):
        cv2.imwrite(spit_tool_output_here + names[i][0:-4] + '_AEfft.bmp', (decodedImages[i] * 255).astype("uint8"))


    print('Model evaluation complete')

    return



