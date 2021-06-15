import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import random
print(f"TensorFlow version: {tf.__version__}")
# import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_y_data(y_path):
    y = np.loadtxt(y_path, dtype=np.int32).reshape(-1,1)
    # change labels range from 1-6 t 0-5, this enables a sparse_categorical_crossentropy loss function
    return y - 1

def load_X_data(X_path):
    X_signal_paths = [X_path + file for file in os.listdir(X_path)]
    X_signals = [np.loadtxt(path, dtype=np.float32) for path in X_signal_paths]
    return np.transpose(np.array(X_signals), (1, 2, 0))

def listToArray(listOfSegs):
    output = []
    for seg in listOfSegs:
        output.append(np.array(seg)[:,:,None])
    return output

def split_indices_per_label(y):
    unique_labels = len(np.unique(y))
    indicies_per_label = [[] for x in range(0, unique_labels)]
    # loop over the six labels
    for i in range(unique_labels):
        indicies_per_label[i] = np.where(np.array(y) == i)[0]
    return indicies_per_label


def plot_cwt_coeffs_per_label(X, label_indicies, label_names, signal, scales, wavelet):
    fig, axs = plt.subplots(nrows=2, ncols=6, sharex=True, sharey=True, figsize=(12, 5))
    listOfRand = [random.choice(range(10)), random.choice(range(10)) ]
    ii = 0
    sample = random.choice(range(10))
    sample = 6
    for ax, indices, name in zip(axs.flat, label_indicies*2, label_names*2):
        if ii == 6:
            sample = random.choice(range(10))
            sample = 5
        # apply  PyWavelets continuous wavelet transfromation function
        coeffs, freqs = pywt.cwt(X[indices[sample], :, signal], scales, wavelet=wavelet)
        #coeefs, freqs = pywt.dwt(X[indices[random.choice(range(10))], :, signal], wavelet=wavelet)
        # create scalogram
        ax.imshow(coeffs, cmap='coolwarm', aspect='auto')
        if name + str(sample) == 'Intaglio6' or name + str(sample) == 'Offset6':
            print(name + str(sample) + ': ' + str(indices[sample]))
            print(max(X[indices[sample], :, signal]))
        if name + str(sample) == 'Intaglio5' or name + str(sample) == 'Offset5':
            print(name + str(sample) + ': ' + str(indices[sample]))
            print(max(X[indices[sample], :, signal]))
        ax.set_title(name + str(sample))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
        ii += 1
    plt.tight_layout()
    plt.show()


def create_cwt_images(X, rescale_size, scales, wavelet_name="morl"):
    n_samples = X.shape[0]
    n_signals = X.shape[2]

    # range of scales from 1 to n_scales
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype='float32')

    for sample in range(n_samples):
        if sample % 1000 == 999:
            print('Converted {} Samples'.format(sample))
        for signal in range(n_signals):
            serie = X[sample, :, signal]
            # continuous wavelet transform
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
            #coeffs, freqs = pywt.dwt(serie, wavelet_name)
            # resize the 2D cwt coeffs
            rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode='constant')
            X_cwt[sample, :, :, signal] = rescale_coeffs

    return X_cwt

def build_cnn_model(activation, input_shape, num_classes):
    model = Sequential()

    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, 5, activation=activation, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
    model.add(MaxPooling2D())
    # model.add(Conv2D(128, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
    # model.add(MaxPooling2D())
    model.add(Flatten())

    # 3 Full connected layer
    model.add(Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_initializer="he_normal"))
    model.add(Dense(64, activation=activation, kernel_initializer="he_normal"))
    model.add(Dense(num_classes, activation='softmax'))  # 6 classes

    # summarize the model
    print(model.summary())
    return model


def compile_and_fit_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs, name):
    y_train = to_categorical(y_train)
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    # define callbacks
    callbacks = [
        ModelCheckpoint(filepath='best_model.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True)]

    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))

    model.save_weights('./hyperspec_models/' + name + '.h5')
    return model, history

